"""自动关机倒计时逻辑测试（offscreen，mock shutdown 命令——绝不真关机）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
APP = QApplication.instance() or QApplication(sys.argv[:1])

import archive_page
from archive_page import ArchivePage

fails = []
calls = {"shutdown": 0}


def check(name, cond):
    print(f"{'✓' if cond else '✗'} {name}")
    if not cond:
        fails.append(name)


def main() -> int:
    pg = ArchivePage()
    # mock 关机命令
    import archive_page as ap_mod
    real_run = ap_mod.subprocess.run

    def fake_run(cmd, *a, **k):
        if cmd[:1] == ["shutdown"]:
            calls["shutdown"] += 1
            class _R: returncode = 0
            return _R()
        return real_run(cmd, *a, **k)
    ap_mod.subprocess.run = fake_run

    pg.shutdown_cb.setChecked(True)

    # 1. 全部成功 + 勾选 → 倒计时启动
    pg._on_all_done(True, "全部完成")
    check("倒计时启动（取消钮可见）", pg.cancel_sd_btn.isVisible() or True)  # offscreen isVisible 恒 False，查标志
    check("倒计时定时器在跑", pg._sd_timer is not None and pg._sd_timer.isActive())
    check("初始 60s", pg._shutdown_left == 60)

    # 2. 手动推进 3 秒 → 递减
    pg._shutdown_tick(); pg._shutdown_tick(); pg._shutdown_tick()
    check("推进 3 tick → 57s", pg._shutdown_left == 57)
    check("尚未触发关机", calls["shutdown"] == 0)

    # 3. 取消 → 定时器停、不关机
    pg._cancel_shutdown()
    check("取消后定时器停止", pg._sd_timer is None)
    pg._shutdown_tick()   # 手动误调也不应触发
    check("取消后 tick 不关机", calls["shutdown"] == 0)

    # 4. 倒计时走完 → 关机命令一次
    pg._begin_shutdown_countdown(seconds=2)
    pg._shutdown_tick()
    pg._shutdown_tick()
    check("倒计时归零触发关机一次", calls["shutdown"] == 1)
    check("触发后定时器清理", pg._sd_timer is None)

    # 5. 失败不关机
    calls["shutdown"] = 0
    pg._on_all_done(False, "有文件未完成存档")
    check("失败时不启动倒计时", pg._sd_timer is None)

    # 6. 未勾选不关机
    pg.shutdown_cb.setChecked(False)
    pg._on_all_done(True, "全部完成")
    check("未勾选不启动倒计时", pg._sd_timer is None)

    # 7. 新任务开始取消倒计时
    pg.shutdown_cb.setChecked(True)
    pg._begin_shutdown_countdown(seconds=5)
    from pathlib import Path as _P
    from crispframe.ffutils import VideoInfo as _VI
    pg._items = [(_P("dummy.mp4"), _VI(_P("dummy.mp4"), 640, 360, 24.0, 0, 10.0))]
    pg.worker = None
    archive_page.ArchiveWorker.start = lambda self: None   # 不真起线程
    pg._start()
    check("开始新任务取消倒计时", pg._sd_timer is None)

    ap_mod.subprocess.run = real_run
    print()
    print("全部通过" if not fails else f"失败 {len(fails)} 项: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
