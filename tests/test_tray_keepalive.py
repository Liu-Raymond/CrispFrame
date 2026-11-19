"""托盘保活测试：任务运行中关窗=隐藏保活；空闲关窗=退出；托盘退出=软停（分片保留）。

offscreen（无真实托盘 → _tray=None 走最小化分支，逻辑同构）。"""
import os
import shutil
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QEvent, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication(sys.argv[:1])

import crispframe.archive_job as aj
from archive_page import ArchivePage
from crispframe.codec import CodecSettings
from crispframe.ffutils import VideoInfo
from crispframe.pipeline import PassthroughCanceled
from gui import APP_TITLE, TrayMainWindow

fails = []


def check(name, cond):
    print(f"{'✓' if cond else '✗'} {name}", flush=True)
    if not cond:
        fails.append(name)


def main() -> int:
    SRC = Path("tmp/keepalive_src.mp4")
    if not SRC.exists():
        import subprocess
        from crispframe.ffutils import ffmpeg
        subprocess.run([ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=40",
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                        "-pix_fmt", "yuv420p", str(SRC)], check=True)
    info = VideoInfo(SRC, 640, 360, 24.0, 960, 40.0)

    page = ArchivePage()
    win = TrayMainWindow(page)
    win.show()
    print(f"托盘可用（offscreen 通常不可用→走最小化分支）: {win._tray is not None}")

    # ---- 场景 A：空闲关窗 → 正常关闭（事件被接受） ----
    ev = QCloseEvent()
    win.closeEvent(ev)
    check("A 空闲关窗事件被接受", ev.isAccepted())

    # ---- 场景 B：任务运行中关窗 → 隐藏保活 ----
    cs = CodecSettings()
    cs.crf, cs.container, cs.bit_depth = 30, "matroska", 8
    worker = archive_page_ArchiveWorker(cs, SRC, info, page)
    page.worker = worker                     # 模拟 _start 后状态（不真 start）
    ev = QCloseEvent()
    win.closeEvent(ev)
    check("B 运行中关窗事件被忽略（保活）", not ev.isAccepted())
    check("B 窗口已收起（隐藏或最小化，均保活）",
          win.isHidden() or win.isMinimized())
    win._show_window()
    check("B 显示窗口可恢复", win.isVisible())

    # ---- 场景 C：托盘退出软停——真实线程：段 1 编码中 halt ----
    done_msg: list[str] = []
    worker.allDone.connect(lambda ok, m: done_msg.append(m))
    worker.start()
    # 等当前段 ffmpeg 起来（_proc 非空）或最多 10s
    for _ in range(100):
        if worker._proc is not None:
            break
        time.sleep(0.1)
    fd_msg: list[str] = []
    worker.fileDone.connect(lambda n, ok, m: fd_msg.append(m))
    worker.halt()                            # 软停：terminate + 跳过剩余
    # 跨线程信号走主线程事件队列——必须泵 processEvents 才会投递
    for _ in range(600):                     # 最长 60s，泵到 allDone 到达
        APP.processEvents()
        if done_msg:
            break
        time.sleep(0.1)
    worker.wait(10_000)
    APP.processEvents()
    print("   [debug] running:", worker.isRunning(), "done_msg =", done_msg,
          "fileDone =", fd_msg, "halt =", worker._halt)
    check("C 软停走暂停路径（分片保留）", parts_dir_exists(SRC, cs))
    check("C allDone 消息为已停止", any("已停止" in m for m in done_msg))
    check("C 无成品残件", not (Path("tmp") / "keepalive_src.cf.archive-crf30.mkv").exists())

    # 清理
    worker._proc = None
    d = aj_parts_dir(SRC, cs)
    if d.exists():
        for f in d.iterdir():
            f.unlink(missing_ok=True)
        d.rmdir()
    SRC.unlink(missing_ok=True)
    print()
    print("全部通过" if not fails else f"失败 {len(fails)} 项: {fails}")
    return 1 if fails else 0


def archive_page_ArchiveWorker(cs, src, info, page):
    from archive_page import ArchiveWorker
    return ArchiveWorker([(src, info)], cs.crf, cs.container)


def parts_dir_exists(src, cs):
    from crispframe.archive_job import parts_dir
    return parts_dir(src, cs).exists()


def aj_parts_dir(src, cs):
    from crispframe.archive_job import parts_dir
    return parts_dir(src, cs)


if __name__ == "__main__":
    raise SystemExit(main())
