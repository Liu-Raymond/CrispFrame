"""存档自愈逻辑测试：run_segmented 的段级重试（SVT 8K 升档 / 静默终止）。

monkeypatch archive_job.run_passthrough，无真实编码（GUI/CLI 共用该编排）。"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QCoreApplication

APP = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])

from crispframe.codec import CodecSettings
from crispframe.ffutils import VideoInfo
from crispframe.pipeline import SilentTermination
from crispframe.archive_job import parts_dir
import crispframe.archive_job as aj

SRC = Path("dummy_src.mp4")
OUT = Path("dummy_out.cf.archive-crf30.mkv")
INFO = VideoInfo(SRC, 640, 360, 24.0, 240, 10.0)   # 10s 源 → 默认 1 段
SVT_8K_ERR = ("[libsvtav1 @ x] Error setting encoder parameters: "
              "bad parameter (0x80001005)")

fails = []


def check(name, cond):
    print(f"{'✓' if cond else '✗'} {name}")
    if not cond:
        fails.append(name)


def patch_run(scheduler):
    """scheduler(attempt) -> None(成功) | Exception；记录每次 encode 的 preset。

    同时接管合并步骤（subprocess.run）：重试逻辑单测不需要真编码/真 concat，
    fake 只落一个占位段文件让编排流程走通。"""
    calls = {"presets": [], "n": 0, "merge": 0}

    class _R:
        returncode = 0
        stderr = ""

    def fake(info, out_path, encode=None, **kw):
        args = encode.args
        calls["presets"].append(args[args.index("-preset") + 1])
        e = scheduler(calls["n"])
        calls["n"] += 1
        if e is not None:
            raise e
        Path(out_path).write_bytes(b"fake-part")

    real_run = aj.subprocess.run

    def fake_run(cmd, *a, **k):
        if cmd and str(cmd[0]).lower().endswith("ffmpeg.exe") and "concat" in cmd:
            calls["merge"] += 1
            return _R()
        return real_run(cmd, *a, **k)

    aj.run_passthrough = fake
    aj.subprocess.run = fake_run
    calls["restore"] = (fake, real_run)
    return calls


def run_one(cs: CodecSettings):
    cs = cs or CodecSettings()          # 与调用方（worker/cli）同款：基线+调参
    logs: list[str] = []
    r = aj.run_segmented(INFO, SRC, OUT, cs, seg_secs=600,
                         log_cb=logs.append)
    return r, logs


def main() -> int:
    orig = aj.run_passthrough
    OUT.unlink(missing_ok=True)
    SRC.write_bytes(b"fake")   # fingerprint 需要 stat 源文件（真实调用方保证存在）

    # 1. 静默终止 → 自动重试本段 → 成功
    calls = patch_run(lambda n: SilentTermination("外部终止") if n == 0 else None)
    r, logs = run_one(None)
    check("静默终止重试后 ok", r == "ok")
    check("共 2 次调用", calls["n"] == 2)
    check("有自动重试日志", any("外部终止" in s and "重试" in s for s in logs))
    OUT.unlink(missing_ok=True)

    # 2. 静默终止 ×2 → fail（分片清理在编排内）
    calls = patch_run(lambda n: SilentTermination("外部终止"))
    r, logs = run_one(None)
    check("两次静默终止 → fail", r == "fail")
    check("只重试一次（2 次调用）", calls["n"] == 2)
    check("报错含人话", any("外部终止" in s for s in logs))
    OUT.unlink(missing_ok=True)

    # 3. SVT 8K 拒绝 → preset 升 5 重试
    calls = patch_run(lambda n: Exception(SVT_8K_ERR) if n == 0 else None)
    r, logs = run_one(None)
    check("SVT 8K 升档重试后 ok", r == "ok")
    check("第二次 preset=5", calls["presets"] == ["4", "5"])
    OUT.unlink(missing_ok=True)

    # 4. 其他错误 → 不重试
    calls = patch_run(lambda n: RuntimeError("磁盘满了"))
    r, logs = run_one(None)
    check("其他错误不重试", r == "fail" and calls["n"] == 1)
    OUT.unlink(missing_ok=True)

    # 5. 暂停分支：编码中 terminate（GUI pause 按钮 → proc.terminate →
    #    SilentTermination）+ pause_check → 残段丢、不重试、直接 pause
    calls = patch_run(lambda n: SilentTermination("外部终止"))
    state = {"go": False}

    def pause_during_encode():
        return state["go"]

    def log_pause_watch(m: str):
        if "段 1/1" in m:      # 编码开始后（GUI 时间线：用户此刻点暂停）
            state["go"] = True

    logs = []
    r = aj.run_segmented(INFO, SRC, OUT, CodecSettings(), seg_secs=600,
                         pause_check=pause_during_encode,
                         log_cb=log_pause_watch)
    check("terminate 暂停分支（不重试直接 pause）",
          r == "pause" and calls["n"] == 1)
    check("暂停分支残段已删（不冒充完成段）", not any("part0000" in m and "完成" in m
                                                for m in logs))
    OUT.unlink(missing_ok=True)
    # 暂停路径按设计保留分片目录——测试收尾自己清，防仓库根遗留
    shutil.rmtree(parts_dir(SRC, CodecSettings()), ignore_errors=True)

    fake, real_run = calls["restore"]
    aj.run_passthrough = orig
    aj.subprocess.run = real_run
    SRC.unlink(missing_ok=True)
    print()
    print("全部通过" if not fails else f"失败 {len(fails)} 项: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
