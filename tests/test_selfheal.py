"""存档编码自愈逻辑单测（monkeypatch run_passthrough，无真实编码）。

覆盖 _encode_selfheal（GUI 存档页）的四个路径：
  1. SilentTermination → 自动重试一次 → 成功
  2. SilentTermination 两次 → 报错（消息含"外部终止"人话）
  3. SVT 8K 拒绝 → preset 自动上调 5 重试 → 成功
  4. 其他错误 → 不重试，原样报错
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QCoreApplication

APP = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])

from crispframe.ffutils import VideoInfo
from crispframe.pipeline import SilentTermination
import archive_page

SRC = Path("dummy.mp4")
INFO = VideoInfo(SRC, 3840, 2160, 60.0, 0, 10.0)
fails = []


def check(name, cond):
    print(f"{'✓' if cond else '✗'} {name}")
    if not cond:
        fails.append(name)


def make_worker():
    logs: list[str] = []
    w = archive_page.ArchiveWorker([(SRC, INFO)], crf=30, container="mp4")
    w.log.connect(logs.append)
    return w, logs


def patch_runpassthrough(scheduler):
    """scheduler(attempt_no) -> None(成功) | Exception。记录每次 encode 的 preset。"""
    calls = {"presets": [], "n": 0}

    def fake(info, out, encode=None, **kw):
        args = encode.args
        calls["presets"].append(args[args.index("-preset") + 1])
        e = scheduler(calls["n"])
        calls["n"] += 1
        if e is not None:
            raise e
    archive_page.run_passthrough = fake
    return calls


def main() -> int:
    out = Path("dummy_out.mkv")
    orig_run = archive_page.run_passthrough

    # 1. 静默终止 → 重试 → 成功
    calls = patch_runpassthrough(lambda n: SilentTermination("外部终止") if n == 0 else None)
    w, logs = make_worker()
    r = w._encode_selfheal(SRC, INFO, out, archive_page.CodecSettings())
    check("静默终止重试后成功", r == "ok")
    check("共调了两次", calls["n"] == 2)
    check("有自动重试日志", any("被外部终止" in s and "重试" in s for s in logs))
    check("两次 preset 不变（4）", calls["presets"] == ["4", "4"])

    # 2. 静默终止 ×2 → 报错
    calls = patch_runpassthrough(lambda n: SilentTermination("外部终止"))
    w, logs = make_worker()
    r = w._encode_selfheal(SRC, INFO, out, archive_page.CodecSettings())
    check("两次静默终止后报错", r.startswith("编码失败"))
    check("报错含人话原因", any("外部结束" in s or "外部终止" in s for s in logs))
    check("只调了两次（不多试）", calls["n"] == 2)

    # 3. SVT 8K 拒绝 → preset 升 5 重试
    calls = patch_runpassthrough(
        lambda n: Exception("[libsvtav1 @ x] Error setting encoder parameters: "
                            "bad parameter (0x80001005)") if n == 0 else None)
    w, logs = make_worker()
    r = w._encode_selfheal(SRC, INFO, out, archive_page.CodecSettings())
    check("SVT 8K 升档重试后成功", r == "ok")
    check("第二次 preset=5", calls["presets"] == ["4", "5"])

    # 4. 其他错误 → 不重试
    calls = patch_runpassthrough(lambda n: RuntimeError("磁盘满了"))
    w, logs = make_worker()
    r = w._encode_selfheal(SRC, INFO, out, archive_page.CodecSettings())
    check("其他错误不重试", r.startswith("编码失败") and calls["n"] == 1)

    archive_page.run_passthrough = orig_run
    print()
    print("全部通过" if not fails else f"失败 {len(fails)} 项: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
