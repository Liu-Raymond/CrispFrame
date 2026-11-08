"""分段存档测试：切片→合并→验证、暂停续跑、manifest 失配、放弃清理。

真实 ffmpeg 小负载（40s 源 / 15s 段），不依赖 GUI。
"""
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crispframe.archive_job import parts_dir, run_segmented
from crispframe.codec import CodecSettings
from crispframe.ffutils import ffmpeg, probe
from crispframe.pipeline import PassthroughCanceled

SRC = Path("tmp/segtest_src.mp4")
OUT = Path("tmp/segtest_out.cf.archive-crf30.mkv")
SEG = 15
fails = []


def check(name, cond):
    print(f"{'✓' if cond else '✗'} {name}", flush=True)
    if not cond:
        fails.append(name)


def make_src() -> None:
    if SRC.exists():
        return
    subprocess.run([ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=40",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=40",
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
                    "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(SRC)],
                   check=True)


def make_cs() -> CodecSettings:
    cs = CodecSettings()
    cs.crf, cs.container, cs.bit_depth = 30, "matroska", 8
    return cs


def main() -> int:
    make_src()
    info = probe(SRC)
    cs = make_cs()
    pd = parts_dir(SRC, cs)
    logs: list[str] = []
    logs_lock = threading.Lock()

    # ---- 场景 1：完整分段存档 ----
    if OUT.exists():
        OUT.unlink()
    if pd.exists():
        shutil.rmtree(pd)
    r = run_segmented(info, SRC, OUT, cs, seg_secs=SEG,
                      log_cb=lambda m: (logs_lock.acquire(),
                                        logs.append(m),
                                        print("   ", m),
                                        logs_lock.release()))
    check("场景1 全流程 ok", r == "ok")
    check("场景1 成品存在", OUT.exists())
    check("场景1 分片目录已清", not pd.exists())
    check("场景1 三段日志", any("1/3" in m for m in logs) and any("3/3" in m for m in logs))

    # ---- 场景 2：第 2 段暂停 → 分片保留 → 续跑 ----
    OUT.unlink(missing_ok=True)
    shutil.rmtree(pd, ignore_errors=True)
    state = {"pause_now": False}

    def pause_when_seg2():
        return state["pause_now"]

    def log_pause_watch(m: str):
        print("   ", m)
        if "段 1 完成" in m:      # 段 1 编完即置位 → 段 2 顶部暂停（分片=1）
            state["pause_now"] = True

    r = run_segmented(info, SRC, OUT, cs, seg_secs=SEG,
                      pause_check=pause_when_seg2,
                      log_cb=log_pause_watch)
    check("场景2 返回 pause", r == "pause")
    check("场景2 分片目录保留", pd.exists())
    parts = sorted(pd.glob("part*.mkv"))
    check("场景2 只有段 1 完成", len(parts) == 1)
    check("场景2 manifest 在", (pd / "manifest.json").exists())

    # 续跑（无暂停）
    r = run_segmented(info, SRC, OUT, cs, seg_secs=SEG,
                      log_cb=lambda m: print("   ", m))
    check("场景2 续跑完成", r == "ok")
    check("场景2 成品存在且分片清", OUT.exists() and not pd.exists())

    # ---- 场景 3：manifest 失配（改 CRF）→ 忽略旧分片 ----
    OUT.unlink(missing_ok=True)
    pd.mkdir(parents=True, exist_ok=True)
    (pd / "part0000.mkv").write_bytes(b"junk")
    (pd / "manifest.json").write_text(json.dumps({"crf": 99}), encoding="utf-8")
    cs2 = make_cs()
    r = run_segmented(info, SRC, OUT, cs2, seg_secs=SEG,
                      log_cb=lambda m: print("   ", m))
    check("场景3 失配后仍完成", r == "ok")
    check("场景3 分片清理", not pd.exists())

    # ---- 场景 4：放弃（cancel 在段 2 触发）→ 分片删除 + PassthroughCanceled ----
    OUT.unlink(missing_ok=True)
    shutil.rmtree(pd, ignore_errors=True)
    cancel_state = {"go": False}

    def cancel_when_seg2():
        return cancel_state["go"]

    def log_cancel_watch(m: str):
        print("   ", m)
        if "段 1 完成" in m:
            cancel_state["go"] = True

    try:
        run_segmented(info, SRC, OUT, cs, seg_secs=SEG,
                      cancel_check=cancel_when_seg2,
                      log_cb=log_cancel_watch)
        check("场景4 抛出 PassthroughCanceled", False)
    except PassthroughCanceled:
        check("场景4 抛出 PassthroughCanceled", True)
    check("场景4 分片已清", not pd.exists())
    check("场景4 无成品残件", not OUT.exists())

    # ---- 场景 5：段数多时分段时长漂移（音频包量化累积）——钳制后必须过严格验证 ----
    OUT.unlink(missing_ok=True)
    shutil.rmtree(pd, ignore_errors=True)
    r = run_segmented(info, SRC, OUT, cs, seg_secs=3,     # 40s/3s = 14 段
                      log_cb=lambda m: print("   ", m))
    check("场景5 多段（14）存档 ok", r == "ok")
    from crispframe.verify import verify_output
    vr = verify_output(OUT, SRC, strict_metadata=True)
    for line in vr.lines + vr.failures:
        print("   ", line)
    check("场景5 严格验证通过（漂移被 -t 钳制）", vr.ok)

    # 清理
    SRC.unlink(missing_ok=True)
    OUT.unlink(missing_ok=True)
    shutil.rmtree(pd, ignore_errors=True)
    print()
    print("全部通过" if not fails else f"失败 {len(fails)} 项: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
