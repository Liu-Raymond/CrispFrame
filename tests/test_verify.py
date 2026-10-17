"""verify.py 三项验证的独立单测（运行时 ffmpeg 合成媒体，不依赖 tests/ 素材）。

运行：.venv/Scripts/python.exe tests/test_verify.py
前提：ffmpeg/ffprobe 在 PATH 或 winget Gyan.FFmpeg 位置（ffutils 自动定位）。
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crispframe.ffutils import ffmpeg
from crispframe.verify import verify_output


def _make_src(dirpath: Path) -> Path:
    """合成 1 视频 + 2 音轨 + 1 字幕 + 章节的 ~4 秒 MKV。

    字幕必须写满片长：-shortest 会以最短流截断整片（曾致截断用例无偏差可检）。"""
    srt = dirpath / "subs.srt"
    srt.write_text(
        "1\n00:00:00,500 --> 00:00:01,500\n第一条\n\n"
        "2\n00:00:02,000 --> 00:00:03,800\n第二条\n",
        encoding="utf-8")
    base = dirpath / "base.mkv"
    subprocess.run(
        [ffmpeg(), "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=4",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
         "-f", "lavfi", "-i", "sine=frequency=880:duration=4",
         "-map", "0:v", "-map", "1:a", "-map", "2:a",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
         "-c:a", "aac", "-shortest", str(base)],
        check=True)
    src = dirpath / "src.mkv"
    subprocess.run(
        [ffmpeg(), "-y", "-v", "error", "-i", str(base), "-i", str(srt),
         "-map", "0", "-map", "1", "-c:v", "copy", "-c:a", "copy",
         "-c:s", "srt", "-shortest", str(src)],
        check=True)
    return src


def _transcode(src: Path, dst: Path, *extra) -> Path:
    subprocess.run(
        [ffmpeg(), "-y", "-v", "error", "-i", str(src), *extra, str(dst)],
        check=True)
    return dst


def main() -> int:
    cases = []
    with tempfile.TemporaryDirectory(prefix="cf_verify_") as td:
        d = Path(td)
        src = _make_src(d)
        out = _transcode(src, d / "out.mkv", "-c:v", "libx264",
                         "-preset", "ultrafast", "-crf", "30", "-c:a", "copy",
                         "-c:s", "copy")

        # 1. 正常：三项全过
        r = verify_output(out, src, strict_metadata=True)
        cases.append(("正常产物全过", r.ok and not r.canceled))

        # 2. 时长篡改（截断 >0.5s 被检出）
        trunc = _transcode(src, d / "trunc.mkv", "-t", "2", "-c", "copy")
        r = verify_output(out, trunc, strict_metadata=True)
        cases.append(("时长篡改检出", not r.ok
                      and any("时长" in f for f in r.failures)))

        # 3. 损坏码流（截断字节 → 解码错误/非零退出）
        corrupt = d / "corrupt.mkv"
        data = out.read_bytes()
        corrupt.write_bytes(data[: len(data) // 3])
        r = verify_output(corrupt, None)
        cases.append(("损坏码流检出", not r.ok))

        # 4. strict 拒绝增强产物 / 宽松通过
        enh = _transcode(src, d / "enh.mkv", "-vf", "scale=640x480", "-r", "50",
                         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                         "-c:a", "copy")
        cases.append(("strict 拒绝元数据漂移",
                      not verify_output(enh, src, strict_metadata=True).ok))
        cases.append(("宽松模式通过",
                      verify_output(enh, src, strict_metadata=False).ok))

        # 5. 取消路径
        r = verify_output(out, None, cancel_check=lambda: True)
        cases.append(("取消路径", r.canceled and not r.ok))

    failed = [name for name, ok in cases if not ok]
    for name, ok in cases:
        print(f"  {'✓' if ok else '✗'} {name}")
    if failed:
        print(f"FAILED: {failed}")
        return 1
    print(f"全部 {len(cases)} 项通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
