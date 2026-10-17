"""存档验证：编码后三项完整性检查（体积对比 / 元数据核对 / 全量解码查错）。

参数基准来自用户验证过的 video-archive-svtav1 工作流（archive-encode.sh）。
隐私原则：只读元数据与解码错误流，永不抽帧/截图/读取画面内容。

- 体积对比只做报告不判失败（容器/码率差异巨大是正常现象）。
- 元数据核对：分辨率/帧率/像素格式逐项比对 + 时长误差 <0.5s（丢尾/掉帧检测）。
  strict_metadata=False 时跳过逐项比对只查时长——增强产物（超分/插帧/平滑）
  的分辨率与帧率会合理变化，逐项比对无意义。
- 全量解码查错：`-v error -f null -` 逐帧解码全片，抽样探测会漏检，必须全量。
  stderr 由后台线程收集（错误风暴时管道缓冲不足以容纳，主线程死读会死锁）。
"""
from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .ffutils import ffmpeg, ffprobe, SUBPROC_FLAGS


def human_size(n: int) -> str:
    b = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} TB"


@dataclass
class VerifyResult:
    ok: bool = True
    canceled: bool = False
    failures: list[str] = field(default_factory=list)   # ✗ 行（已带原因）
    lines: list[str] = field(default_factory=list)      # ✓ 与信息行（直接打印）
    decode_errors: list[str] = field(default_factory=list)
    size_bytes: int = 0
    src_size_bytes: int = 0
    duration: float = 0.0
    src_duration: float = 0.0

    def fail(self, msg: str):
        self.ok = False
        self.failures.append(msg)


def probe_stream_meta(path: Path) -> dict:
    """视频流元数据：width/height/r_frame_rate(原始字符串)/pix_fmt。"""
    out = subprocess.run(
        [ffprobe(), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,pix_fmt",
         "-of", "json", str(path)],
        capture_output=True, text=True, **SUBPROC_FLAGS,
    ).stdout
    return json.loads(out)["streams"][0]


def _probe_duration(path: Path) -> float:
    out = subprocess.run(
        [ffprobe(), "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True, **SUBPROC_FLAGS,
    ).stdout
    return float(json.loads(out).get("format", {}).get("duration") or 0)


def _meta_line(meta: dict, dur: float) -> str:
    return (f"{meta['width']}x{meta['height']} "
            f"{meta['r_frame_rate']}fps {meta['pix_fmt']}  时长={dur:.2f}s")


def _decode_check(out: Path, total_s: float, progress_callback, cancel_check,
                  r: VerifyResult) -> None:
    """全量解码查错。stdout 是 -progress 键值流，stderr 是解码错误行。"""
    cmd = [ffmpeg(), "-nostdin", "-hide_banner", "-nostats",
           "-progress", "pipe:1", "-v", "error",
           "-i", str(out), "-f", "null", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, errors="replace", **SUBPROC_FLAGS)
    assert proc.stdout is not None and proc.stderr is not None

    err_holder: list[str] = []       # 只留前 500 行（损坏大文件可产生百万行级错误）
    err_total = [0]

    def drain_errors():
        for line in proc.stderr:     # 逐行排空到 EOF 防管道死锁
            if not line.strip():
                continue
            err_total[0] += 1
            if len(err_holder) < 500:
                err_holder.append(line.rstrip())

    t = threading.Thread(target=drain_errors, daemon=True)
    t.start()
    for line in proc.stdout:
        if cancel_check and cancel_check():
            proc.terminate()
            r.canceled = True
            break
        # ffmpeg 的 out_time_ms 历史上单位是微秒（字段名有误导），按微秒换算
        if progress_callback and total_s > 0 and line.startswith("out_time_ms="):
            try:
                us = int(line.split("=", 1)[1].strip())
                progress_callback(min(us / 1e6, total_s), total_s)
            except ValueError:
                pass
    t.join(timeout=10)
    proc.wait()
    t.join()                          # 兜底收尾，不丢最后几行错误

    if r.canceled:
        r.fail("✗ 验证被用户取消")
        return
    r.decode_errors = err_holder
    if proc.returncode != 0:
        r.fail(f"✗ 全量解码非零退出（退出码 {proc.returncode}，码流可能损坏）")
    if err_total[0]:
        tail = f"（已截取前 {len(err_holder)} 行）" if err_total[0] > len(err_holder) else ""
        r.fail(f"✗ 发现 {err_total[0]} 行解码错误{tail}")


def verify_output(
    out: Path,
    src: Path | None = None,
    strict_metadata: bool = True,
    progress_callback=None,      # progress_callback(done_s, total_s)
    cancel_check=None,           # -> bool，True 时终止解码并标记 canceled
) -> VerifyResult:
    """对成品执行三项验证并返回结构化结果（含可直接打印的 ✓/✗ 行）。"""
    out = Path(out)
    r = VerifyResult()
    if src is not None:
        r.src_size_bytes = Path(src).stat().st_size
    r.size_bytes = out.stat().st_size

    # ---- 1. 体积对比（仅报告，不判失败） ----
    size_line = f"体积: {human_size(r.size_bytes)}"
    if src is not None:
        size_line += f" / 源 {human_size(r.src_size_bytes)} (含音轨字幕容器差异)"
    r.lines.append(size_line)

    # ---- 2. 元数据核对 ----
    r.duration = _probe_duration(out)
    if r.duration <= 0:
        r.fail("✗ 无法读取输出时长（文件损坏或非媒体文件）")
        return r
    meta = probe_stream_meta(out)
    r.lines.append(f"输出: {_meta_line(meta, r.duration)}")
    if src is not None:
        r.src_duration = _probe_duration(Path(src))
        src_meta = probe_stream_meta(Path(src))
        r.lines.append(f"源片: {_meta_line(src_meta, r.src_duration)}")
        if strict_metadata:
            same = (int(meta["width"]), int(meta["height"]),
                    meta["r_frame_rate"], meta["pix_fmt"]) == \
                   (int(src_meta["width"]), int(src_meta["height"]),
                    src_meta["r_frame_rate"], src_meta["pix_fmt"])
            r.lines.append("✓ 分辨率/帧率/像素格式一致" if same
                           else "✗ 分辨率/帧率/像素格式不一致")
            if not same:
                r.fail("✗ 分辨率/帧率/像素格式与源片不一致")
        if abs(r.duration - r.src_duration) < 0.5:
            r.lines.append("✓ 时长一致（误差<0.5s）")
        else:
            r.fail(f"✗ 时长偏差超 0.5s（源 {r.src_duration:.2f}s / 出 "
                   f"{r.duration:.2f}s，疑似丢尾/掉帧）")

    # ---- 3. 全量解码查错 ----
    r.lines.append("全量解码查错中（只收集解码错误，约数分钟到半小时）...")
    _decode_check(out, r.duration, progress_callback, cancel_check, r)
    if not r.canceled:
        r.lines.append("✓ 全片解码零错误" if not r.decode_errors
                       else f"✗ 解码错误前 20 行：\n" +
                            "\n".join(r.decode_errors[:20]))
    return r
