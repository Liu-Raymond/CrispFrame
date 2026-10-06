"""主管线：ffmpeg 解码 → 批处理帧处理链（去噪/去压缩/超分…）→ 可选插帧 → NVENC 编码。

设计约束：
- 音轨原样保留（copy）；解码器可 -t 限长（--max-seconds，用于测试/预览）。
- stage 协议：list[Frame] 进 list[Frame] 出（批处理）；插帧是跨批的时序操作，独立处理。
- 内存中只有常数个帧，4K 也不吃内存。
"""
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from .ffutils import VideoInfo, ffmpeg, probe

Frame = np.ndarray  # HxWx3 uint8 BGR
BatchStage = Callable[[list[Frame]], list[Frame]]


@dataclass
class EncodeConfig:
    """旧四字段为 CLI/NVENC 直通路径；GUI 经 codec.build_encode_config_kwargs 填泛化字段。"""
    encoder: str = "av1_nvenc"   # av1_nvenc | hevc_nvenc
    crf: int = 28                # av1: 24-30 感知无损带；hevc 建议 20-24
    preset: str = "p5"           # p1 最快 … p7 最慢质最好
    pix_fmt: str = "yuv420p"
    # —— 泛化字段（codec.py 生成；args 设置后覆盖上面四项） ——
    args: list[str] | None = None          # 完整 -c:v 编码参数
    vf: str | None = None                  # 编码侧 -vf（默认 bt709 转换）
    decode_vf: list[str] | None = None     # 解码侧滤镜（如 bwdif 反交错）
    fps_override: float | None = None      # 输出帧率（默认 源×插帧倍率）
    audio_args: list[str] | None = None    # 默认 ["-c:a","copy"]
    sub_args: list[str] | None = None      # 默认 []；非空时附带字幕流映射


class Interpolator:
    """接口：给定前后两帧，返回中间帧（0.5 位置）。"""
    def midframe(self, a: Frame, b: Frame) -> Frame:
        raise NotImplementedError


def _decoder_cmd(info: VideoInfo, max_seconds: float | None,
                 extra_vf: list[str] | None = None) -> list[str]:
    cmd = [ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error",
           "-i", str(info.path), "-map", "0:v:0"]
    if extra_vf:
        cmd += ["-vf", ",".join(extra_vf)]
    if max_seconds:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    return cmd


def _encoder_cmd(info: VideoInfo, out_size: tuple[int, int], out_fps: float,
                 cfg: EncodeConfig, out_path: Path) -> list[str]:
    ow, oh = out_size
    fps = cfg.fps_override if cfg.fps_override and cfg.fps_override > 0 else out_fps
    codec_args = cfg.args if cfg.args is not None else [
        "-c:v", cfg.encoder, "-preset", cfg.preset,
        "-rc", "vbr", "-cq", str(cfg.crf), "-b:v", "0",
        "-pix_fmt", cfg.pix_fmt,
    ]
    vf = cfg.vf or "scale=out_color_matrix=bt709:out_range=tv"
    audio = cfg.audio_args if cfg.audio_args is not None else ["-c:a", "copy"]
    subs = cfg.sub_args or []
    cmd = [
        ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
        "-r", f"{fps:.6f}", "-i", "-",
        "-i", str(info.path),
        "-map", "0:v:0", "-map", "1:a?", *audio, *subs,
        # RGB(bgr24, full)→YUV 转换显式用 bt709/limited（ffmpeg 默认 bt601 会造成色偏），
        # 并把元数据写入输出，播放器才不会二次误读
        "-vf", vf,
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-color_range", "tv",
        *codec_args,
    ]
    # movflags 是 mp4/mov muxer 私有选项，发给 matroska 会直接报错
    if out_path.suffix.lower() in (".mp4", ".mov"):
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(out_path))
    return cmd


def run(
    info: VideoInfo,
    out_path: Path,
    stages: list[BatchStage] | None = None,
    interpolator: Interpolator | None = None,
    encode: EncodeConfig | None = None,
    batch_size: int = 1,
    max_seconds: float | None = None,
    progress_callback=None,
    cancel_check=None,
) -> dict:
    """执行完整管线。

    progress_callback(done:int, total:int)：每帧回调（GUI 进度条用）。
    cancel_check() -> bool：返回 True 时中止处理（GUI 停止按钮用）。
    """
    stages = stages or []
    encode = encode or EncodeConfig()
    batch_size = max(1, batch_size)
    frame_bytes_in = info.width * info.height * 3
    out_fps = info.fps * (2 if interpolator is not None else 1)
    if max_seconds:
        total = int(min(info.duration or 1e9, max_seconds) * info.fps)
    else:
        total = info.frames

    decoder = subprocess.Popen(
        _decoder_cmd(info, max_seconds, encode.decode_vf), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, bufsize=frame_bytes_in * (batch_size + 2),
    )
    encoder: subprocess.Popen | None = None
    enc_err_holder: list[str] = []
    stats = {"frames_in": 0, "frames_out": 0, "process_fps": 0.0, "wall_s": 0.0}
    t0 = time.perf_counter()
    proc_time = 0.0
    prev: Frame | None = None
    pending: list[Frame] = []

    def ensure_encoder(first: Frame):
        nonlocal encoder
        oh, ow = first.shape[:2]
        encoder = subprocess.Popen(
            _encoder_cmd(info, (ow, oh), out_fps, encode, out_path),
            stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=ow * oh * 3 * 2,
        )

    def write_out(frame: Frame):
        nonlocal encoder
        if encoder is None:
            ensure_encoder(frame)
        try:
            # memoryview 避免 tobytes() 的整帧拷贝（4K 帧 24MB）
            encoder.stdin.write(memoryview(frame))  # type: ignore[union-attr]
            stats["frames_out"] += 1
        except (BrokenPipeError, OSError) as e:
            err = encoder.stderr.read().decode(errors="replace") if encoder.stderr else ""
            raise RuntimeError(f"编码器中断:\n{err[-2000:]}") from e

    def flush(batch: list[Frame]):
        """处理一批：过 stage 链 → 插帧展开 → 写编码器。"""
        nonlocal prev, proc_time
        t1 = time.perf_counter()
        seq = batch
        for stage in stages:
            seq = stage(seq)
        proc_time += time.perf_counter() - t1
        for frame in seq:
            if interpolator is not None and prev is not None:
                t1 = time.perf_counter()
                mid = interpolator.midframe(prev, frame)
                proc_time += time.perf_counter() - t1
                write_out(mid)
            write_out(frame)
            prev = frame

    assert decoder.stdout is not None
    pbar = None if progress_callback else tqdm(
        total=total or None, unit="f", dynamic_ncols=True,
        desc=f"CrispFrame {info.path.name}")
    try:
        while True:
            if cancel_check and cancel_check():
                break
            raw = decoder.stdout.read(frame_bytes_in)
            if not raw or len(raw) < frame_bytes_in:
                break
            pending.append(np.frombuffer(raw, np.uint8).reshape(info.height, info.width, 3))
            stats["frames_in"] += 1
            if pbar:
                pbar.update(1)
            elif progress_callback:
                progress_callback(stats["frames_in"], total or 0)
            if len(pending) >= batch_size:
                flush(pending)
                pending = []
        if pending:
            flush(pending)
            pending = []
    finally:
        # 只做清理和错误收集，不在这里 raise——避免掩盖循环里的真实异常
        if pbar:
            pbar.close()
        if decoder.stdout:
            decoder.stdout.close()
        dec_err = decoder.stderr.read().decode(errors="replace") if decoder.stderr else ""
        decoder.wait()
        if encoder is not None:
            try:
                encoder.stdin.close()  # type: ignore[union-attr]
            except OSError:
                pass
            enc_err = encoder.stderr.read().decode(errors="replace") if encoder.stderr else ""
            enc_ret = encoder.wait()
        else:
            enc_err, enc_ret = "", 0

    if decoder.returncode not in (0, None):
        raise RuntimeError(f"解码器退出码 {decoder.returncode}:\n{dec_err[-2000:]}")
    if enc_ret != 0:
        raise RuntimeError(f"编码器退出码 {enc_ret}:\n{enc_err[-2000:]}")

    stats["wall_s"] = time.perf_counter() - t0
    stats["process_fps"] = stats["frames_in"] / proc_time if proc_time else 0.0
    return stats
