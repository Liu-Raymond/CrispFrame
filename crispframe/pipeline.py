"""主管线：ffmpeg 解码 → 批处理帧处理链（去噪/去压缩/超分…）→ 可选插帧 → NVENC 编码。

设计约束：
- 音轨原样保留（copy）；解码器可 -t 限长（--max-seconds，用于测试/预览）。
- stage 协议：list[Frame] 进 list[Frame] 出（批处理）；插帧是跨批的时序操作，独立处理。
- 内存中只有常数个帧，4K 也不吃内存。
"""
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from .ffutils import VideoInfo, ffmpeg, probe, SUBPROC_FLAGS

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
    map_all_streams: bool = False          # 存档模式：全流保留（多音轨/章节）。
    # 单进程直编（run_passthrough）下是 -map 0 全映射；双进程增强管线
    #（_encoder_cmd）下是选择性映射（视频恒取管线帧），两处语义各自成立


class Interpolator:
    """接口：给定前后两帧，返回中间帧（0.5 位置）。"""
    def midframe(self, a: Frame, b: Frame) -> Frame:
        raise NotImplementedError


def _decoder_cmd(info: VideoInfo, max_seconds: float | None,
                 extra_vf: list[str] | None = None,
                 start: float | None = None) -> list[str]:
    cmd = [ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]   # -ss 在 -i 前 = 关键帧快定位
    cmd += ["-i", str(info.path), "-map", "0:v:0"]
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
    if cfg.map_all_streams:
        # 存档模式：视频流恒取输入 0（管线帧）——切勿用 -map 1 全映射，那会把
        # 源文件的视频流抢进输出、丢弃增强帧（直通场景产物看似正常，极难察觉）。
        # 音轨/字幕/章节从源（输入 1）映射，-dn 丢数据流，全局元数据（标题等）取源。
        streams = ["-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
                   "-dn", "-map_chapters", "1", "-map_metadata", "1"]
    else:
        streams = ["-map", "0:v:0", "-map", "1:a?"]
    cmd = [
        ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
        "-r", f"{fps:.6f}", "-i", "-",
        "-i", str(info.path),
        *streams, *audio, *subs,
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


def run_passthrough(
    info: VideoInfo,
    out_path: Path,
    encode: EncodeConfig | None = None,
    max_seconds: float | None = None,
    progress_callback=None,      # progress_callback(done_s, total_s)——单位是秒
    cancel_check=None,
) -> dict:
    """纯直通转码（存档）：单进程 ffmpeg 解码→编码，不经 rawvideo 管道。

    与独立 ffmpeg 脚本同构，因此与增强管线的 run() 有三点本质不同：
    - 解码与编码同进程，SVT-AV1 内部全核线程——CPU 满载（双进程管线会被
      解码/管道/转换的单线程段卡住供帧，编码器吃不满）
    - 色彩/时基元数据 passthrough（不注 bt709、不 -vf），HDR 等特殊色彩
      原样保留，产物元数据与源一致
    - 源即输入 0，`-map 0` 与脚本同构（双输入抢流的坑在单进程不存在）
    仅适用于 stages 为空且无插帧的场景；需要帧级处理必须走 run()。"""
    encode = encode or EncodeConfig()
    cmd = [ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-progress", "pipe:1", "-i", str(info.path)]
    if max_seconds is not None:
        cmd += ["-t", f"{max_seconds:.3f}"]
    if encode.map_all_streams:
        # 存档语义：全流保留（音轨/字幕/附件位流之外的映射由 -map 0 覆盖），
        # 章节/全局元数据同样取自源；数据流丢弃
        cmd += ["-map", "0", "-dn", "-map_chapters", "0", "-map_metadata", "0"]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a?"]
    # 时基跟随源（ffmpeg 9 已无 -vsync）；重编码视频 + 音轨/字幕按容器策略
    cmd += ["-fps_mode", "passthrough", *encode.args]
    audio = encode.audio_args if encode.audio_args is not None else ["-c:a", "copy"]
    subs = encode.sub_args if encode.sub_args is not None else []
    cmd += [*audio, *subs]
    # movflags 是 mp4 系 muxer 私有选项，发给 matroska 会直接报错
    if out_path.suffix.lower() in (".mp4", ".mov", ".m4v", ".f4v", ".3gp"):
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(out_path))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, errors="replace", **SUBPROC_FLAGS)
    assert proc.stdout is not None and proc.stderr is not None
    err_holder: list[str] = []

    def drain_errors():
        # SVT 的 info banner（约 17 行）会占满截断额度、淹掉真正的 ffmpeg
        # 错误（在 stderr 末尾），直接不收集
        err_holder.extend(line.rstrip() for line in proc.stderr
                          if line.strip() and not line.lstrip().startswith("Svt["))

    t = threading.Thread(target=drain_errors, daemon=True)
    t.start()
    total = info.duration or 0.0
    for line in proc.stdout:
        if cancel_check and cancel_check():
            proc.terminate()
            t.join(timeout=10)
            proc.wait()
            raise PassthroughCanceled()
        # 与 verify 同款：out_time_ms 历史上单位是微秒
        if progress_callback and total > 0 and line.startswith("out_time_ms="):
            try:
                us = int(line.split("=", 1)[1].strip())
                progress_callback(min(us / 1e6, total), total)
            except ValueError:
                pass
    t.join(timeout=10)
    proc.wait()
    t.join()
    if proc.returncode != 0:
        detail = "\n".join(err_holder[:20]) or f"退出码 {proc.returncode}"
        raise RuntimeError(f"直通编码失败:\n{detail}")
    return {"canceled": False, "wall_s": 0.0, "passthrough": True}


class PassthroughCanceled(RuntimeError):
    """run_passthrough 被用户取消（产物为残缺文件，调用方负责删除）。"""


def run(
    info: VideoInfo,
    out_path: Path,
    stages: list[BatchStage] | None = None,
    interpolator: Interpolator | None = None,
    encode: EncodeConfig | None = None,
    batch_size: int = 1,
    max_seconds: float | None = None,
    start: float | None = None,
    progress_callback=None,
    cancel_check=None,
) -> dict:
    """执行完整管线。

    start：从源的指定秒数开始（预览选段用）。
    progress_callback(done:int, total:int)：每帧回调（GUI 进度条用）。
    cancel_check() -> bool：返回 True 时中止处理（GUI 停止按钮用）。
    """
    stages = stages or []
    encode = encode or EncodeConfig()
    batch_size = max(1, batch_size)
    frame_bytes_in = info.width * info.height * 3
    out_fps = info.fps * (2 if interpolator is not None else 1)
    if max_seconds is not None or start:
        remaining = max(0.0, (info.duration or 1e9) - (start or 0.0))
        total = int(min(max_seconds if max_seconds is not None else 1e9,
                        remaining) * info.fps)
    else:
        total = info.frames

    decoder = subprocess.Popen(
        _decoder_cmd(info, max_seconds, encode.decode_vf, start),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, bufsize=frame_bytes_in * (batch_size + 2),
        **SUBPROC_FLAGS,
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
            **SUBPROC_FLAGS,
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
