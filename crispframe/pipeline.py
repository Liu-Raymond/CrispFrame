"""存档管线：单进程 ffmpeg 直编（run_passthrough）。

存档语义：解码与编码同进程、-map 0 全流保留、色彩/时基元数据跟随源。
失败分类：正常失败（有 stderr）→ RuntimeError；进程被外部无声终止
（零错误输出）→ SilentTermination（环境性瞬时故障，调用方可安全重试）；
用户取消 → PassthroughCanceled。
"""
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .ffutils import VideoInfo, ffmpeg, probe, SUBPROC_FLAGS


@dataclass
class EncodeConfig:
    """存档编码配置（由 codec.build_encode_config_kwargs 生成）。"""
    args: list[str] | None = None          # 完整 -c:v 编码参数
    audio_args: list[str] | None = None    # 默认 ["-c:a","copy"]
    sub_args: list[str] | None = None      # 默认 []；非空时附带字幕流映射
    map_all_streams: bool = False          # 全流保留（多音轨/字幕/章节/元数据）


def run_passthrough(
    info: VideoInfo,
    out_path: Path,
    encode: EncodeConfig | None = None,
    max_seconds: float | None = None,
    progress_callback=None,      # progress_callback(done_s, total_s)——单位是秒
    cancel_check=None,
) -> dict:
    """存档直编：单进程 ffmpeg 解码→编码。

    - 解码与编码同进程，SVT-AV1 内部全核线程——CPU 满载
    - 色彩/时基元数据 passthrough（不注 bt709、不 -vf），HDR 等特殊色彩
      原样保留，产物元数据与源一致
    - 源即输入 0，`-map 0` 全映射（多音轨/字幕/章节/元数据跟随源）"""
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
        # SVT 的 info/warn banner（约 17 行）会占满截断额度、淹掉真正的 ffmpeg
        # 错误（在 stderr 末尾），不收集；但 Svt[error] 行是唯一精确的诊断标志
        #（如 8K 级 preset 限制 "limited to M5"），必须保留
        err_holder.extend(line.rstrip() for line in proc.stderr
                          if line.strip()
                          and not line.lstrip().startswith(("Svt[info", "Svt[warn")))

    t = threading.Thread(target=drain_errors, daemon=True)
    t.start()
    total = info.duration or 0.0
    last_t = [0.0]                   # 最后一次进度（静默终止时报告死在哪）
    for line in proc.stdout:
        if cancel_check and cancel_check():
            proc.terminate()
            t.join(timeout=10)
            proc.wait()
            raise PassthroughCanceled()
        # 与 verify 同款：out_time_ms 历史上单位是微秒
        if line.startswith("out_time_ms="):
            try:
                us = int(line.split("=", 1)[1].strip())
                last_t[0] = min(us / 1e6, total) if total > 0 else us / 1e6
                if progress_callback and total > 0:
                    progress_callback(last_t[0], total)
            except ValueError:
                pass
    t.join(timeout=10)
    proc.wait()
    t.join()
    if proc.returncode != 0:
        detail = "\n".join(err_holder[:20])
        if not detail:
            # 零错误输出的非零退出：ffmpeg 被外部终止（任务管理器结束进程即此类，
            # 实测退出码 1 或 0xFFFFFFFF；正常失败必有 stderr，自主取消走 PassthroughCanceled）。
            # 报人话 + 已编码位置，供上层自愈重试判断
            pos = (f"已编码到约 {last_t[0]:.0f}/{total:.0f} 秒" if total > 0
                   else f"已编码到约 {last_t[0]:.0f} 秒")
            raise SilentTermination(
                f"编码进程异常终止（退出码 {proc.returncode}），未留下任何错误输出"
                f"（{pos}）。常见原因：进程被外部结束（任务管理器/杀毒软件）"
                f"或系统内存压力；若非主动操作，可直接重试")
        raise RuntimeError(f"直通编码失败:\n{detail}")
    return {"canceled": False, "wall_s": 0.0, "passthrough": True}


class PassthroughCanceled(RuntimeError):
    """run_passthrough 被用户取消（产物为残缺文件，调用方负责删除）。"""


class SilentTermination(RuntimeError):
    """ffmpeg 非零码退出且零错误输出——被外部终止（任务管理器/杀软/内存压力）。

    正常编码失败必有 stderr（-loglevel error），自主取消走 PassthroughCanceled；
    这类无声死亡几乎都是环境性的（瞬时），调用方可安全地自动重试一次。"""
