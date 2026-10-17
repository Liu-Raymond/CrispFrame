"""编码设置引擎：CRF/preset 模式，SVT-AV1 软编存档工作流为默认。

参数基准来自用户验证过的 video-archive-svtav1 工作流（crf30/preset4/tune=0/MKV）：
- SVT-AV1 软编 = 存档默认；NVENC 仅"草稿"（同画质体积约 6 倍，插帧/超分内容尤甚）
- 质量档位注记：25-27 稳 / 30 甜点 / 32 省（SVT-AV1 crf 尺度）
- ffmpeg 9 已无 -vsync；容器相关（movflags）只发给 mp4 系
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field, fields, asdict

from .ffutils import ffmpeg, SUBPROC_FLAGS

# x264/x265 的 preset 用名称（数值不通用）；1-10 → 列表序
_X264_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast",
                 "medium", "slow", "slower", "veryslow", "placebo"]


# ---------------------------------------------------------------- 视频容器清单
# muxer 键 → 定义。字幕策略三档：copy 原样 / mov_text 转换（mp4 系）/ drop 丢弃
#（容器不支持字幕轨）。音轨默认 copy，不兼容由 ffmpeg 报错透出（说明列有提示）。
VIDEO_CONTAINERS: dict[str, dict] = {
    "matroska": {"label": "MKV（推荐·全兼容）", "ext": ".mkv", "subs": "copy",
                 "note": "字幕/多音轨/章节原样保留"},
    "mp4":      {"label": "MP4（通用）", "ext": ".mp4", "subs": "mov_text",
                 "note": "兼容性最好；字幕转 mov_text，个别音频需转 AAC"},
    "mov":      {"label": "MOV（QuickTime）", "ext": ".mov", "subs": "mov_text",
                 "note": "苹果生态；限制同 MP4"},
    "m4v":      {"label": "M4V（iTunes）", "ext": ".m4v", "subs": "mov_text",
                 "note": "Apple TV/iTunes；限制同 MP4"},
    "avi":      {"label": "AVI（老式）", "ext": ".avi", "subs": "drop",
                 "note": "老容器：不支持字幕，AV1/HEVC 兼容性差"},
    "webm":     {"label": "WebM（网页）", "ext": ".webm", "subs": "drop",
                 "note": "不支持字幕轨；音轨只吃 Opus/Vorbis（需转 AAC 输出会失败）"},
    "flv":      {"label": "FLV（Flash/直播）", "ext": ".flv", "subs": "drop",
                 "note": "网站/推流常用；不支持字幕"},
    "mpegts":   {"label": "MPEG-TS（电视/推流）", "ext": ".ts", "subs": "drop",
                 "note": "电视广播/网络推流；不支持软字幕"},
    "mpeg":     {"label": "MPEG-PS（VCD/DVD 系）", "ext": ".mpg", "subs": "drop",
                 "note": "老式光盘容器；仅 MPEG 系视频编码"},
    "vob":      {"label": "VOB（DVD 对象）", "ext": ".vob", "subs": "drop",
                 "note": "DVD 影片切片；仅 MPEG-2 系"},
    "asf":      {"label": "ASF/WMV（微软）", "ext": ".asf", "subs": "drop",
                 "note": "微软流媒体；仅 WMV/VC-1 系"},
    "3gp":      {"label": "3GP（老手机）", "ext": ".3gp", "subs": "mov_text",
                 "note": "3G 手机录像；限制同 MP4 系"},
    "3g2":      {"label": "3G2（老手机 CDMA）", "ext": ".3g2", "subs": "mov_text",
                 "note": "限制同 MP4 系"},
    "ogg":      {"label": "Ogg（开源）", "ext": ".ogv", "subs": "drop",
                 "note": "开源容器；音轨 Vorbis/Theora 系"},
    "nut":      {"label": "NUT（实验·全能）", "ext": ".nut", "subs": "drop",
                 "note": "实验性容器；srt 字幕 copy 不被接受（自动丢弃）"},
    "mxf":      {"label": "MXF（广电）", "ext": ".mxf", "subs": "drop",
                 "note": "广播级专业容器；限制严格"},
    "dv":       {"label": "DV（磁带）", "ext": ".dv", "subs": "drop",
                 "note": "DV 磁带格式；仅 DV 视频"},
    "gxf":      {"label": "GXF（广电）", "ext": ".gxf", "subs": "drop",
                 "note": "广播级；MPEG/MJPEG 系"},
    "wtv":      {"label": "WTV（Windows 电视录制）", "ext": ".wtv", "subs": "drop",
                 "note": "Windows Media Center 录制格式；srt 字幕 copy 不被接受（自动丢弃）"},
    "rm":       {"label": "RM（RealMedia）", "ext": ".rm", "subs": "drop",
                 "note": "远古流媒体；muxer 硬限最多 2 条流，多音轨/字幕源不可用"},
}

_muxers_cache: list[str] | None = None


def available_muxers() -> list[str]:
    """本机 ffmpeg 实际支持 muxing 的格式名（首次调用后缓存）。

    `ffmpeg -muxers` 输出行形如 " DE ac3   Raw AC-3"（2 字符标志 + 名字 +
    描述），标志含 E = Muxing supported。"""
    global _muxers_cache
    if _muxers_cache is None:
        try:
            out = subprocess.run([ffmpeg(), "-hide_banner", "-muxers"],
                                 capture_output=True, text=True,
                                 **SUBPROC_FLAGS).stdout
            names = []
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2 and len(parts[0]) <= 2 and "E" in parts[0]:
                    names.append(parts[1])
            _muxers_cache = names or sorted(VIDEO_CONTAINERS)
        except Exception:  # noqa: BLE001  ffmpeg 缺失时退回静态清单
            _muxers_cache = sorted(VIDEO_CONTAINERS)
    return _muxers_cache


# 编码器族 → 兼容容器集（ffmpeg 9.0.2 full 实测：单音轨 / 多音轨 / 带字幕
# 三种源形态逐一编码验证；muxer 层硬拒不兼容组合，如
# "av1 only supported in MP4 and AVIF"；rm muxer 硬限最多 2 条流，
# 与多音轨存档语义冲突，任何族都不含）。未列出的编码器默认全清单可用。
_CODEC_FAMILY_CONTAINERS = {
    "av1":  {"matroska", "mp4", "avi", "flv", "mpegts", "asf", "nut", "wtv"},
    "h264": {"matroska", "mp4", "mov", "m4v", "avi", "flv", "mpegts", "asf",
             "3gp", "3g2", "nut", "wtv"},
    "hevc": {"matroska", "mp4", "mov", "avi", "flv", "mpegts", "nut"},
}

_CODEC_FAMILY = {   # 编码器键 → 族
    "svtav1": "av1", "av1_nvenc": "av1",
    "libx264": "h264", "h264_nvenc": "h264",
    "libx265": "hevc", "hevc_nvenc": "hevc",
}


def codec_family(encoder_key: str) -> str | None:
    return _CODEC_FAMILY.get(encoder_key)


def available_containers(encoder_key: str | None = None) -> dict[str, dict]:
    """清单 ∩ 本机 ffmpeg 实际支持（保持定义顺序）。
    给出 encoder_key 时再按该编码器的实测兼容集过滤。"""
    muxers = set(available_muxers())
    family = _CODEC_FAMILY.get(encoder_key or "")
    allowed = _CODEC_FAMILY_CONTAINERS.get(family) if family else None
    return {k: v for k, v in VIDEO_CONTAINERS.items()
            if k in muxers and (allowed is None or k in allowed)}


def sub_codec_for(container: str) -> str:
    """容器的字幕编码策略：copy / mov_text / drop（见 VIDEO_CONTAINERS 定义）。"""
    return VIDEO_CONTAINERS.get(container, {"subs": "drop"})["subs"]


# 存档输出像素格式映射：源 pix_fmt → (位深, 输出 pix_fmt)。
# 严格元数据一致只保证表内格式；4:2:2/4:4:4/yuvj 等源不在表内，
# 由调用方预检拒绝（编码后验证必不一致，提前失败避免空转）。
_ARCHIVE_PIX_MAP = {
    "yuv420p": (8, "yuv420p"),
    "yuv420p10le": (10, "yuv420p10le"),
}


def archive_pix_fmt(src_pix: str) -> tuple[int, str] | None:
    """源像素格式 → (位深, 输出 pix_fmt)；不在支持表内返回 None。"""
    return _ARCHIVE_PIX_MAP.get(src_pix)


@dataclass
class EncoderSpec:
    key: str                     # 设置键（ENCODERS 字典键 / CodecSettings.encoder）
    name: str                    # GUI 显示名
    draft: bool                  # 草稿档（NVENC）
    crf_range: tuple[int, int]
    crf_default: int
    crf_notes: dict[int, str]    # 滑杆刻度注记
    preset_range: tuple[int, int]
    preset_default: int
    depths: tuple[int, ...] = (8, 10)
    depth_default: int = 10
    note: str = ""
    ffcodec: str = ""            # ffmpeg -c:v 值；空 = 同 key（svtav1 → libsvtav1）

    def codec_args(self, crf: int, preset: int, depth: int,
                   mode: str = "crf", bitrate_kbps: int = 12000) -> list[str]:
        raise NotImplementedError

    @property
    def ffmpeg_name(self) -> str:
        return self.ffcodec or self.key


def _pix(depth: int) -> str:
    return "yuv420p10le" if depth >= 10 else "yuv420p"


class _SvtAv1(EncoderSpec):
    def codec_args(self, crf, preset, depth, mode="crf", bitrate_kbps=12000):
        args = ["-c:v", "libsvtav1", "-preset", str(int(preset))]
        if mode == "crf":
            # tune=0 = VQ 视觉优化（用户验证的存档工作流参数）
            args += ["-crf", str(int(crf)), "-svtav1-params", "tune=0"]
        else:
            args += ["-rc", "vbr", "-b:v", f"{int(bitrate_kbps)}k"]
        args += ["-pix_fmt", _pix(depth)]
        return args


class _X26x(EncoderSpec):
    def codec_args(self, crf, preset, depth, mode="crf", bitrate_kbps=12000):
        name = _X264_PRESETS[max(0, min(len(_X264_PRESETS) - 1, int(preset) - 1))]
        args = ["-c:v", self.ffmpeg_name, "-preset", name]
        if mode == "crf":
            args += ["-crf", str(int(crf))]
        else:
            args += ["-b:v", f"{int(bitrate_kbps)}k"]
        args += ["-pix_fmt", _pix(depth)]
        return args


class _Nvenc(EncoderSpec):
    def codec_args(self, crf, preset, depth, mode="crf", bitrate_kbps=12000):
        # 参考工作流草稿分支：vbr + aq + lookahead
        args = ["-c:v", self.ffmpeg_name, "-preset", f"p{max(1, min(7, int(preset)))}",
                "-rc", "vbr", "-maxrate", "60M", "-bufsize", "120M",
                "-spatial-aq", "1", "-temporal-aq", "1", "-rc-lookahead", "32"]
        if mode == "crf":
            args += ["-cq", str(int(crf)), "-b:v", "0"]
        else:
            args += ["-b:v", f"{int(bitrate_kbps)}k"]
        args += ["-pix_fmt", _pix(depth)]
        return args


ENCODERS: dict[str, EncoderSpec] = {
    "svtav1": _SvtAv1("svtav1", "SVT-AV1 软编（推荐）", False,
                      (0, 63), 30, {25: "稳", 27: "稳", 30: "甜点", 32: "省"},
                      (1, 13), 4, (8, 10), 10,
                      "CPU 软编，同画质体积最小；插帧/超分内容尤甚（约 NVENC 的 1/6）",
                      ffcodec="libsvtav1"),
    "libx265": _X26x("libx265", "x265 软编", False,
                     (0, 51), 20, {}, (1, 10), 6, (8, 10), 10,
                     note="兼容性最好的软编备选"),
    "libx264": _X26x("libx264", "x264 软编", False,
                     (0, 51), 19, {}, (1, 10), 6, (8,), 8,
                     note="最大兼容；仅 8bit"),
    "av1_nvenc": _Nvenc("av1_nvenc", "AV1 NVENC（草稿）", True,
                        (0, 51), 28, {}, (1, 7), 5, (8, 10), 8,
                        note="快 10-20 倍，同画质体积约软编 6 倍——只建议草稿预览"),
    "hevc_nvenc": _Nvenc("hevc_nvenc", "HEVC NVENC（草稿）", True,
                         (0, 51), 26, {}, (1, 7), 5, (8, 10), 8,
                         note="快 10-20 倍，同画质体积约软编 6 倍——只建议草稿预览"),
}
ENCODERS["libx264"].depths = (8,)
ENCODERS["libx264"].depth_default = 8


# ---------------------------------------------------------------- 设置

@dataclass
class CodecSettings:
    encoder: str = "svtav1"
    quality_mode: str = "crf"        # crf（推荐）| abr（平均码率）
    crf: int = 30
    bitrate_kbps: int = 12000
    preset: int = 4                  # 语义随编码器：svt 1-13 / x26x 1-10 / nvenc 1-7
    bit_depth: int = 10              # 8 | 10
    container: str = "matroska"      # mkv（推荐·全兼容）| mp4 | mov | …（键=VIDEO_CONTAINERS）
    audio_mode: str = "copy"         # copy | aac
    audio_bitrate_kbps: int = 192
    subtitle_mode: str = "copy"      # copy | drop
    fps_mode: str = "source"         # source | custom
    fps_custom: float = 60.0
    res_mode: str = "source"         # source | custom
    res_custom: str = "3840x2160"
    deinterlace: bool = False

    # —— 序列化（配置文件用） ——
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "CodecSettings":
        valid = {f.name for f in fields(CodecSettings)}
        s = CodecSettings()
        for k, v in (d or {}).items():
            if k in valid:
                setattr(s, k, v)
        spec = ENCODERS.get(s.encoder)
        if spec is None:
            s.encoder, spec = "svtav1", ENCODERS["svtav1"]
        s.crf = max(spec.crf_range[0], min(spec.crf_range[1], int(s.crf)))
        s.preset = max(spec.preset_range[0], min(spec.preset_range[1], int(s.preset)))
        if s.bit_depth not in spec.depths:
            s.bit_depth = spec.depth_default
        if s.container == "mkv":          # 旧配置的简写键迁移为 muxer 键
            s.container = "matroska"
        if s.container not in VIDEO_CONTAINERS:
            s.container = "matroska"
        if s.quality_mode not in ("crf", "abr"):
            s.quality_mode = "crf"
        return s

    def spec(self) -> EncoderSpec:
        return ENCODERS[self.encoder]

    @staticmethod
    def for_encoder(key: str) -> "CodecSettings":
        """切换编码器时用：crf/preset/位深取该编码器自己的默认值。"""
        spec = ENCODERS[key]
        return CodecSettings(encoder=key, crf=spec.crf_default,
                             preset=spec.preset_default, bit_depth=spec.depth_default)

    def ext(self) -> str:
        return VIDEO_CONTAINERS.get(self.container, {"ext": ".mkv"})["ext"]


# ---------------------------------------------------------------- ffmpeg 命令构建

def video_codec_args(s: CodecSettings) -> list[str]:
    return s.spec().codec_args(s.crf, s.preset, s.bit_depth,
                               s.quality_mode, s.bitrate_kbps)


def encode_vf(s: CodecSettings) -> str:
    """编码侧 -vf：自定义分辨率缩放 + 显式 bt709/limited（ffmpeg 默认 bt601 会偏色）。"""
    parts = []
    if s.res_mode == "custom":
        w, _, h = (s.res_custom or "").lower().partition("x")
        if w.isdigit() and h.isdigit():
            parts.append(f"scale={w}:{h}")
    parts.append("scale=out_color_matrix=bt709:out_range=tv")
    return ",".join(parts)


def decode_vf(s: CodecSettings) -> list[str]:
    """解码侧滤镜：反交错需在源元数据可读的解码端做（send_frame 保持帧率）。"""
    return ["bwdif=mode=send_frame:parity=auto:deint=all"] if s.deinterlace else []


def audio_args(s: CodecSettings) -> list[str]:
    if s.audio_mode == "aac":
        return ["-c:a", "aac", "-b:a", f"{int(s.audio_bitrate_kbps)}k"]
    return ["-c:a", "copy"]


def sub_args(s: CodecSettings) -> list[str]:
    if s.subtitle_mode == "drop":
        return ["-sn"]
    # 字幕编码按容器策略（copy/mov_text/drop，见 VIDEO_CONTAINERS）；
    # 映射本身由调用方的 -map 负责，这里只出编码方式
    strategy = sub_codec_for(s.container)
    if strategy == "drop":
        return ["-sn"]            # 容器不支持字幕轨（如 AVI/WebM/FLV）
    enc = "mov_text" if strategy == "mov_text" else "copy"
    return ["-map", "1:s?", "-c:s", enc]


def out_fps(s: CodecSettings, source_fps: float, has_interp: bool) -> float:
    """输出帧率：source = 源 ×（插帧 2）；custom 直接生效（含插帧时同样覆盖）。"""
    if s.fps_mode == "custom" and s.fps_custom > 0:
        return float(s.fps_custom)
    return source_fps * (2 if has_interp else 1)


def smart_hint(s: CodecSettings, module_keys: list[str]) -> str | None:
    """智能提示条：有插帧/平滑/超分模块却选了硬编时提醒（用户验证的 6 倍体积差）。"""
    heavy = any(k in ("sr", "interp", "smooth") for k in module_keys)
    if heavy and s.spec().draft:
        return ("当前链含超分/插帧/平滑模块：强烈建议改用 SVT-AV1 软编，"
                "同画质下 NVENC 体积约为软编的 6 倍")
    if heavy and s.encoder != "svtav1" and not s.spec().draft:
        return "插帧/平滑/超分内容建议 SVT-AV1 软编（体积最小、质量最稳）"
    return None


def build_encode_config_kwargs(s: CodecSettings, source_fps: float,
                               has_interp: bool, vf_pre: str = "",
                               map_all: bool = False) -> dict:
    """生成 pipeline.EncodeConfig 的泛化字段（GUI/新管线入口用）。

    vf_pre：模块注入的编码侧前置滤镜（如平滑模块的 tmix 混合 + fps 回采），
    置于分辨率/色彩转换之前——混合在管线的 bgr24 RGB 域语义下进行。
    fps_override 保持 2x（rawvideo 输入标签语义），实际输出帧率由 vf 内的
    fps 滤镜回采决定，回采后 PTS 恒定、muxer 元数据正确。
    map_all：存档模式全流映射（多音轨/字幕/章节原样保留）。"""
    vf = encode_vf(s)
    # 平滑（vf_pre 非空）的输出帧率由 vf 内 fps 滤镜锁定为源帧率，编码输入
    # 标签恒为 2x；自定义帧率在此组合下会压缩时长/音画失步，强制忽略
    fps_override = (source_fps * 2 if vf_pre
                    else out_fps(s, source_fps, has_interp))
    if map_all:
        # 全流映射下字幕的映射由 -map 完成，这里只定编码方式
        #（重复 -map 会把字幕流映射两遍）
        if s.subtitle_mode == "drop":
            subs = ["-sn"]
        else:
            strategy = sub_codec_for(s.container)
            subs = ["-sn"] if strategy == "drop" else ["-c:s", strategy]
    else:
        subs = sub_args(s)
    return {
        "args": video_codec_args(s),
        "vf": f"{vf_pre},{vf}" if vf_pre else vf,
        "decode_vf": decode_vf(s),
        "audio_args": audio_args(s),
        "sub_args": subs,
        "fps_override": fps_override,
        "map_all_streams": map_all,
    }
