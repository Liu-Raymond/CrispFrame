"""编码设置引擎：CRF/preset 模式，SVT-AV1 软编存档工作流为默认。

参数基准来自用户验证过的 video-archive-svtav1 工作流（crf30/preset4/tune=0/MKV）：
- 定版配方：crf30/preset4/tune=0；质量档位注记：25-27 稳 / 30 甜点 / 32 省
- ffmpeg 9 已无 -vsync；容器相关（movflags）只发给 mp4 系
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field, fields, asdict

from .ffutils import ffmpeg, SUBPROC_FLAGS

# x264/x265 的 preset 用名称（数值不通用）；1-10 → 列表序
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


# ---------------------------------------------------------------------------
# SVT-AV1 的 8K 级 preset 限制（2026-11-04 实测钉界）
# preset4 拒绝域：7104x4312 / 7680x2880(22.1M) / 6000x4000 / 7680x4320；
# 通过域：4320x2880 / 5120x2880 / 7104x2880(20.5M) / 3840x4312 / 7680x1080。
# 结论：面积制（宽高需同时大：单边超不触发），阈值在 20.5M~22.1M 之间，
# ffmpeg 报 "Error setting encoder parameters: bad parameter (0x80001005)"，
# SVT 明示 "8k+ resolution support is limited to M5 and faster presets"。
# 阈值取保守 20M（≈5000x4000）：绝不漏拒（宁可多升档——preset5 只是更慢些）。
SVT_8K_AREA = 20_000_000


def svtav1_fix_preset(preset: int, width: int, height: int) -> tuple[int, bool]:
    """8K 级分辨率（面积 ≥ SVT_8K_AREA）SVT-AV1 拒绝 preset<5：
    返回 (修正后 preset, 是否上调)；preset≥5 原样返回。"""
    if preset < 5 and width * height >= SVT_8K_AREA:
        return 5, True
    return preset, False


def is_svt_8k_error(msg: str) -> bool:
    """SVT 8K preset 拒绝的报错特征。两道标志：SVT 原生行（pipeline 的
    stderr 收集保留 Svt[error] 行）与 ffmpeg 包装层的 libsvtav1 bad parameter
    （bad parameter 本身泛化，须与 libsvtav1 同现才判）。"""
    m = msg.lower()
    return ("limited to m5" in m or "8k+ resolution support" in m
            or ("libsvtav1" in m and "bad parameter" in m))


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


ENCODERS: dict[str, EncoderSpec] = {
    "svtav1": _SvtAv1("svtav1", "SVT-AV1 软编", False,
                      (0, 63), 30, {25: "稳", 27: "稳", 30: "甜点", 32: "省"},
                      (1, 13), 4, (8, 10), 10,
                      "CPU 软编，同画质体积最小", ffcodec="libsvtav1"),
}


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

    def ext(self) -> str:
        return VIDEO_CONTAINERS.get(self.container, {"ext": ".mkv"})["ext"]


# ---------------------------------------------------------------- ffmpeg 命令构建

def video_codec_args(s: CodecSettings) -> list[str]:
    return s.spec().codec_args(s.crf, s.preset, s.bit_depth,
                               s.quality_mode, s.bitrate_kbps)


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


def build_encode_config_kwargs(s: CodecSettings, map_all: bool = True) -> dict:
    """生成 pipeline.EncodeConfig 的字段（存档全流映射）。

    字幕映射由 -map 完成，这里只定编码方式（drop / 按容器策略）。"""
    if s.subtitle_mode == "drop":
        subs = ["-sn"]
    else:
        strategy = sub_codec_for(s.container)
        subs = ["-sn"] if strategy == "drop" else ["-c:s", strategy]
    return {
        "args": video_codec_args(s),
        "audio_args": audio_args(s),
        "sub_args": subs,
        "map_all_streams": map_all,
    }
