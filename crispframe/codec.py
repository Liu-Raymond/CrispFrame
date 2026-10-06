"""编码设置引擎：CRF/preset 模式，SVT-AV1 软编存档工作流为默认。

参数基准来自用户验证过的 video-archive-svtav1 工作流（crf30/preset4/tune=0/MKV）：
- SVT-AV1 软编 = 存档默认；NVENC 仅"草稿"（同画质体积约 6 倍，插帧/超分内容尤甚）
- 质量档位注记：25-27 稳 / 30 甜点 / 32 省（SVT-AV1 crf 尺度）
- ffmpeg 9 已无 -vsync；容器相关（movflags）只发给 mp4/mov
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict

# x264/x265 的 preset 用名称（数值不通用）；1-10 → 列表序
_X264_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast",
                 "medium", "slow", "slower", "veryslow", "placebo"]


@dataclass
class EncoderSpec:
    key: str                     # ffmpeg -c:v 值 / 设置键
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

    def codec_args(self, crf: int, preset: int, depth: int,
                   mode: str = "crf", bitrate_kbps: int = 12000) -> list[str]:
        raise NotImplementedError


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
        args = ["-c:v", self.key, "-preset", name]
        if mode == "crf":
            args += ["-crf", str(int(crf))]
        else:
            args += ["-b:v", f"{int(bitrate_kbps)}k"]
        args += ["-pix_fmt", _pix(depth)]
        return args


class _Nvenc(EncoderSpec):
    def codec_args(self, crf, preset, depth, mode="crf", bitrate_kbps=12000):
        # 参考工作流草稿分支：vbr + aq + lookahead
        args = ["-c:v", self.key, "-preset", f"p{max(1, min(7, int(preset)))}",
                "-rc", "vbr", "-maxrate", "60M", "-bufsize", "120M",
                "-spatial-aq", "1", "-temporal-aq", "1", "-rc-lookahead", "32"]
        if mode == "crf":
            args += ["-cq", str(int(crf)), "-b:v", "0"]
        else:
            args += ["-b:v", f"{int(bitrate_kbps)}k"]
        args += ["-pix_fmt", _pix(depth)]
        return args


ENCODERS: dict[str, EncoderSpec] = {
    "svtav1": _SvtAv1("libsvtav1", "SVT-AV1 软编（推荐）", False,
                      (0, 63), 30, {25: "稳", 27: "稳", 30: "甜点", 32: "省"},
                      (1, 13), 4, (8, 10), 10,
                      note="CPU 软编，同画质体积最小；插帧/超分内容尤甚（约 NVENC 的 1/6）"),
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
    container: str = "mkv"           # mkv（推荐）| mp4
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
        if s.container not in ("mkv", "mp4"):
            s.container = "mkv"
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
        return ".mkv" if self.container == "mkv" else ".mp4"


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
    # mp4 只吃 mov_text（ass/srt 直 copy 会失败）；mkv 原样复制
    return ["-map", "1:s?", "-c:s", "mov_text" if s.container == "mp4" else "copy"]


def out_fps(s: CodecSettings, source_fps: float, has_interp: bool) -> float:
    """输出帧率：source = 源 ×（插帧 2）；custom 直接生效（含插帧时同样覆盖）。"""
    if s.fps_mode == "custom" and s.fps_custom > 0:
        return float(s.fps_custom)
    return source_fps * (2 if has_interp else 1)


def smart_hint(s: CodecSettings, module_keys: list[str]) -> str | None:
    """智能提示条：有插帧/超分模块却选了硬编时提醒（用户验证的 6 倍体积差）。"""
    heavy = any(k in ("sr", "interp") for k in module_keys)
    if heavy and s.spec().draft:
        return ("当前链含超分/插帧模块：强烈建议改用 SVT-AV1 软编，"
                "同画质下 NVENC 体积约为软编的 6 倍")
    if heavy and s.encoder != "svtav1" and not s.spec().draft:
        return "插帧/超分内容建议 SVT-AV1 软编（体积最小、质量最稳）"
    return None


def build_encode_config_kwargs(s: CodecSettings, source_fps: float,
                               has_interp: bool) -> dict:
    """生成 pipeline.EncodeConfig 的泛化字段（GUI/新管线入口用）。"""
    return {
        "args": video_codec_args(s),
        "vf": encode_vf(s),
        "decode_vf": decode_vf(s),
        "audio_args": audio_args(s),
        "sub_args": sub_args(s),
        "fps_override": out_fps(s, source_fps, has_interp),
    }
