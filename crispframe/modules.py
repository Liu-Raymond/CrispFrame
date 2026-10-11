"""AI 模块注册表：GUI v2 模块链的单一事实来源。

模块 = 用户可添加/排序/开关的增强环节。四个内置模块：
去压缩 → 降噪 → 超分 → 插帧（默认顺序；插帧实际在管线链尾执行）。

- ModuleSpec：模块类型定义（参数 schema + 构建）
- ModuleInstance：链上的一个模块实例（参数值 + 开关），可 JSON 序列化
- build_chain()：把实例链组装成 pipeline.run 需要的 (stages, interpolator)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import registry
from .pipeline import Interpolator

# ---------------------------------------------------------------- 参数 schema


@dataclass
class ParamSpec:
    key: str
    label: str
    kind: str                    # choice | float | int | bool
    default: Any
    choices: list[tuple[str, Any]] = field(default_factory=list)  # (显示, 值)
    min: float = 0.0
    max: float = 0.0
    step: float = 0.0
    tip: str = ""

    def coerce(self, value: Any) -> Any:
        """从 JSON/UI 回来的值规整成合法值（容器串/类型漂移兜底）。"""
        if self.kind == "choice":
            for _, v in self.choices:
                if value == v or str(value) == str(v):
                    return v
            return self.default
        if self.kind == "bool":
            return bool(value)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return self.default
        value = max(self.min, min(self.max, value))
        return int(round(value)) if self.kind == "int" else round(value, 2)


@dataclass
class BuildContext:
    """构建 stage 时的源信息与后端开关。"""
    width: int
    height: int
    fps: float
    use_trt: bool = True
    use_amp: bool = True
    compile_model: bool = False
    upstream_scale: int = 1       # 链上超分模块的放大倍数，build_chain 填充（插帧 auto 尺度用）


@dataclass
class BuiltModule:
    stages: list = field(default_factory=list)          # list[BatchStage]
    interpolator: Interpolator | None = None


# ---------------------------------------------------------------- ModuleSpec


@dataclass
class ModuleSpec:
    key: str
    name: str                     # 显示名
    description: str
    order: int                    # 默认链序（去压缩 0 → 降噪 1 → 超分 2 → 插帧 3）
    params: list[ParamSpec] = field(default_factory=list)
    singleton: bool = False       # 链中最多一个（插帧）
    always_last: bool = False     # 时序操作，视觉上可拖动但实际恒在链尾

    def build(self, params: dict, ctx: BuildContext) -> BuiltModule:
        raise NotImplementedError

    def summary(self, params: dict, src_h: int = 0) -> str:
        """模块行的参数摘要（一行）。src_h = 源高度，auto 模型解析用，未知传 0。"""
        return ""


# ---------------------------------------------------------------- 自动批大小

def auto_batch(spec: registry.ModelSpec, width: int, height: int) -> int:
    """按输出分辨率与模型体量定批：4K 输出下大模型取 1（显存压力），小模型可 4。"""
    out_px = width * height * spec.scale * spec.scale
    weight_mb = spec.path.stat().st_size / 1e6
    if out_px >= 7_500_000:      # ≈4K 输出
        return 4 if weight_mb <= 15 else 1
    if out_px >= 3_500_000:      # ≈1440p
        return 4 if weight_mb <= 15 else 2
    return 4


def _make_sr(spec: registry.ModelSpec, ctx: BuildContext,
             tone_match: bool = False) -> Any:
    from .sr import SuperRes
    return SuperRes(spec, use_amp=ctx.use_amp,
                    batch_size=auto_batch(spec, ctx.width, ctx.height),
                    compile_model=ctx.compile_model, use_trt=ctx.use_trt,
                    tone_match=tone_match)


# ---------------------------------------------------------------- 四个模块

class _Deh264Spec(ModuleSpec):
    def __init__(self):
        super().__init__(
            key="deh264", name="去压缩",
            description="修复 H.264 压缩伪影（块状/蚊噪），建议放在超分之前",
            order=0)

    def build(self, params: dict, ctx: BuildContext) -> BuiltModule:
        from .sr import SuperRes
        spec = registry.resolve("deh264")
        return BuiltModule(stages=[_make_sr(spec, ctx)])

    def summary(self, params: dict, src_h: int = 0) -> str:
        return "deH264 压缩伪影修复 1x"


class _DenoiseSpec(ModuleSpec):
    def __init__(self):
        super().__init__(
            key="denoise", name="降噪",
            description="去除传感器/胶片噪点（SCUNet），建议放在超分之前",
            order=1)

    def build(self, params: dict, ctx: BuildContext) -> BuiltModule:
        spec = registry.resolve("scunet")
        return BuiltModule(stages=[_make_sr(spec, ctx)])

    def summary(self, params: dict, src_h: int = 0) -> str:
        return "SCUNet 去噪 1x"


# 超分模型选项：第一项 auto = 按 内容类型 + 目标高度/源高度 自动选（元组 = (显示, 值)）
_SR_MODEL_CHOICES = [
    ("自动（按目标分辨率选）", "auto"),
    ("NomosUni 2x（真人）", "nomosuni"),
    ("AnimeJaNai 2x（动漫）", "anime"),
    ("Nomos8k 4x（弱去噪·真人）", "nomos8k-weak"),
    ("ClearReality 4x", "clearreality"),
    ("UltraSharp 4x", "ultrasharp"),
    ("Nomos8k 4x（强去噪）", "nomos8k"),
    ("RealESRGAN 4x", "x4plus"),
]
_SR_DISPLAY = {v: d for d, v in _SR_MODEL_CHOICES}

_TARGET_CHOICES = [
    ("3840x2160（4K UHD·推荐）", "3840x2160"),
    ("2560x1440（2K QHD）", "2560x1440"),
    ("1920x1080（FHD）", "1920x1080"),
    ("与源相同（仅增强不放大）", ""),
]
_TARGET_DISPLAY = {v: d for d, v in _TARGET_CHOICES}
_TARGET_SHORT = {"3840x2160": "4K", "2560x1440": "2K", "1920x1080": "1080p", "": "与源相同"}
_TARGET_H = {"3840x2160": 2160, "2560x1440": 1440, "1920x1080": 1080}


def effective_model(params: dict, source_h: int = 0) -> str:
    """解析超分实际使用的模型键。

    显式指定优先；auto 按 目标高度/源高度 比例选：≤2.5 用 2x 系（一步到位），
    更大用 4x 系（2x 不够高，如 720p→4K 比例 3）。动漫目前仅 2x 系模型。
    """
    key = str(params.get("model", "auto"))
    if key != "auto":
        return key
    content = str(params.get("content", "real"))
    if content == "anime":
        return "anime"
    target_h = _TARGET_H.get(str(params.get("target_res") or ""))
    if target_h is None or source_h <= 0:   # 与源相同 / 源未知 → 2x 系
        return "nomosuni"
    return "nomosuni" if target_h / source_h <= 2.5 else "nomos8k-weak"


class _SuperResSpec(ModuleSpec):
    def __init__(self):
        super().__init__(
            key="sr", name="超分辨率",
            description="放大并重建细节：选目标分辨率，系统自动配模型倍数"
                        "（定版快速档：2x 模型 + 色调对齐 + 锐化 0.6 + 4K）",
            order=2,
            params=[
                ParamSpec("content", "内容类型", "choice", "real",
                          [("真人（推荐）", "real"), ("动漫", "anime")],
                          tip="决定自动模型从哪个系列选"),
                ParamSpec("target_res", "目标分辨率", "choice", "3840x2160", _TARGET_CHOICES,
                          tip="建议对齐显示设备原生分辨率；目标高度超过源 2.5 倍时"
                              "自动换 4x 模型（如 720p→4K）"),
                ParamSpec("sharpen", "锐化补偿", "float", 0.6,
                          min=0, max=2, step=0.1,
                          tip="0.4-0.6 合适区间，1.0 以上易过头"),
                ParamSpec("tone_match", "色调对齐源片", "bool", True,
                          tip="修正部分模型固有偏暗/发灰"),
                ParamSpec("model", "模型（高级）", "choice", "auto", _SR_MODEL_CHOICES,
                          tip="默认自动：按 内容+目标高度/源高度 选倍数；"
                              "手动指定后不再自动切换"),
            ])

    def build(self, params: dict, ctx: BuildContext) -> BuiltModule:
        from .gpu_post import GpuPost
        spec = registry.resolve(effective_model(params, ctx.height))
        stages = [_make_sr(spec, ctx, tone_match=bool(params["tone_match"]))]
        if params["target_res"]:
            w, _, h = str(params["target_res"]).partition("x")
        else:  # 与源相同：模型放大后再缩回，仅取增强效果
            w, h = ctx.width, ctx.height
        stages.append(GpuPost(int(w), int(h), sharpen=float(params["sharpen"])))
        return BuiltModule(stages=stages)

    def summary(self, params: dict, src_h: int = 0) -> str:
        if str(params["model"]) == "auto" and not src_h:
            model_disp = "自动模型"
        else:
            model_disp = _SR_DISPLAY.get(effective_model(params, src_h),
                                         effective_model(params, src_h)).split("（")[0]
        content = "动漫" if params["content"] == "anime" else "真人"
        target = _TARGET_SHORT.get(params["target_res"], params["target_res"])
        return f"{content} · {model_disp} · {target} · 锐化{params['sharpen']:.1f}"


class _InterpSpec(ModuleSpec):
    def __init__(self):
        super().__init__(
            key="interp", name="插帧",
            description="RIFE 光流插帧，帧率翻倍（跨帧时序操作，恒在链尾执行）",
            order=3, singleton=True, always_last=True,
            params=[
                ParamSpec("factor", "倍率", "choice", 2,
                          [("2x（帧率翻倍）", 2)]),
                ParamSpec("rife_scale", "光流尺度", "choice", 0,
                          [("自动（≥1440p 用 UHD 模式）", 0),
                           ("1.0（标准）", 1.0),
                           ("0.5（UHD，省 4 倍算力）", 0.5)],
                          tip="UHD 模式是官方推荐，高分辨率下约省 4 倍算力"),
            ])

    def build(self, params: dict, ctx: BuildContext) -> BuiltModule:
        from .interp import RIFEInterpolator
        scale = float(params["rife_scale"])
        if scale <= 0:  # auto：按插帧实际处理的分辨率（含上游 SR 放大）判断
            scale = 0.5 if ctx.height * ctx.upstream_scale >= 1400 else 1.0
        return BuiltModule(
            interpolator=RIFEInterpolator(device="cuda", scale=scale, use_trt=ctx.use_trt))

    def summary(self, params: dict, src_h: int = 0) -> str:
        scale = float(params["rife_scale"])
        s = "RIFE 2x · " + ("UHD 模式" if scale == 0.5 else "标准" if scale == 1.0 else "自动尺度")
        return s


SPECS: dict[str, ModuleSpec] = {s.key: s for s in
                                (_Deh264Spec(), _DenoiseSpec(), _SuperResSpec(), _InterpSpec())}


def get(key: str) -> ModuleSpec:
    return SPECS[key]


def available() -> list[ModuleSpec]:
    """按默认链序返回全部模块定义。"""
    return sorted(SPECS.values(), key=lambda s: s.order)


# ---------------------------------------------------------------- 实例与链

@dataclass
class ModuleInstance:
    key: str
    params: dict = field(default_factory=dict)
    enabled: bool = True
    src_h: int = 0                 # 源高度运行时上下文（auto 模型解析/摘要用），不入 JSON

    def __post_init__(self):
        spec = SPECS[self.key]
        self.params = {p.key: p.coerce(self.params.get(p.key, p.default))
                       for p in spec.params}

    @property
    def spec(self) -> ModuleSpec:
        return SPECS[self.key]

    def summary(self) -> str:
        return self.spec.summary(self.params, self.src_h)

    def to_dict(self) -> dict:
        return {"key": self.key, "params": self.params, "enabled": self.enabled}

    @staticmethod
    def from_dict(d: dict) -> "ModuleInstance":
        return ModuleInstance(d["key"], dict(d.get("params") or {}), bool(d.get("enabled", True)))


def default_chain(content: str = "real") -> list[ModuleInstance]:
    """新任务默认链：定版快速档（模型 auto 按内容+目标分辨率解析，插帧默认关）。"""
    sr = ModuleInstance("sr", {"content": content})
    interp = ModuleInstance("interp")
    interp.enabled = False
    return [sr, interp]


def build_chain(instances: list[ModuleInstance],
                ctx: BuildContext) -> tuple[list, Interpolator | None]:
    """实例链 → (stages, interpolator)。禁用模块跳过；插帧提取为 interpolator。"""
    ordered = sorted(instances, key=lambda m: m.spec.order)
    # 插帧 auto 尺度需要知道上游超分的实际放大倍数（目标高度/源高度）
    for inst in ordered:
        if inst.enabled and inst.key == "sr":
            target = str(inst.params["target_res"] or "")
            ctx.upstream_scale = (int(target.partition("x")[2]) // ctx.height
                                  if target else 1)
            break
    stages: list = []
    interpolator = None
    for inst in ordered:
        if not inst.enabled:
            continue
        built = inst.spec.build(inst.params, ctx)
        stages.extend(built.stages)
        if built.interpolator is not None:
            if interpolator is not None:
                raise ValueError("插帧模块只能添加一个")
            interpolator = built.interpolator
    return stages, interpolator
