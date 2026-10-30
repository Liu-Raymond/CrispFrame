"""AI 模块注册表：GUI v2 模块链的单一事实来源。

模块 = 用户可添加/排序/开关的增强环节。五个内置模块：
去压缩 → 降噪 → 超分 → 插帧/平滑（默认顺序；时序模块实际在管线链尾执行）。

- ModuleSpec：模块类型定义（参数 schema + 构建）
- ModuleInstance：链上的一个模块实例（参数值 + 开关），可 JSON 序列化
- build_chain()：把实例链组装成 pipeline.run 需要的 (stages, interpolator, vf_pre)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2

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
    upstream_scale: float = 1.0   # 链上超分模块的输出高度/源高度（可为 <1 的缩小档），
    # build_chain 按最终输出分辨率填充；插帧/平滑 auto 尺度判定 = 源高×此值（即实际处理高度）


@dataclass
class BuiltModule:
    stages: list = field(default_factory=list)          # list[BatchStage]
    interpolator: Interpolator | None = None
    vf_pre: str = ""      # 编码侧前置滤镜（tmix 混合+fps 回采），build_chain 收集给 codec


# 平滑三档：时域混合窗从轻到重（中档 = 2026-10 用户验收定版，等效 180° 快门）
_SMOOTH_MIX = {
    "light":  "tmix=frames=2:weights='1 3'",   # 当前帧主导，轻微平滑
    "medium": "tmix=frames=2",                  # 相邻均混，等效 180° 快门
    "heavy":  "tmix=frames=3",                  # 更宽时域窗，最强平滑（运动模糊加重）
}
_SMOOTH_LABEL = {"light": "轻度", "medium": "中度", "heavy": "重度"}


class StrengthBlend:
    """AI 修复输出与原帧按强度 alpha 混合（1.0 = 纯 AI 输出）。

    修复类模型没有强度旋钮，alpha 混合是通用的"强度"等效实现：
    0.6 = 60% 模型输出 + 40% 原帧。strength <= 0 时直接透传原帧
    （等价于模块关闭），>= 0.995 时跳过混合（纯 AI 输出）。"""

    def __init__(self, inner, strength: float):
        self.inner = inner
        self.strength = max(0.0, min(1.0, float(strength)))

    def __call__(self, frames):
        a = self.strength
        if a <= 0.005:
            return frames                     # 强度 0：原帧直通
        out = self.inner(frames)
        if a >= 0.995:
            return out
        # 帧数一致性是 1x 修复模型的契约（若未来包装会变帧数的 stage，这里会拦截）
        assert len(out) == len(frames), "StrengthBlend: inner 输出帧数与输入不一致"
        return [cv2.addWeighted(o, a, i, 1.0 - a, 0) for o, i in zip(out, frames)]


# ---------------------------------------------------------------- ModuleSpec


@dataclass
class ModuleSpec:
    key: str
    name: str                     # 显示名
    description: str
    order: int                    # 默认链序（去压缩 0 → 降噪 1 → 超分 2 → 插帧 3 → 平滑 4）
    params: list[ParamSpec] = field(default_factory=list)
    singleton: bool = False       # 链中最多一个（插帧/平滑这类时序模块）
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


# ---------------------------------------------------------------- 五个模块

class _Deh264Spec(ModuleSpec):
    def __init__(self):
        super().__init__(
            key="deh264", name="去压缩",
            description="修复 H.264 压缩伪影（块状/蚊噪），建议放在超分之前",
            order=0,
            params=[
                ParamSpec("strength", "强度", "float", 1.0,
                          min=0.0, max=1.0, step=0.05,
                          tip="1.0 = 完全 AI 修复；调低 = 与原帧混合，去痕迹更保守"),
            ])

    def build(self, params: dict, ctx: BuildContext) -> BuiltModule:
        strength = float(params["strength"])
        if strength <= 0.005:
            return BuiltModule(stages=[])     # 强度 0：等价关闭，模型都不加载
        spec = registry.resolve("deh264")
        return BuiltModule(stages=[StrengthBlend(_make_sr(spec, ctx), strength)])

    def summary(self, params: dict, src_h: int = 0) -> str:
        a = float(params.get("strength", 1.0))
        return f"deH264 压缩伪影修复 · 强度{a:.0%}"


class _DenoiseSpec(ModuleSpec):
    def __init__(self):
        # 降噪模型下拉按磁盘文件自动出现。GAN 变体经 A/B 目测与 PSNR 无可见
        # 差异且慢 2.5 倍（8s vs 3.2s/帧 @4K）已移除；如需找回：
        # huggingface.co/krnl/ScuNET/resolve/main/scunet_color_real_gan.pth (72MB)
        choices = [("SCUNet PSNR（保守·干净）", "scunet")]
        if (registry.PRETRAINED_DIR / "NAFNet-SIDD-width32.pth").exists():
            choices.append(("NAFNet SIDD 32（轻量·4K 可用）", "nafnet-sidd"))
        super().__init__(
            key="denoise", name="降噪",
            description="去除传感器/胶片噪点（SCUNet），建议放在超分之前",
            order=1,
            params=[
                ParamSpec("model", "模型", "choice", "scunet", choices,
                          tip="PSNR 保守干净，去噪同时轻微增强细节"),
                ParamSpec("strength", "强度", "float", 1.0,
                          min=0.0, max=1.0, step=0.05,
                          tip="1.0 = 完全 AI 降噪；调低 = 与原帧混合，保细节更保守"),
            ])

    def build(self, params: dict, ctx: BuildContext) -> BuiltModule:
        strength = float(params["strength"])
        if strength <= 0.005:
            return BuiltModule(stages=[])
        spec = registry.resolve(params["model"])
        return BuiltModule(stages=[StrengthBlend(_make_sr(spec, ctx), strength)])

    def summary(self, params: dict, src_h: int = 0) -> str:
        a = float(params.get("strength", 1.0))
        names = {"nafnet-sidd": "NAFNet SIDD 32"}
        model = names.get(params.get("model"), "SCUNet")
        tail = f" · 强度{a:.0%}" if a < 0.995 else ""
        return f"{model} 去噪{tail}"


# 超分模型选项：第一项 auto = 按 内容类型 + 目标高度/源高度 自动选（元组 = (显示, 值)）
# 模型经同帧 A/B（画质目测 + 速度 + 显存）筛选：无可辨差异的 UltraSharp/RealESRGAN
# （慢 60 倍/显存 4 倍）与 ClearReality（与 Nomos8k-weak 定位重叠）已移除出下拉；
# registry 条目保留供 CLI 高级用法，权重文件留在磁盘可随时恢复
_SR_MODEL_CHOICES = [
    ("自动（按目标分辨率选）", "auto"),
    ("NomosUni 2x（真人）", "nomosuni"),
    ("AnimeJaNai 2x（动漫）", "anime"),
    ("Nomos8k 4x（弱去噪·真人）", "nomos8k-weak"),
]
_SR_DISPLAY = {v: d for d, v in _SR_MODEL_CHOICES}

_TARGET_CHOICES = [
    ("3840x2160（4K UHD·推荐）", "3840x2160"),
    ("2560x1440（2K QHD）", "2560x1440"),
    ("1920x1080（FHD）", "1920x1080"),
    ("2x（源×2 直出）", "x2"),
    ("4x（源×4 直出）", "x4"),
    ("与源相同（仅增强不放大）", ""),
]
_TARGET_DISPLAY = {v: d for d, v in _TARGET_CHOICES}
_TARGET_SHORT = {"3840x2160": "4K", "2560x1440": "2K", "1920x1080": "1080p",
                 "x2": "2x直出", "x4": "4x直出", "": "与源相同"}
_TARGET_H = {"3840x2160": 2160, "2560x1440": 1440, "1920x1080": 1080}


def resolve_target(target_res: str, src_w: int, src_h: int) -> tuple[int, int] | None:
    """目标分辨率统一解析（build/摘要/GUI 共用）：
    "xN" → 源×N 整数倍直出；"WxH" → 固定尺寸；空 → None（与源相同）。"""
    t = str(target_res or "")
    tl = t.lower()
    if tl.startswith("x") and tl[1:].isdigit():
        n = int(tl[1:])
        return src_w * n, src_h * n
    w, _, h = tl.partition("x")
    if w.isdigit() and h.isdigit():
        return int(w), int(h)
    return None


def target_scale(target_res: str, src_h: int) -> int:
    """源 → 最终输出的总放大倍数（整数近似）。仅诊断/展示用；
    插帧 auto 尺度判定走 build_chain 的真实输出高度比（upstream_scale）。"""
    t = str(target_res or "")
    if t.startswith("x") and t[1:].isdigit():
        return int(t[1:])
    if src_h > 0 and t.lower().partition("x")[2].isdigit():
        return max(1, int(t.lower().partition("x")[2]) // src_h)
    return 1


def effective_model(params: dict, source_h: int = 0) -> str:
    """解析超分实际使用的模型键。

    显式指定优先；auto 按 目标高度/源高度 比例选：≤2.5 用 2x 系（一步到位），
    更大用 4x 系（2x 不够高，如 720p→4K 比例 3）。倍数档（x2/x4）按倍数折算
    目标高度走同一比例逻辑；动漫目前仅 2x 系模型（x4 时 2x 增强 + bicubic 上采）。
    """
    key = str(params.get("model", "auto"))
    if key != "auto":
        return key
    content = str(params.get("content", "real"))
    if content == "anime":
        return "anime"
    target_res = str(params.get("target_res") or "")
    if target_res.startswith("x") and target_res[1:].isdigit():
        target_h = source_h * int(target_res[1:]) if source_h > 0 else None
    else:
        h_part = target_res.lower().partition("x")[2]
        target_h = (int(h_part) if h_part.isdigit()      # 任意 WxH 都按高度比例选
                    else _TARGET_H.get(target_res))
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
                              "自动换 4x 模型（如 720p→4K）。x2/x4 = 整数倍直出："
                              "模型输出即最终输出（不缩放）。注意 ×4 的输出是源的 4 倍"
                              "（1440p 源→8K 内可行；4K 源×4=16K 会显存不足，勿用）"),
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
        w_h = resolve_target(params["target_res"], ctx.width, ctx.height)
        if w_h is None:  # 与源相同：模型放大后再缩回，仅取增强效果
            w, h = ctx.width, ctx.height
        else:
            w, h = w_h   # 固定尺寸或 xN 直出（等尺寸时 GpuPost 自动跳过缩放）
        stages.append(GpuPost(int(w), int(h), sharpen=float(params["sharpen"])))
        return BuiltModule(stages=stages)

    def summary(self, params: dict, src_h: int = 0) -> str:
        if str(params["model"]) == "auto" and not src_h:
            model_disp = "自动模型"
        else:
            model_disp = _SR_DISPLAY.get(effective_model(params, src_h),
                                         effective_model(params, src_h)).split("（")[0]
        content = "动漫" if params["content"] == "anime" else "真人"
        tr = str(params["target_res"])
        if tr.startswith("x") and tr[1:].isdigit():
            n = int(tr[1:])
            target = (f"{_TARGET_SHORT.get(tr, tr)}"
                      + (f"·{src_h * n}p" if src_h > 0 else ""))
        else:
            target = _TARGET_SHORT.get(tr, tr)
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


class _SmoothSpec(ModuleSpec):
    def __init__(self):
        super().__init__(
            key="smooth", name="平滑",
            description="RIFE 2x 插帧 + 时域运动补偿混合后回采源帧率"
                        "（等效快门平滑，抹平内容本身的时域顿挫；输出帧率不变）",
            order=4, singleton=True, always_last=True,
            params=[
                ParamSpec("level", "平滑程度", "choice", "medium",
                          [("轻度（当前帧主导）", "light"),
                           ("中度（180° 快门·推荐）", "medium"),
                           ("重度（最强平滑）", "heavy")],
                          tip="中度 = 验收定版配方；重度运动模糊明显，先试中度"),
            ])

    def build(self, params: dict, ctx: BuildContext) -> BuiltModule:
        from .interp import RIFEInterpolator
        # auto 尺度与插帧模块一致；TRT 撕裂 bug 未解，平滑恒走 torch bf16
        scale = 0.5 if ctx.height * ctx.upstream_scale >= 1400 else 1.0
        mix = _SMOOTH_MIX[params["level"]]
        return BuiltModule(
            interpolator=RIFEInterpolator(device="cuda", scale=scale, use_trt=False),
            vf_pre=f"{mix},fps={ctx.fps:.6f}")

    def summary(self, params: dict, src_h: int = 0) -> str:
        return f"RIFE 2x + 混合回采 · {_SMOOTH_LABEL[params['level']]}"


SPECS: dict[str, ModuleSpec] = {s.key: s for s in
                                (_Deh264Spec(), _DenoiseSpec(), _SuperResSpec(),
                                 _InterpSpec(), _SmoothSpec())}


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


def resolve_temporal_conflict(instances: list[ModuleInstance]) -> None:
    """链级互斥收敛（就地修改）：插帧/平滑同时启用时保留插帧（链序靠前的原功能）。
    用于加载外部配置（profile JSON 可能含旧版/手编辑的非法组合）。"""
    seen = False
    for inst in sorted(instances, key=lambda m: m.spec.order):
        if inst.key in ("interp", "smooth") and inst.enabled:
            inst.enabled = not seen
            seen = True


def default_chain(content: str = "real") -> list[ModuleInstance]:
    """新任务默认链：定版快速档（模型 auto 按内容+目标分辨率解析，插帧默认关）。"""
    sr = ModuleInstance("sr", {"content": content})
    interp = ModuleInstance("interp")
    interp.enabled = False
    return [sr, interp]


def build_chain(instances: list[ModuleInstance],
                ctx: BuildContext) -> tuple[list, Interpolator | None, str]:
    """实例链 → (stages, interpolator, vf_pre)。禁用模块跳过；时序模块提取为
    interpolator，其编码侧前置滤镜（混合+回采）经 vf_pre 交给 codec。"""
    ordered = sorted(instances, key=lambda m: m.spec.order)
    # 插帧 auto 尺度判定依据 = 插帧实际处理的高度（超分最终输出高度）÷ 源高。
    # 用真实输出高度比而非整数倍：2160p 源选 1080p 缩小档时比值 0.5，
    # 按处理高度 1080 判定应用标准模式（1.0）而非 UHD（0.5）
    for inst in ordered:
        if inst.enabled and inst.key == "sr":
            w_h = resolve_target(str(inst.params["target_res"] or ""),
                                 ctx.width, ctx.height)
            out_h = w_h[1] if w_h else ctx.height
            ctx.upstream_scale = (out_h / ctx.height) if ctx.height else 1.0
            break
    stages: list = []
    interpolator = None
    vf_pre = ""
    for inst in ordered:
        if not inst.enabled:
            continue
        built = inst.spec.build(inst.params, ctx)
        stages.extend(built.stages)
        if built.interpolator is not None:
            if interpolator is not None:
                raise ValueError("插帧与平滑模块只能二选一（都是时序链尾操作）")
            interpolator = built.interpolator
        if built.vf_pre:
            vf_pre = built.vf_pre
    return stages, interpolator, vf_pre
