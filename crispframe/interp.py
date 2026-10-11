"""RIFE v4.26 插帧。权重 rife4.26.pkl（hzwer 发布），网络定义 vendor 自 Practical-RIFE (MIT)。

输入输出均为 HxWx3 uint8 BGR；内部按 RIFE 惯例转 0-1 RGB 并 pad 到 64 的倍数。
"""
import sys

import numpy as np
import torch

from ._rife.IFNet_HDv3 import IFNet
from .registry import PRETRAINED_DIR

_WEIGHTS = PRETRAINED_DIR / "rife4.26.pkl"


def _convert(param: dict) -> dict:
    # 训练时 DDP 保存的 key 带 "module." 前缀
    return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}


class _RifeExport(torch.nn.Module):
    """ONNX 导出专用：timestep 作为显式张量输入，绕开 python 标量分支与 trace 常量折叠。"""

    def __init__(self, net: IFNet, scale_list: list[float]):
        super().__init__()
        self.net = net
        self.scale_list = scale_list

    def forward(self, x, ts):
        return self.net(x, timestep=ts, scale_list=self.scale_list)[0]


class RIFEInterpolator:
    def __init__(self, device: str = "cuda", scale: float = 1.0, use_amp: bool = True,
                 use_trt: bool = True):
        if not _WEIGHTS.exists():
            raise FileNotFoundError(f"缺少 RIFE 权重: {_WEIGHTS}")
        self.device = device
        self.net = IFNet().to(device).eval()
        state = _convert(torch.load(_WEIGHTS, map_location="cpu"))
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        # heavy 版推理不用 teacher/caltime，缺这几个 key 属正常
        real_missing = [k for k in missing if "teacher" not in k and "caltime" not in k]
        if real_missing:
            raise RuntimeError(f"RIFE 权重缺 key: {real_missing[:5]}")
        # 实测 channels_last 对 RIFE（多级 interpolate、小通道）无收益，保持默认布局
        self.scale = scale
        # scale=0.5（4K 官方 UHD 推荐）：每级 block 在减半分辨率算光流，计算量约降 4 倍
        self.scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        # IFBlock 内部 interpolate(scale) 后还有 4 倍下/上采样，输入必须被 scale_list[0]*4 整除
        self.pad_div = max(64, int(16 / scale) * 4)
        self.use_amp = use_amp and device.startswith("cuda")
        self.amp_dtype = torch.bfloat16  # Blackwell 上 bf16 快且无 fp16 下溢问题
        self.runner = None
        # TRT 对本模型禁用：TensorRT 11.3 执行 ONNX GridSample 的坐标语义与 torch
        # 不一致（torch align_corners=True 的 backwarp 被错误采样），输出水平撕裂
        # （2026-10-11 排查：torch/ONNX/ORT 三方一致正确，仅 TRT 错；fp16/fp32、
        # 新旧引擎均复现）。SR 模型不受影响。待 TRT 修复后可重新启用。
        self.use_trt = False

    def _trt_engine(self, h: int, w: int):
        import tensorrt as _trt  # noqa: N813
        from .trt import get_engine
        key = f"rife426_{h}x{w}_s{self.scale}_b1_fp16_trt{_trt.__version__.replace('.', '')}"
        print(f"[crispframe] 构建/加载 RIFE TensorRT 引擎 {key} …", file=sys.stderr)
        wrap = _RifeExport(self.net, self.scale_list)
        # 导出固定用 CPU（cuda trace 会被常量折叠的 cpu 常量绊倒），engine 构建设备无关；
        # 注意 wrap.cpu() 会连带把 self.net 移到 CPU，finally 里必须移回
        wrap = wrap.cpu()
        try:
            x = torch.zeros(1, 6, h, w)
            ts = torch.full((1, 1, h, w), 0.5)
            runner = get_engine(wrap, (x, ts), key)
        finally:
            self.net.to(self.device)
        self.runner = runner
        return self.runner

    def _to_tensor(self, bgr: np.ndarray):
        # uint8 HWC BGR 直接上卡，BGR→RGB/归一化/pad 全在 GPU 做（省 CPU 3x24MB 拷贝）
        t = torch.from_numpy(bgr).to(self.device)
        # unsqueeze(0)：补 batch 维。缺了它张量是 3D（两帧会沿高度拼接、ts 形状错乱），
        # TRT 对 shape 不匹配只打日志不报错、静默输出全零帧（表现为插帧一黑一正常闪屏）
        t = t.unsqueeze(0).permute(0, 3, 1, 2).flip(1).float().div_(255.0)
        h, w = t.shape[-2:]
        ph = (self.pad_div - h % self.pad_div) % self.pad_div
        pw = (self.pad_div - w % self.pad_div) % self.pad_div
        if ph or pw:
            t = torch.nn.functional.pad(t, (0, pw, 0, ph))
        return t, h, w

    @torch.inference_mode()
    def midframe(self, a_bgr: np.ndarray, b_bgr: np.ndarray) -> np.ndarray:
        i0, h, w = self._to_tensor(a_bgr)
        i1, _, _ = self._to_tensor(b_bgr)
        if self.runner is not None:
            x = torch.cat((i0, i1), 1)
            ts = torch.full_like(i0[:, :1], 0.5)
            mid = self.runner([x, ts])[0]
        elif self.use_trt and self.device.startswith("cuda"):
            try:
                self._trt_engine(i0.shape[-2], i0.shape[-1])
                return self.midframe(a_bgr, b_bgr)
            except Exception as e:
                print(f"[crispframe] RIFE TensorRT 不可用（{type(e).__name__}），回退 torch", file=sys.stderr)
                self.use_trt = False
                return self.midframe(a_bgr, b_bgr)
        else:
            with torch.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                mid = self.net(torch.cat((i0, i1), 1), timestep=0.5, scale_list=self.scale_list)[0]
        # 裁剪 pad、RGB→BGR、量化、contiguous 化全在 GPU，d2h 只搬一次 uint8
        mid = mid[..., :h, :w].flip(1).clamp_(0, 1).mul_(255.0).round_().byte()
        arr = mid.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
        return arr
