"""锐化后处理：unsharp mask 补偿运动模糊帧的观感（SR 放大会等比放大源的运动模糊）。

GPU 实现（torch conv2d，4K 下 <5ms/帧且吃批处理红利）；CPU 版曾是 6 倍速度杀手。
threshold：低于该幅度的差异不锐化，避免放大编码噪声。
"""
import numpy as np
import torch

from .pipeline import Frame


class Sharpen:
    def __init__(self, amplitude: float = 0.8, radius: float = 1.6,
                 threshold: int = 2, device: str = "cuda"):
        self.amplitude = amplitude
        self.threshold = threshold
        self.device = device
        self.batch = 4
        # 高斯核（sigma=radius），3x3x1x1 depthwise
        k = self._gaussian_kernel(radius)
        self.kernel = k

    @staticmethod
    def _gaussian_kernel(sigma: float) -> torch.Tensor:
        r = max(1, int(sigma * 3))
        xs = torch.arange(-r, r + 1, dtype=torch.float32)
        g = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        k2 = g[:, None] @ g[None, :]
        return k2.expand(3, 1, k2.shape[0], k2.shape[1]).contiguous()

    @torch.inference_mode()
    def __call__(self, frames: list[Frame]) -> list[Frame]:
        bgr = np.stack(frames)
        t = torch.from_numpy(bgr).to(self.device)
        t = t.permute(0, 3, 1, 2).float()
        blur = torch.nn.functional.conv2d(t, self.kernel.to(self.device),
                                          padding=self.kernel.shape[-1] // 2, groups=3)
        diff = t - blur
        diff = torch.where(diff.abs() < self.threshold, torch.zeros_like(diff), diff)
        out = t + self.amplitude * diff
        out = out.clamp_(0, 255).round_().byte()
        arr = out.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return [arr[i] for i in range(arr.shape[0])]
