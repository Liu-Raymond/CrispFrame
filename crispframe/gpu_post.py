"""超分后的 GPU 复合后处理节点：resize → sharpen 一气呵成。

全部在显存内完成，超分输出不落回 CPU，避免 4K 帧多次 PCIe 往返
（每个独立 stage 的 numpy 往返是 16MB/帧级别的浪费）。
tone-match 由 SuperRes 内部 GPU 逻辑承担（有输入帧参考），不在此节点。
"""
import numpy as np
import torch


class GpuPost:
    def __init__(self, width: int, height: int, sharpen: float = 0.8,
                 sharpen_radius: float = 1.6, sharpen_threshold: int = 2,
                 device: str = "cuda"):
        self.width, self.height = width, height
        self.sharpen = sharpen
        self.sharpen_threshold = sharpen_threshold
        self.device = device
        self.kernel = self._gaussian(sharpen_radius)

    @staticmethod
    def _gaussian(sigma: float) -> torch.Tensor:
        r = max(1, int(sigma * 3))
        xs = torch.arange(-r, r + 1, dtype=torch.float32)
        g = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        k2 = g[:, None] @ g[None, :]
        return k2.expand(3, 1, k2.shape[0], k2.shape[1]).contiguous()

    @torch.inference_mode()
    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        bgr = np.stack(frames)
        t = torch.from_numpy(bgr).to(self.device, non_blocking=True)
        t = t.permute(0, 3, 1, 2).float()
        if t.shape[-2:] != (self.height, self.width):
            t = torch.nn.functional.interpolate(t, size=(self.height, self.width),
                                                mode="bicubic", align_corners=False)
        if self.sharpen > 0:
            blur = torch.nn.functional.conv2d(t, self.kernel.to(self.device),
                                              padding=self.kernel.shape[-1] // 2, groups=3)
            diff = t - blur
            diff = torch.where(diff.abs() < self.sharpen_threshold,
                               torch.zeros_like(diff), diff)
            t = t + self.sharpen * diff
        t = t.clamp_(0, 255).round_().byte()
        arr = t.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return [arr[i] for i in range(arr.shape[0])]
