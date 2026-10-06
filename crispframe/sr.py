"""超分推理：spandrel 加载任意社区超分模型（ESRGAN/SPAN/Compact 等架构）。

性能三板斧：多帧批处理、channels_last 内存布局、torch.compile（CUDA Graphs）。
"""
import sys

import numpy as np
import torch
from spandrel import ImageModelDescriptor, ModelLoader

from .registry import ModelSpec


class SuperRes:
    def __init__(self, spec: ModelSpec, device: str = "cuda", use_amp: bool = True,
                 batch_size: int = 4, compile_model: bool = False, channels_last: bool = True,
                 use_trt: bool = True, tone_match: bool = False):
        model = ModelLoader().load_from_file(str(spec.path))
        if not isinstance(model, ImageModelDescriptor):
            raise TypeError(f"{spec.path} 不是图像超分模型")
        self.spec = spec
        self.model: ImageModelDescriptor = model.to(device).eval()
        self.device = device
        self.scale = model.scale
        self.batch_size = max(1, batch_size)
        self.use_amp = use_amp and device.startswith("cuda")
        self.amp_dtype = torch.bfloat16  # Blackwell 上 bf16 与 fp16 同速且无下溢
        if channels_last:
            self.model.model.to(memory_format=torch.channels_last)
        # compile 内部 nn.Module（descriptor 本身是普通包装对象）
        self._fwd = self.model.model
        self._compiled = False
        if compile_model:
            # dynamic=False + 固定 batch：走 CUDA Graphs，消除 kernel launch 开销；
            # 个别架构（如 SPAN）与 channels_last 组合会触发 inductor 断言，调用时自动降级
            self._fwd = torch.compile(self._fwd, dynamic=False, mode="reduce-overhead")
            self._compiled = True
        self.use_trt = use_trt and device.startswith("cuda")
        self.runner = None  # TensorRT 引擎，首帧按实际 shape 构建并缓存
        # 部分 SR 模型有固有 tone shift（如 Nomos8k 系偏暗 ~2%）；开启后把输出
        # 亮度统计（luma 均值/标准差）逐帧对齐输入源，消除观感色偏
        self.tone_match = tone_match

    def _trt_engine(self, t: torch.Tensor):
        import tensorrt as _trt  # noqa: N813
        from .trt import get_engine
        b, _, h, w = t.shape
        key = f"{self.spec.name}_{h}x{w}_b{b}_fp16_trt{_trt.__version__.replace('.', '')}"
        print(f"[crispframe] 构建/加载 {self.spec.name} TensorRT 引擎 {key} …", file=sys.stderr)
        self.runner = get_engine(self.model.model, (t.detach().clone(),), key)
        return self.runner

    def _torch_forward(self, t: torch.Tensor) -> torch.Tensor:
        try:
            with torch.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                return self._fwd(t)
        except (AssertionError, RuntimeError) as e:
            if not self._compiled:
                raise
            print(f"[crispframe] {self.spec.name}: torch.compile 不兼容（{type(e).__name__}），已回退 eager",
                  file=sys.stderr)
            self._compiled = False
            self._fwd = self.model.model
            with torch.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                return self._fwd(t)

    @torch.inference_mode()
    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """B 帧 HxWx3 uint8 BGR 进，B 帧 (scale·H)x(scale·W)x3 uint8 BGR 出。"""
        bgr = np.stack(frames)  # B,H,W,3
        t = torch.from_numpy(bgr).to(self.device)
        # BGR→RGB 在 GPU 上做，省 CPU 拷贝
        t = t.permute(0, 3, 1, 2).flip(1).float().div_(255.0)
        if self.runner is not None or self.use_trt:
            # TRT 按标准 NCHW 解读 data_ptr；channels_last 内存直接喂会得到乱序输出
            t = t.contiguous()
        else:
            t = t.contiguous(memory_format=torch.channels_last)
        if self.runner is not None:
            out = self.runner([t])[0]
        elif self.use_trt:
            try:
                self._trt_engine(t)
                out = self.runner([t])[0]
            except Exception as e:
                print(f"[crispframe] {self.spec.name}: TensorRT 不可用（{type(e).__name__}），回退 torch",
                      file=sys.stderr)
                self.use_trt = False
                out = self._torch_forward(t)
        else:
            out = self._torch_forward(t)
        if self.tone_match:
            ref = torch.nn.functional.interpolate(t, size=out.shape[-2:], mode="bicubic",
                                                  align_corners=False)
            # bt709 luma；仅对齐亮度均值/标准差，不动色彩
            w = torch.tensor([0.2126, 0.7152, 0.0722], device=out.device).view(1, 3, 1, 1)
            out_y = (out * w).sum(1, keepdim=True)
            ref_y = (ref * w).sum(1, keepdim=True)
            gain = (ref_y.std() / (out_y.std() + 1e-6)).clamp(0.8, 1.25)
            bias = (ref_y.mean() - out_y.mean() * gain).clamp(-0.15, 0.15)
            out = out * gain + bias
            out = out.clamp_(0, 1)
        out = out.flip(1).float().clamp_(0, 1).mul_(255.0).round_().byte()
        # contiguous 化在 GPU 上做，保证 d2h 后可直接 memoryview/写管道
        arr = out.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return [arr[i] for i in range(arr.shape[0])]
