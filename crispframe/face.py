"""面部增强：YuNet 检测人脸 → CodeFormer 只修复人脸区域 → 羽化贴回。

- 检测在半分辨率上做（4K 帧检测开销降 4 倍，脸在 4K 下足够大）
- 人脸框坐标量化 + 跨帧 EMA 平滑，抑制修复区域的逐帧抖动
- 无脸/小脸帧直接透传
"""
import cv2
import numpy as np
import torch
from spandrel import ModelLoader
from spandrel_extra_arches import install

from .registry import PRETRAINED_DIR

install()  # 注册 CodeFormer 等额外架构

_WEIGHTS = PRETRAINED_DIR / "codeformer.pth"
_YUNET = PRETRAINED_DIR / "yunet_2023mar.onnx"

_FACE_MIN = 48        # 小于该边长(px)的人脸不处理
_CROP_PAD = 0.25      # 脸框扩边比例（覆盖额头/下巴）
_QUANT = 4            # 框坐标量化(px)，配合 EMA 抑制抖动
_EMA = 0.6            # 框平滑系数（对上一帧同位置框）


class FaceEnhance:
    def __init__(self, fidelity: float = 0.7, device: str = "cuda",
                 downscale_detect: int = 2):
        if not _WEIGHTS.exists():
            raise FileNotFoundError(f"缺少 CodeFormer 权重: {_WEIGHTS}")
        if not _YUNET.exists():
            raise FileNotFoundError(f"缺少 YuNet 检测器: {_YUNET}")
        self.device = device
        self.fidelity = fidelity
        self.downscale_detect = downscale_detect
        self.model = ModelLoader().load_from_file(str(_WEIGHTS)).model.to(device).eval()
        self._detectors: dict[tuple[int, int], cv2.FaceDetectorYN] = {}
        self._prev_boxes: list[tuple[int, int, int, int]] = []

    def _detector(self, w: int, h: int) -> cv2.FaceDetectorYN:
        key = (w, h)
        if key not in self._detectors:
            det = cv2.FaceDetectorYN.create(str(_YUNET), "", (w, h),
                                            score_threshold=0.6)
            self._detectors[key] = det
        return self._detectors[key]

    def _detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]
        # 检测降采样自适应：目标是让检测图短边 ~900px（YuNet 对 ≥100px 脸很稳）
        ds = max(1, min(h, w) // 900)
        small = cv2.resize(frame_bgr, (w // ds, h // ds)) if ds > 1 else frame_bgr
        det = self._detector(small.shape[1], small.shape[0])
        _, faces = det.detect(small)
        boxes = []
        if faces is not None:
            for f in faces:
                x, y, fw, fh = (int(v) * ds for v in f[:4])
                if min(fw, fh) < _FACE_MIN * ds:
                    continue
                boxes.append((x, y, fw, fh))
        # 与上一帧的框做 EMA 平滑（按最近邻配对），抑制逐帧抖动
        smoothed = []
        for x, y, fw, fh in boxes:
            qx, qy, qfw, qfh = (v // _QUANT * _QUANT for v in (x, y, fw, fh))
            best = None
            for px, py, pfw, pfh in self._prev_boxes:
                if abs(px - qx) < fw and abs(py - qy) < fh:
                    d = abs(px - qx) + abs(py - qy)
                    if best is None or d < best[0]:
                        best = (d, (px, py, pfw, pfh))
            if best is not None:
                px, py, pfw, pfh = best[1]
                qx = int(round(qx * _EMA + px * (1 - _EMA)))
                qy = int(round(qy * _EMA + py * (1 - _EMA)))
            smoothed.append((qx, qy, qfw, qfh))
        self._prev_boxes = smoothed
        return smoothed

    def _fix_face(self, frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> None:
        h, w = frame_bgr.shape[:2]
        x, y, fw, fh = box
        # 扩边并夹到帧内
        pw, ph = int(fw * _CROP_PAD), int(fh * _CROP_PAD)
        x0, y0 = max(0, x - pw), max(0, y - ph)
        x1, y1 = min(w, x + fw + pw), min(h, y + fh + ph)
        crop = frame_bgr[y0:y1, x0:x1]
        if min(crop.shape[:2]) < 16:
            return
        ch, cw = crop.shape[:2]
        t = torch.from_numpy(crop[:, :, ::-1].copy())
        t = t.permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(self.device)
        t = torch.nn.functional.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(t, weight=self.fidelity)[0]
        out = out.squeeze(0).clamp_(0, 1).mul_(255.0).round_().byte()
        fixed = out.permute(1, 2, 0).contiguous().cpu().numpy()[:, :, ::-1]
        # 在 512 空间做椭圆羽化混合（大图 float 运算是 4K 贴回的耗时大户），再一次性 resize 贴回
        crop512 = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
        mask = np.zeros((512, 512), np.uint8)
        cv2.ellipse(mask, (256, 256), (236, 236), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=21)
        m3 = mask.astype(np.float32)[..., None] / 255.0
        blend = (fixed.astype(np.float32) * m3 + crop512.astype(np.float32) * (1 - m3)).astype(np.uint8)
        frame_bgr[y0:y1, x0:x1] = cv2.resize(blend, (cw, ch), interpolation=cv2.INTER_LANCZOS4)

    @torch.inference_mode()
    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        for frame in frames:
            boxes = self._detect(frame)
            for box in boxes:
                self._fix_face(frame, box)
        return frames
