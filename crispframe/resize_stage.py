"""目标分辨率缩放 stage：超分后缩到标准输出分辨率（如 3840x2160）。

原因：4x 模型输出的超必要分辨率（如 5K）在 4K 屏上需播放器实时缩小，
缩放算法差异会导致整体发软；先高质量缩小到目标分辨率再锐化，观感最锐。
"""
import cv2

from .pipeline import Frame


class ResizeStage:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __call__(self, frames: list[Frame]) -> list[Frame]:
        out = []
        for f in frames:
            if f.shape[1] == self.width and f.shape[0] == self.height:
                out.append(f)
            else:
                out.append(cv2.resize(f, (self.width, self.height),
                                      interpolation=cv2.INTER_AREA))
        return out
