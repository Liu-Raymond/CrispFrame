"""CrispFrame 应用图标生成：模糊→锐利的视频帧 + 星芒，多尺寸 ico。

输出：launcher/crispframe.ico（16-256 全尺寸）+ tmp/icon_preview.png（预览）
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
S = 1024  # 超采样画布，最后缩到各尺寸


def rounded_mask(size, radius):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return m


def build_base() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))

    # ---- 背景：深板岩垂直渐变 + 顶部微弱青色辉光 ----
    bg = Image.new("RGBA", (S, S))
    top, bottom = (22, 32, 46), (10, 14, 20)
    px = bg.load()
    for y in range(S):
        t = y / S
        px[0, y] = px[1, y] = (
            round(top[0] + (bottom[0] - top[0]) * t),
            round(top[1] + (bottom[1] - top[1]) * t),
            round(top[2] + (bottom[2] - top[2]) * t),
            255,
        )
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([S * 0.18, S * 0.02, S * 0.82, S * 0.5], fill=(53, 208, 255, 46))
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    bg.alpha_composite(glow)
    img.paste(bg, (0, 0), rounded_mask(S, 200))

    d = ImageDraw.Draw(img)
    # 描边：极细高光让圆角底在浅色桌面上也有轮廓
    d.rounded_rectangle([2, 2, S - 3, S - 3], radius=198, outline=(255, 255, 255, 26), width=5)

    # ---- 16:9 视频帧 ----
    fx0, fy0, fx1, fy1 = S * 0.18, S * 0.28, S * 0.82, S * 0.72   # 16:9
    sharp = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sharp)
    sd.rounded_rectangle([fx0, fy0, fx1, fy1], radius=44, fill=(28, 42, 60, 255),
                         outline=(226, 244, 255, 255), width=22)

    # 帧内内容：右侧"锐利"= 明亮山峦 + 太阳（高对比细线）；随后左侧复制并糊掉
    pad = 44
    ix0, iy0, ix1, iy1 = fx0 + pad, fy0 + pad, fx1 - pad, fy1 - pad
    # 内景裁剪底
    sd.rectangle([ix0, iy0, ix1, iy1], fill=(12, 24, 38, 255))
    # 远山（两座）
    sd.polygon([(ix0, iy1), (ix0 + (ix1 - ix0) * 0.42, iy0 + (iy1 - iy0) * 0.30),
                (ix0 + (ix1 - ix0) * 0.62, iy1)], fill=(46, 84, 120, 255))
    sd.polygon([(ix0 + (ix1 - ix0) * 0.34, iy1), (ix0 + (ix1 - ix0) * 0.66, iy0 + (iy1 - iy0) * 0.16),
                (ix1, iy1)], fill=(74, 138, 190, 255))
    # 太阳
    sx, sy, sr = ix0 + (ix1 - ix0) * 0.74, iy0 + (iy1 - iy0) * 0.30, (iy1 - iy0) * 0.14
    sd.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(255, 209, 102, 255))

    blurred = sharp.filter(ImageFilter.GaussianBlur(16))

    # 对角分割：左下=糊，右上=锐（分割线从帧左上到右下 25%→75% 对角）
    import math
    ang = math.radians(-18)  # 分割线近乎水平、略微上斜
    mask = Image.new("L", (S, S), 0)
    md = ImageDraw.Draw(mask)
    cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
    L = S
    dx, dy = math.cos(ang) * L, math.sin(ang) * L
    # 线上方(右上)为锐区域
    md.polygon([(cx - dx, cy - dy), (cx + dx, cy + dy), (cx + dx, cy - S), (cx - dx, cy - S)], fill=255)

    frame = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    frame.paste(blurred, (0, 0), mask.point(lambda v: 255 - v))
    frame.paste(sharp, (0, 0), mask)
    img.alpha_composite(frame)

    # 分割线：亮青高光线（略超出帧边制造"切片"感），裁回帧圆角内
    line = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ld = ImageDraw.Draw(line)
    ld.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=(53, 208, 255, 255), width=14)
    line = line.filter(ImageFilter.GaussianBlur(2))
    frame_clip = Image.new("L", (S, S), 0)
    ImageDraw.Draw(frame_clip).rounded_rectangle([fx0 + 4, fy0 + 4, fx1 - 4, fy1 - 4], radius=40, fill=255)
    img.alpha_composite(Image.composite(line, Image.new("RGBA", (S, S), (0, 0, 0, 0)), frame_clip))

    # ---- 星芒：锐侧上部，白核青芒 ----
    kx, ky = fx1 - S * 0.115, fy0 + S * 0.075
    star = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    std_ = ImageDraw.Draw(star)
    R, r = S * 0.085, S * 0.014
    for w, scale, col in ((30, 1.0, (53, 208, 255, 160)), (14, 0.62, (255, 255, 255, 230))):
        pts = []
        for i in range(8):
            ang2 = math.pi / 4 * i - math.pi / 2
            rad = R * scale if i % 2 == 0 else r * scale
            pts.append((kx + math.cos(ang2) * rad, ky + math.sin(ang2) * rad))
        std_.polygon(pts, fill=col)
    img.alpha_composite(star)

    return img


def main():
    base = build_base()
    preview = base.resize((256, 256), Image.LANCZOS)
    preview.save(ROOT / "tmp" / "icon_preview.png")

    icon_path = ROOT / "launcher" / "crispframe.ico"
    sizes = [256, 128, 64, 48, 32, 24, 16]
    base.resize((256, 256), Image.LANCZOS).save(
        icon_path, format="ICO", sizes=[(s, s) for s in sizes],
        append_images=[base.resize((s, s), Image.LANCZOS) for s in sizes[1:]])
    print("saved:", icon_path)


if __name__ == "__main__":
    main()
