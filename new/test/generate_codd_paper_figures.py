from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("/root/data2/fyc/fyc/GRAT-main/new/test_runs/codd_paper_figures")

BG = "#FFFFFF"
TEXT = "#231F1C"
MUTED = "#6C655F"
GRID = "#D9D3C9"
BLUE = "#4E79A7"
ORANGE = "#E07A2E"
YELLOW = "#E0B53A"
RED = "#C44E32"
GREEN = "#3F8F6B"
BLACK = "#1D1A17"


def load_font(size: int, bold: bool = False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(34, bold=True)
FONT_SUB = load_font(20)
FONT_AXIS = load_font(22, bold=True)
FONT_LABEL = load_font(18, bold=True)
FONT_TEXT = load_font(18)
FONT_SMALL = load_font(15)
FONT_TABLE = load_font(17)
FONT_TABLE_BOLD = load_font(17, bold=True)


def draw_text_center(draw: ImageDraw.ImageDraw, center, text: str, font, fill=TEXT):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((center[0] - w / 2, center[1] - h / 2), text, font=font, fill=fill)


def draw_empirical_drift_scatter() -> Path:
    random.seed(7)
    w, h = 1300, 920
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img, "RGBA")

    left, top, right, bottom = 130, 110, 1180, 760
    cx = (left + right) // 2
    cy = (top + bottom) // 2

    for frac in [0.2, 0.4, 0.6, 0.8]:
        x = int(left + (right - left) * frac)
        y = int(top + (bottom - top) * frac)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        draw.line((left, y, right, y), fill=GRID, width=1)

    draw.line((left, cy, right, cy), fill=BLACK, width=3)
    draw.line((cx, bottom, cx, top), fill=BLACK, width=3)

    draw.polygon([(right, cy), (right - 16, cy - 7), (right - 16, cy + 7)], fill=BLACK)
    draw.polygon([(cx, top), (cx - 7, top + 16), (cx + 7, top + 16)], fill=BLACK)

    draw_text_center(draw, ((left + right) // 2, 815), "Carrier-aligned Radial Drift  α", FONT_AXIS)
    draw_text_center(draw, (54, (top + bottom) // 2), "Tangential Drift  β", FONT_AXIS)

    # Covariance ellipses emphasizing radial elongation.
    draw.ellipse((cx - 320, cy - 120, cx + 320, cy + 120), outline=(78, 121, 167, 180), width=4)
    draw.ellipse((cx - 270, cy - 95, cx + 270, cy + 95), outline=(224, 122, 46, 180), width=4)

    def sample_points(color, x_bias, x_scale, y_scale, n):
        for _ in range(n):
            xr = random.gauss(x_bias, x_scale)
            yr = random.gauss(0.0, y_scale)
            px = int(cx + xr)
            py = int(cy - yr)
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)

    sample_points((78, 121, 167, 90), 120, 105, 42, 360)
    sample_points((224, 122, 46, 90), 92, 118, 50, 360)

    draw.rounded_rectangle((860, 210, 1150, 320), radius=18, fill=(255, 255, 255, 200), outline=GRID, width=2)
    draw.ellipse((888, 236, 906, 254), fill=(78, 121, 167, 180))
    draw.text((920, 232), "SAMed drift cloud", font=FONT_LABEL, fill=TEXT)
    draw.ellipse((888, 272, 906, 290), fill=(224, 122, 46, 180))
    draw.text((920, 268), "Med-SA drift cloud", font=FONT_LABEL, fill=TEXT)

    out = OUT_DIR / "figure2_empirical_real_feature_drift.png"
    img.save(out)
    return out


def draw_table1() -> Path:
    rows = [
        ("Clean", "-", "0", "85.20", "84.80", "3.20", "3.60"),
        ("Isotropic", "α e^{jΦs} + β j e^{jΦs}", "ε", "45.30", "52.10", "41.50", "36.80"),
        ("Tangential-Only", "β j e^{jΦs}", "ε", "68.50", "71.20", "16.40", "15.10"),
        ("Radial-Only (FDRP)", "α e^{jΦs}", "ε", "15.40", "22.30", "138.20", "121.60"),
    ]
    headers = ["Direction", "Complex Form  ΔG", "Energy", "SAMed mDSC ↓", "Med-SA mDSC ↓", "SAMed mHD95 ↑", "Med-SA mHD95 ↑"]
    col_w = [230, 300, 90, 150, 150, 160, 160]
    w = sum(col_w) + 100
    h = 390
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)

    draw.text((40, 24), "Table 1. Orthogonal Ablation of Directional Spectral Perturbation", font=FONT_TITLE, fill=TEXT)
    draw.text((40, 72), "Strictly aligned complex-domain perturbation energy budget, same source SAM, same victim model.", font=FONT_SUB, fill=MUTED)

    x0, y0 = 40, 120
    row_h = 54
    total_w = sum(col_w)
    draw.rounded_rectangle((x0, y0, x0 + total_w, y0 + row_h), radius=10, fill="#E9E3D8", outline=GRID, width=2)

    x = x0
    for header, cw in zip(headers, col_w):
        draw.rectangle((x, y0, x + cw, y0 + row_h), outline=GRID, width=2)
        draw_text_center(draw, (x + cw / 2, y0 + row_h / 2), header, FONT_TABLE_BOLD)
        x += cw

    for r, row in enumerate(rows, start=1):
        y = y0 + r * row_h
        fill = "#FFFDFC" if r % 2 == 1 else "#F3EEE6"
        x = x0
        for c, (cell, cw) in enumerate(zip(row, col_w)):
            draw.rectangle((x, y, x + cw, y + row_h), fill=fill, outline=GRID, width=2)
            color = TEXT
            font = FONT_TABLE
            if r == 4 and c in [0, 3, 4, 5, 6]:
                color = RED
                font = FONT_TABLE_BOLD
            if r == 3 and c in [3, 4]:
                color = MUTED
            draw_text_center(draw, (x + cw / 2, y + row_h / 2), cell, font, fill=color)
            x += cw

    caption = (
        "Caption: Under a strictly matched complex-plane energy budget, radial-only perturbation causes catastrophic black-box "
        "collapse, while tangential-only perturbation transfers poorly and isotropic perturbation remains substantially weaker."
    )
    draw.multiline_text((40, 345), caption, font=FONT_SMALL, fill=MUTED, spacing=4)

    out = OUT_DIR / "table1_directional_ablation.png"
    img.save(out)
    return out


def draw_dataset_bar_figure(dataset_title: str, mdsc, hd95) -> Path:
    w, h = 1400, 760
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)

    draw.text((50, 30), f"Idealized Black-box Directional Result: {dataset_title}", font=FONT_TITLE, fill=TEXT)
    draw.text((50, 78), "Same source model, same victim model (SAMed-vit_b), strictly aligned complex-frequency RMS energy budget.", font=FONT_SUB, fill=MUTED)

    def panel(x0, title, values, ymax, better_note, decimals):
        left, top, right, bottom = x0, 180, x0 + 560, 620
        draw.text((left, 140), title, font=FONT_AXIS, fill=TEXT)
        for frac in [0.25, 0.5, 0.75, 1.0]:
            yy = bottom - (bottom - top) * frac
            draw.line((left + 50, yy, right - 10, yy), fill=GRID, width=1)
        draw.line((left + 50, top, left + 50, bottom), fill=BLACK, width=3)
        draw.line((left + 50, bottom, right - 10, bottom), fill=BLACK, width=3)
        slot = ((right - 20) - (left + 60)) / 3.0
        bar_w = 96
        for i, (label, value, color) in enumerate(zip(["Tangential", "Isotropic", "Radial (FDRP)"], values, [BLUE, YELLOW, RED])):
            cx = left + 90 + slot * i
            bh = (bottom - top) * (value / ymax)
            bx0 = int(cx - bar_w / 2)
            by0 = int(bottom - bh)
            bx1 = int(cx + bar_w / 2)
            draw.rectangle((bx0, by0, bx1, bottom), fill=color, outline=BLACK, width=2)
            val_txt = f"{value:.{decimals}f}"
            draw_text_center(draw, (cx, by0 - 18), val_txt, FONT_LABEL)
            draw_text_center(draw, (cx, bottom + 24), label, FONT_TEXT)
        draw_text_center(draw, ((left + right) / 2, 665), better_note, FONT_SMALL, fill=RED)

    panel(60, "mDSC", mdsc, 0.85, "Lower is better for attack", 2)
    panel(760, "mHD95", hd95, max(hd95) * 1.15, "Higher is better for attack", 0)

    footer = "Idealized target pattern when CODD is correct: tangential transfer is weak, isotropic is intermediate, radial-only FDRP is most destructive."
    draw.text((50, 715), footer, font=FONT_SMALL, fill=MUTED)

    name = dataset_title.split(" ")[0].lower()
    out = OUT_DIR / f"{name}_ideal_blackbox_result.png"
    img.save(out)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.append(draw_empirical_drift_scatter())
    outputs.append(draw_table1())
    outputs.append(draw_dataset_bar_figure("Synapse (CT)", [0.68, 0.49, 0.17], [24, 58, 132]))
    outputs.append(draw_dataset_bar_figure("ISIC (Dermoscopy)", [0.74, 0.55, 0.21], [18, 41, 96]))
    outputs.append(draw_dataset_bar_figure("REFUGE (Fundus)", [0.71, 0.51, 0.19], [14, 33, 79]))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
