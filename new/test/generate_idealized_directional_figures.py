from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("/root/data2/fyc/fyc/GRAT-main/new/test_runs/idealized_directional_figures")


IDEAL_RESULTS = {
    "Synapse (CT) / SAMed-vit_b": {
        "mDSC": [0.68, 0.49, 0.17],
        "mHD95": [24.0, 58.0, 132.0],
    },
    "ISIC (Dermoscopy) / SAMed-vit_b": {
        "mDSC": [0.74, 0.55, 0.21],
        "mHD95": [18.0, 41.0, 96.0],
    },
    "REFUGE (Fundus) / SAMed-vit_b": {
        "mDSC": [0.71, 0.51, 0.19],
        "mHD95": [14.0, 33.0, 79.0],
    },
}


LABELS = ["Tangential", "Isotropic", "Radial (FDRP)"]
COLORS = ["#5B8FF9", "#F6BD16", "#E8684A"]
BG = "#F7F5EF"
TEXT = "#231F1C"
GRID = "#D7D1C7"


def load_font(size: int, bold: bool = False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font(30, bold=True)
SUB_FONT = load_font(18)
AXIS_FONT = load_font(20, bold=True)
LABEL_FONT = load_font(18, bold=True)
VALUE_FONT = load_font(18, bold=True)
NOTE_FONT = load_font(16)


def draw_bar_panel(draw: ImageDraw.ImageDraw, x0: int, y0: int, w: int, h: int, title: str, labels, values, ymax: float, lower_is_better: bool) -> None:
    draw.text((x0, y0 - 38), title, fill=TEXT, font=AXIS_FONT)
    left = x0 + 20
    right = x0 + w - 10
    top = y0 + 10
    bottom = y0 + h - 55

    for frac in [0.25, 0.5, 0.75, 1.0]:
        yy = bottom - (bottom - top) * frac
        draw.line((left, yy, right, yy), fill=GRID, width=1)

    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)

    slot = (right - left) / len(labels)
    bar_w = int(slot * 0.55)
    for i, (label, value, color) in enumerate(zip(labels, values, COLORS)):
        cx = left + slot * (i + 0.5)
        bh = 0 if ymax == 0 else (bottom - top) * (value / ymax)
        bx0 = int(cx - bar_w / 2)
        by0 = int(bottom - bh)
        bx1 = int(cx + bar_w / 2)
        by1 = bottom
        draw.rectangle((bx0, by0, bx1, by1), fill=color, outline=TEXT, width=2)

        txt = f"{value:.2f}" if ymax <= 2 else f"{value:.0f}"
        tw = draw.textbbox((0, 0), txt, font=VALUE_FONT)[2]
        draw.text((cx - tw / 2, by0 - 26), txt, fill=TEXT, font=VALUE_FONT)

        label_lines = label.replace(" (", "\n(").split("\n")
        for j, line in enumerate(label_lines):
            lw = draw.textbbox((0, 0), line, font=LABEL_FONT)[2]
            draw.text((cx - lw / 2, bottom + 10 + j * 20), line, fill=TEXT, font=LABEL_FONT)

    note = "Lower is better for attack" if lower_is_better else "Higher is better for attack"
    note_w = draw.textbbox((0, 0), note, font=NOTE_FONT)[2]
    draw.text((x0 + w / 2 - note_w / 2, y0 + h - 20), note, fill="#8C2F1C", font=NOTE_FONT)


def plot_single(title: str, values: dict[str, list[float]]) -> None:
    img = Image.new("RGB", (1500, 700), BG)
    draw = ImageDraw.Draw(img)

    draw.text((50, 28), title, fill=TEXT, font=TITLE_FONT)
    sub = "Strictly aligned complex-frequency RMS budget, same source model, same victim model"
    draw.text((50, 78), sub, fill=TEXT, font=SUB_FONT)

    draw_bar_panel(draw, 60, 170, 650, 450, "Ideal mDSC Trend", LABELS, values["mDSC"], 0.85, lower_is_better=True)
    draw_bar_panel(draw, 790, 170, 650, 450, "Ideal mHD95 Trend", LABELS, values["mHD95"], max(values["mHD95"]) * 1.15, lower_is_better=False)

    footer = "Illustrative idealized result under the hypothesis that radial-only FDRP uniquely captures the transferable vulnerability."
    draw.text((50, 650), footer, fill=TEXT, font=NOTE_FONT)

    name = title.split(" / ")[0].lower().replace(" ", "_").replace("(", "").replace(")", "")
    img.save(OUT_DIR / f"{name}_ideal_directional_result.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for title, values in IDEAL_RESULTS.items():
        plot_single(title, values)
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
