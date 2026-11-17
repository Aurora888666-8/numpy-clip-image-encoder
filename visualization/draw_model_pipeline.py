"""Render the report overview using Pillow (no model execution required)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    scale = 2
    image = Image.new("RGB", (1000 * scale, 1040 * scale), "white")
    draw = ImageDraw.Draw(image)
    navy, blue, red = "#142d4e", "#275d97", "#a62938"

    def text(x, y, value, size=21, bold=False, color=navy):
        font = ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size * scale)
        assert x * scale + draw.textlength(value, font=font) < image.width
        draw.text((x * scale, y * scale), value, fill=color, font=font, anchor="lt")

    def box(x, y, w, h, fill="#eef5fc", edge="#a7bed9"):
        draw.rounded_rectangle(tuple(scale * v for v in (x, y, x+w, y+h)),
                               radius=20, fill=fill, outline=edge, width=3)

    def arrow(x, y, end, right=False):
        def point(t, offset=0):
            return ((t if right else x+offset)*scale, (y+offset if right else t)*scale)
        start = x if right else y
        for a in range(start, end-9, 10 if right else end-start):
            draw.line([point(a), point(min(a+(6 if right else end-start), end-8))],
                      fill=blue, width=4)
        draw.polygon([point(end), point(end-9, -5), point(end-9, 5)], fill=blue)

    def node(y, h, title, lines, todo=None):
        box(64, y, 478, h, edge=red if todo else "#a7bed9")
        text(82, y+13, title, 23, True)
        if todo:
            text(424, y+16, todo, 17, True, red)
        for i, line in enumerate(lines):
            text(82, y+43+i*24, line, 19)

    def side(y, h, title, lines):
        box(600, y, 356, h, "#f6f8fb")
        text(618, y+15, title, 22, True)
        for i, line in enumerate(lines):
            text(618, y+48+i*24, line, 18)

    text(64, 32, "NumPy CLIP ViT-B/32", 37, True)
    text(64, 86, "Core Image Encoder Pipeline", 24, True, blue)
    text(600, 85, "Added Visualization / Analysis", 23, True, blue)
    text(600, 114, "Dashed arrows denote analysis branches.", 17)
    node(127, 72, "Input Image", ["Preprocessed tensor  [B, 3, 224, 224]"])
    node(222, 72, "Patchify", ["7×7 patches, 32×32 pixels  [B, 49, 3072]"], "TODO-01")
    node(317, 96, "Patch Embedding", ["+ CLS Token + Positional Embedding", "[B, 50, 768]"], "TODO-05")
    node(436, 72, "Pre-LayerNorm", ["[B, 50, 768]"])
    node(531, 190, "12× Transformer Blocks", ["Each block preserves [B, 50, 768]"])
    for y, todo, label in [(607, "TODO-02", "Split Heads (inside attention)"),
                           (642, "TODO-03", "Multi-Head Attention"),
                           (677, "TODO-04", "MLP + Residual")]:
        box(81, y, 444, 30, "#ffffff")
        text(91, y+6, todo, 16, True, red)
        text(191, y+6, label, 17)
    node(744, 120, "Output Stage", ["Select CLS Token → Post-LayerNorm",
                                    "[B, 768] → Projection → [B, 512]",
                                    "L2 Normalize"], "TODO-06")
    node(887, 72, "Output Feature", ["[B, 512]"])
    for a, b in [(199,222),(294,317),(413,436),(508,531),(721,744),(864,887)]:
        arrow(303, a, b)

    side(222, 149, "A  Patch Visualization", [
        "Verify against patchify output", "Restore 7×7 patch grid",
        "Number patches 0–48", "Display a selected patch"])
    arrow(542, 258, 600, right=True)
    side(531, 195, "B  Attention Visualization", [
        "Compute Block 12 attention weights", "CLS → 49 patches",
        "Mean over 12 heads → 7×7 grid", "Overlay on the original image",
        "Shows attention distribution only"])
    arrow(542, 656, 600, right=True)
    side(744, 165, "C  Numerical Validation", [
        "check.py --stage all", "assets PASS",
        "preprocess / patches / tokens / block0",
        "last_tokens / output / batch: PASS"])
    text(64, 990, "Solid arrows: encoder forward pass    •    Red labels: completed TODOs", 18)
    out = Path(__file__).resolve().parents[1] / "outputs/report_figures/model_pipeline_with_extensions.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, dpi=(300, 300))
    print(out)


if __name__ == "__main__":
    main()
