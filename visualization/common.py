"""Shared image preparation and rendering for the optional CLIP visualizations."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw
from preprocess_image import PreprocessImage
from runtime import ROOT


class DisplayPreprocess(PreprocessImage):
    def to_tensor(self, image):
        self.crop = image.copy()
        return super().to_tensor(image)


def prepare(path):
    preprocess = DisplayPreprocess()
    tensor = preprocess(path)
    with Image.open(path) as image:
        original = image.convert("RGB")
    return original, preprocess.crop, tensor


def parser(description):
    cli = argparse.ArgumentParser(description=description)
    cli.add_argument("image", type=Path)
    return cli


def panel(items, footer=""):
    canvas = Image.new("RGB", (240 * len(items), 280), "#f3f5f8")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(items):
        preview = image.copy()
        preview.thumbnail((224, 224), Image.Resampling.LANCZOS)
        left = 240 * index + (240 - preview.width) // 2
        canvas.paste(preview, (left, 34 + (224 - preview.height) // 2))
        draw.text((240 * index + 8, 10), label, fill="#142238")
    draw.text((8, 262), footer, fill="#142238")
    return canvas


def save(image, source, kind):
    output = ROOT / "outputs/visualizations" / f"{source.stem}_{kind}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output)
