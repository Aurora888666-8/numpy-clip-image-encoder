"""Show the original image, numbered model patches, and an optional selected patch."""
import numpy as np
from PIL import Image, ImageDraw
from common import panel, parser, prepare, save
from image_encoder import ImageEncoder


def main():
    cli = parser(__doc__)
    cli.add_argument("--patch", type=int, choices=range(49), metavar="0..48")
    args = cli.parse_args()
    original, crop, tensor = prepare(args.image)
    # patchify needs no weights or instance state.
    actual = ImageEncoder.patchify(None, tensor)[0]
    grid = crop.copy()
    draw = ImageDraw.Draw(grid)
    for index in range(49):
        row, col = divmod(index, 7)
        x, y = col * 32, row * 32
        np.testing.assert_array_equal(
            actual[index], tensor[0, :, y:y + 32, x:x + 32].ravel()
        )
        draw.rectangle((x, y, x + 31, y + 31), outline="white")
        draw.rectangle((x + 2, y + 2, x + 17, y + 14), fill="#142238")
        draw.text((x + 3, y + 2), str(index), fill="white")
    print("Verified exact patchify agreement for all 49 patches.")
    save(panel([("Original image", original), ("Model crop | 7 x 7 patches", grid)]),
         args.image, "patches")
    if args.patch is not None:
        row, col = divmod(args.patch, 7)
        selected = crop.crop((col * 32, row * 32, (col + 1) * 32, (row + 1) * 32))
        zoom = selected.resize((224, 224), Image.Resampling.NEAREST)
        save(panel([(f"Patch {args.patch} | 32 x 32 pixels", zoom)]),
             args.image, f"patch_{args.patch:02d}")


if __name__ == "__main__":
    main()
