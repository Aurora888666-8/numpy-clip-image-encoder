"""Overlay final-block CLS-to-patch attention on the original image."""
import numpy as np
from PIL import Image, ImageDraw
from common import panel, parser, prepare, save
from helpers import attention_weights
from image_encoder import ImageEncoder
from runtime import load_weights


def last_block_attention(encoder, images):
    """Read-only inspection; returns [B,H,query,key] after softmax."""
    x = encoder.pre_norm(encoder.embed(images))
    for block in encoder.blocks[:-1]:
        x = block(x)
    block = encoder.blocks[-1]
    attn = block.attention
    qkv = block.norm1(x) @ attn.qkv_weight.T + attn.qkv_bias
    q = attn.split_heads(qkv[:, :, :attn.width])
    k = attn.split_heads(qkv[:, :, attn.width:2 * attn.width])
    scores = (q @ k.swapaxes(-2, -1)) / np.sqrt(np.float32(attn.head_dim))
    return attention_weights(scores)


def overlay(original, heat):
    # Map the preprocessing crop back to the source using the same resize and rounding.
    w, h = original.size
    rw, rh = (224, int(224 * h / w)) if w <= h else (int(224 * w / h), 224)
    left, top = round((rw - 224) / 2), round((rh - 224) / 2)
    x0, x1 = round(left * w / rw), round((left + 224) * w / rw)
    y0, y1 = round(top * h / rh), round((top + 224) * h / rh)
    intensity = heat / heat.max()
    rgb = np.stack([255 * intensity, 110 * intensity, 255 * (1 - intensity)], axis=-1)
    heat_image = Image.fromarray(np.rint(rgb).astype(np.uint8))
    mapped = heat_image.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
    result = original.copy()
    box = (x0, y0, x1, y1)
    result.paste(Image.blend(original.crop(box), mapped, 0.55), (x0, y0))
    ImageDraw.Draw(result).rectangle((x0, y0, x1 - 1, y1 - 1), outline="white")
    return result


def main():
    args = parser(__doc__).parse_args()
    original, _, tensor = prepare(args.image)
    encoder = ImageEncoder(load_weights())
    weights = last_block_attention(encoder, tensor)
    assert weights.shape == (1, 12, 50, 50)
    assert np.isfinite(weights).all() and (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=-1), 1, atol=1e-6)
    heat = weights[0, :, 0, 1:].mean(axis=0).reshape(7, 7)
    print(f"Verified attention shape {weights.shape} and normalized rows; "
          f"CLS-to-patch grid {heat.shape}, total mass {heat.sum():.6f}.")
    save(panel([("Original image", original),
                ("Block 12 | Heads 1-12 mean | CLS", overlay(original, heat))],
               "Blue: low. Orange: high. Attention distribution only."),
         args.image, "attention")


if __name__ == "__main__":
    main()
