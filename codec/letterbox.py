"""Tile and aspect-preserving square geometry, shared by encode and decode.

Legacy fixed-rate and square hyperprior payloads letterbox each tile into a square. The v6
hyperprior path can also encode native-aspect or overlapped native-aspect tiles, but it still uses
the tile grid and final whole-image letterbox helpers here for wrapper-level Gemini restoration.
"""

import math

from PIL import Image

S = 192          # VQ-VAE working resolution (square)
TILE = 512       # target max tile size; images larger than this are split into a grid


def tile_grid(W, H, tile=TILE):
    """Number of (cols, rows) tiles: each image dimension is split into ceil(dim/tile) parts,
    so each tile is ≤ `tile` px and roughly square. 1024×512→(2,1), 1024×1024→(2,2),
    760×512→(2,1), 600×315→(2,1), ≤512²→(1,1)."""
    return max(1, math.ceil(W / tile)), max(1, math.ceil(H / tile))


def tile_boxes(W, H, cols, rows):
    """Pixel boxes (left, top, right, bottom) partitioning W×H into cols×rows tiles (row-major)."""
    xs = [round(i * W / cols) for i in range(cols + 1)]
    ys = [round(j * H / rows) for j in range(rows + 1)]
    return [(xs[c], ys[r], xs[c + 1], ys[r + 1]) for r in range(rows) for c in range(cols)]


def lb_dims(W, H, size=S):
    """Content dimensions inside the size×size square (one axis == size, the other ≤ size)."""
    scale = size / max(W, H)
    nw = min(size, max(1, round(W * scale)))
    nh = min(size, max(1, round(H * scale)))
    return nw, nh


def letterbox(pil, size=S):
    """Scale `pil` to fit a size×size square, centre it, pad the rest black.
    Returns (square_image, (orig_W, orig_H))."""
    pil = pil.convert("RGB")
    W, H = pil.size
    nw, nh = lb_dims(W, H, size)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(pil.resize((nw, nh), Image.LANCZOS), ((size - nw) // 2, (size - nh) // 2))
    return canvas, (W, H)


def content_box(W, H, G, size=S):
    """The content region (bars excluded) inside a G×G square, scaled from the size×size layout."""
    nw, nh = lb_dims(W, H, size)
    cw = round(G * nw / size)
    ch = round(G * nh / size)
    left = (G - cw) // 2
    top = (G - ch) // 2
    return (left, top, left + cw, top + ch)


def unletterbox(square_img, W, H):
    """Crop the letterbox bars from a square image and resample to the exact original (W, H)."""
    G = square_img.size[0]
    return square_img.crop(content_box(W, H, G)).resize((W, H), Image.LANCZOS)


def pick_tier(W, H):
    """Nearest Gemini output tier (1K/2K) to the original's long side. Returns (label, px)."""
    m = max(W, H)
    return ("2K", 2048) if abs(m - 2048) <= abs(m - 1024) else ("1K", 1024)
