#!/usr/bin/env python3
"""Cascade inner VQ encoder — image -> compact CSC1 token payload.

Tiles any image, maps each tile through the selected VQ encode geometry, encodes the 4-level token
stream, and writes a CSC1 token payload. The ORIGINAL width/height and tile grid are stored in the
header so the decoder can reconstruct the blurry guide at the exact original size.

By default the tokens are entropy-coded with the learned hyperprior (CSC1 **v6**). v6 stores the
encode resolution, tile mode, and overlap margin, which supports native-aspect and overlap tiles.
Use --fixed-rate for the legacy bit-packed format (v3, black-letterbox only, no hyperprior
dependency).

Usage:
    python encode.py photo.jpg                                # -> photo.nit (hyperprior-coded, default)
    python encode.py photo.jpg -o out.nit --fixed-rate        # legacy bit-packed (v3)
"""

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "model"
sys.path.insert(0, str(HERE))
from vqvae import VQVAE4L, VQVAE4LConfig
from letterbox import letterbox, tile_grid, tile_boxes, TILE, S

MAGIC = b"CSC1"
VERSION = 3  # v3: header carries original (W, H) + tile grid (cols, rows); tokens for each tile
CROP = 192
LEVELS = ["l1", "l2", "l3", "l4"]
# Letterbox handles resize/aspect; this only tensorises + normalises to [-1, 1].
_NORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])


def load_model(ckpt_path, device=None):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = ckpt["model_config"]
    cfg = VQVAE4LConfig(**{k: v for k, v in mc.items()
                           if k in VQVAE4LConfig.__dataclass_fields__})
    model = VQVAE4L(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model, device, cfg


def pack_bits(values, bits):
    """Pack a 1-D array of non-negative ints (< 2**bits) into a byte string, MSB-first."""
    v = np.asarray(values, dtype=np.uint32)
    bitmat = ((v[:, None] >> np.arange(bits - 1, -1, -1)) & 1).astype(np.uint8)
    return np.packbits(bitmat.reshape(-1)).tobytes()


def encode(model, cfg, device, pil_image, tile_size=TILE, res=S):
    pil_image = pil_image.convert("RGB")
    W, H = pil_image.size
    cols, rows = tile_grid(W, H, tile_size)
    boxes = tile_boxes(W, H, cols, rows)
    bits = [max(1, math.ceil(math.log2(getattr(cfg, f"{lv}_codebook_size")))) for lv in LEVELS]

    header = bytearray(MAGIC)
    header.append(VERSION)
    header += struct.pack("IIBB", W, H, cols, rows)   # original size + tile grid
    specs_written = False
    bodies = bytearray()
    for box in boxes:                                  # each tile: crop → letterbox to res² → encode
        square, _ = letterbox(pil_image.crop(box), size=res)
        x = _NORM(square).unsqueeze(0).to(device)
        with torch.no_grad():
            idx = model.encode(x)
        for lv, b in zip(LEVELS, bits):
            arr = idx[lv][0].cpu().numpy().astype(np.uint32)
            if not specs_written:
                h, w = arr.shape
                header += struct.pack("BBB", h, w, b)  # token-map shape (identical for every tile)
            bodies += pack_bits(arr.reshape(-1), b)
        specs_written = True
    return bytes(header) + bytes(bodies), (cols, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input image")
    ap.add_argument("-o", "--output", default=None,
                    help="output .nit file (default: <input>.nit)")
    ap.add_argument("--ckpt", default=str(MODEL / "vqvae_v05.pt"))
    ap.add_argument("--hp", default=str(MODEL / "hyperprior.pt"),
                    help="hyperprior weights (for the default v6 entropy coder)")
    ap.add_argument("--fixed-rate", action="store_true",
                    help="legacy bit-packed format (v3); skips the hyperprior")
    ap.add_argument("--qp", type=int, default=S,
                    help=f"quality parameter = per-tile encode resolution, a multiple of 32 "
                         f"in 32..512 (default {S}). Lower = smaller file / more generated detail, higher = "
                         f"more faithful. Ladder e.g. 128/160/192/224/256. Needs the multi-resolution "
                         f"model for non-{S} values.")
    ap.add_argument("--tile-size", type=int, default=TILE,
                    help=f"max tile size; images larger split into a grid (default {TILE})")
    ap.add_argument("--mode", choices=["black", "neighbor", "native", "native_ov"], default="native_ov",
                    help="tile encoding mode: black=letterbox (black bars), neighbor=letterbox with "
                         "real neighbour fill, native=true aspect ratio (no bars), "
                         "native_ov=native with overlap (default: native_ov)")
    ap.add_argument("--margin", type=int, default=32,
                    help="overlap margin in pixels for native_ov mode; 0..255 (default 32)")
    args = ap.parse_args()
    if args.qp % 32 != 0 or not 32 <= args.qp <= 512:
        ap.error("--qp (encode resolution) must be a multiple of 32 in the range 32..512")
    if not 0 <= args.margin <= 255:
        ap.error("--margin must be in the range 0..255")
    out_path = args.output or str(Path(args.input).with_suffix(".nit"))

    src = Image.open(args.input)
    W, H = src.size

    if args.fixed_rate:
        # Fixed-rate (v3) only supports the original black-letterbox path
        model, device, cfg = load_model(Path(args.ckpt))
        data, (cols, rows) = encode(model, cfg, device, src, args.tile_size, res=args.qp)
        fmt = "v3 fixed-rate"
    else:
        from hp_codec import load_models, encode_hp
        m, hp, _, device = load_models(Path(args.ckpt), Path(args.hp))  # CPU (deterministic coding)
        data, (cols, rows) = encode_hp(m, hp, device, src, tile_size=args.tile_size, res=args.qp,
                                       mode=args.mode, margin=args.margin)
        fmt = f"v6 hyperprior mode={args.mode}"

    Path(out_path).write_bytes(data)
    n = cols * rows
    grid = f"{cols}x{rows} tiles" if n > 1 else "single tile (no split)"
    print(f"Encoded {args.input} ({W}x{H}) -> {out_path}  ({len(data)/1024:.2f} KB, "
          f"~{len(data)/1024/n:.2f} KB/tile, qp={args.qp}, {grid}, {fmt}; size+grid+res in header)")


if __name__ == "__main__":
    main()
