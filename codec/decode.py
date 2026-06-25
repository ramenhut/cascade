#!/usr/bin/env python3
"""Cascade decoder — .nit token file -> reconstructed image.

Two-stage decode:
  1. VQ-VAE codebook lookup -> structurally-correct but blurry image (instant, no network).
  2. Gemini ('gemini-3-pro-image-preview') restores texture/sharpness from the blurry decode,
     keeping composition/colour faithful. This is where the perceptual quality comes from.

Output is restored to the ORIGINAL size recorded at encode time: Gemini renders a square at the
nearest tier (1K/2K, or --resolution), then the letterbox bars are cropped and the image is
resampled to the exact original W×H. Use --tokens-only for stage 1 alone (blurry VQ). Gemini
decode needs GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.

Usage:
    python decode.py photo.nit -o photo_out.png                    # full (VQ + Gemini), original size
    python decode.py photo.nit -o photo_out.png --resolution 2K    # force render tier
    python decode.py photo.nit -o photo_out.png --preserve-style   # honor source medium (anime/art/…)
    python decode.py photo.nit -o photo_tokens.png --tokens-only   # blurry VQ only
"""

import argparse
import io
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "model"
sys.path.insert(0, str(HERE))
from vqvae import VQVAE4L, VQVAE4LConfig
from letterbox import letterbox, unletterbox, pick_tier, tile_boxes

TIER_PX = {"1K": 1024, "2K": 2048, "4K": 4096}

MAGIC = b"CSC1"
LEVELS = ["l1", "l2", "l3", "l4"]
CROP = 192

ENHANCE_PROMPT = (
    "This is a low-resolution compressed photograph that has lost fine detail. Reconstruct it as "
    "a crisp, high-resolution photograph, faithfully upscaling it: keep the exact same composition, "
    "colors, and objects, and synthesize realistic fine detail and texture. Do not add or remove "
    "objects. If the image has solid black letterbox bars (padding), keep those areas solid black."
)

# Used with --preserve-style: detect the source medium/style and reconstruct in THAT style,
# instead of always rendering a photograph (e.g. keeps flat 2D anime/illustration as flat art).
STYLE_PROMPT = (
    "This is a low-resolution, blurry compressed image. FIRST identify the visual medium/style of "
    "the original — e.g. photograph, flat 2D anime/cartoon, line art, watercolor, oil painting, 3D "
    "render — and reconstruct it crisply IN THAT SAME STYLE. Do NOT convert it into a photograph. "
    "Keep the exact same composition, colors, characters, and objects, and restore detail/texture "
    "appropriate to that style. If it is flat 2D anime/cartoon, preserve flat cel-shading, clean bold "
    "outlines, and solid color fills — do not add realistic 3D shading, skin pores, or photographic "
    "texture. If solid black letterbox bars (padding) are present, keep those areas solid black."
)


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
    return model, device


def unpack_bits(buf, n, bits):
    allbits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8))[:n * bits].reshape(n, bits)
    weights = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint32)
    return (allbits.astype(np.uint32) * weights).sum(axis=1)


def read_dfc(path, device):
    data = Path(path).read_bytes()
    assert data[:4] == MAGIC, "not a CSC1 (Cascade) file"
    version = data[4]
    off = 5
    if version >= 3:
        W, H, cols, rows = struct.unpack("IIBB", data[off:off + 10]); off += 10
    elif version == 2:
        W, H = struct.unpack("II", data[off:off + 8]); off += 8; cols = rows = 1
    else:
        W = H = CROP; cols = rows = 1  # v1: no stored size — treat as square single tile
    specs = []
    for _ in LEVELS:
        h, w, bits = struct.unpack("BBB", data[off:off + 3]); off += 3
        specs.append((h, w, bits))
    body, pos = data[off:], 0
    tiles = []
    for _ in range(cols * rows):
        idx = {}
        for lv, (h, w, bits) in zip(LEVELS, specs):
            n = h * w
            nbytes = math.ceil(n * bits / 8)
            vals = unpack_bits(body[pos:pos + nbytes], n, bits); pos += nbytes
            idx[lv] = torch.from_numpy(vals.astype(np.int64)).reshape(1, h, w).to(device)
        tiles.append(idx)
    return tiles, (W, H), (cols, rows)


def to_pil(t):
    img = t[0].clamp(-1, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8))


def gemini_enhance(image, image_size="1K", prompt=ENHANCE_PROMPT, model="gemini-3-pro-image-preview"):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY), or use --tokens-only.")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    buf = io.BytesIO(); image.save(buf, format="PNG"); buf.seek(0)
    cfg_kwargs = dict(response_modalities=["IMAGE", "TEXT"])
    try:
        cfg_kwargs["image_config"] = types.ImageConfig(image_size=image_size, aspect_ratio="1:1")
    except Exception:
        pass  # older SDKs: fall back to default (~1K square)
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
                  types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(**cfg_kwargs))
    cand = resp.candidates[0] if resp.candidates else None
    if cand and cand.content and cand.content.parts:
        for part in cand.content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
    fr = getattr(cand, "finish_reason", None)
    raise RuntimeError(
        f"Gemini returned no image (finish_reason={fr}). "
        "IMAGE_RECITATION = the source closely matches known/stock imagery and the model "
        "declined to reproduce it (a Gemini content policy, not a pipeline error).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input .nit file")
    ap.add_argument("-o", "--output", required=True, help="output image")
    ap.add_argument("--ckpt", default=str(MODEL / "vqvae_v05.pt"))
    ap.add_argument("--hp", default=str(MODEL / "hyperprior.pt"),
                    help="hyperprior weights (needed to decode v4 hyperprior-coded files)")
    ap.add_argument("--tokens-only", action="store_true",
                    help="skip Gemini; output the blurry VQ-only reconstruction")
    ap.add_argument("--resolution", choices=["auto", "1K", "2K", "4K"], default="auto",
                    help="Gemini render tier (auto = nearest to the original's long side)")
    ap.add_argument("--preserve-style", action="store_true",
                    help="detect the source's medium/style (anime, illustration, painting, …) and "
                         "reconstruct in it, instead of always rendering a photograph")
    args = ap.parse_args()

    # Dispatch on the format version: v4/v5/v6 = hyperprior-entropy-coded (default of encode.py),
    # v1/v2/v3 = fixed-rate bit-packed. v4+ tokens must be decoded on CPU (deterministic).
    data = Path(args.input).read_bytes()
    assert data[:4] == MAGIC, "not a CSC1 (Cascade) file"
    if data[4] >= 4:
        from hp_codec import load_models, decode_hp_tokens
        model, hp, _, device = load_models(Path(args.ckpt), Path(args.hp))
        tiles, (W, H), (cols, rows), mode, margin = decode_hp_tokens(model, hp, data, device)
    else:
        model, device = load_model(Path(args.ckpt))
        tiles, (W, H), (cols, rows) = read_dfc(args.input, device)
        mode, margin = "black", 0

    # Decode each tile and assemble a full-resolution blurry mosaic.
    # v6 native-aspect tiles use mode-aware unmap; v4/v5/v1-v3 always unletterbox.
    boxes = tile_boxes(W, H, cols, rows)
    mosaic = Image.new("RGB", (W, H))
    for idx, box in zip(tiles, boxes):
        with torch.no_grad():
            blurry = to_pil(model.decode_at_level(idx, max_level=3))
        l, t, r, b = box; tw, th = r - l, b - t
        if mode == "native":
            rec = blurry.resize((tw, th), Image.LANCZOS)
        elif mode == "native_ov":
            el = max(0, l - margin); et = max(0, t - margin)
            er = min(W, r + margin); eb = min(H, b + margin)
            etw, eth = er - el, eb - et
            rec = blurry.resize((etw, eth), Image.LANCZOS).crop((l - el, t - et, l - el + tw, t - et + th))
        else:
            rec = unletterbox(blurry, tw, th)
        mosaic.paste(rec, (l, t))

    grid = f"{cols}x{rows} tiles" if cols * rows > 1 else "single tile"
    if args.tokens_only:
        mosaic.save(args.output)
        print(f"Decoded (tokens only, blurry, {grid}) -> {args.output}  ({W}x{H})")
    else:
        tier = pick_tier(W, H)[0] if args.resolution == "auto" else args.resolution
        # ONE Gemini pass over the whole mosaic (avoids per-tile seams): square-pad, render, un-letterbox
        G = TIER_PX[tier]
        sq, _ = letterbox(mosaic, size=max(W, H))            # full-res square (black bars on short axis)
        sq = sq.resize((G, G), Image.LANCZOS)                # bound upload to the render tier
        prompt = STYLE_PROMPT if args.preserve_style else ENHANCE_PROMPT
        out = gemini_enhance(sq, image_size=tier, prompt=prompt)   # square G×G
        final = unletterbox(out, W, H)                       # crop bars + resample to original
        final.save(args.output)
        style = " preserve-style" if args.preserve_style else ""
        print(f"Decoded (VQ {grid} + 1 Gemini pass @ {tier}{style}, {out.size[0]}² → {W}x{H}) -> {args.output}")


if __name__ == "__main__":
    main()
