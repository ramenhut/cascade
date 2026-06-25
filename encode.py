#!/usr/bin/env python3
"""Cascade encoder — content-adaptive guide selection (VQ-VAE vs modern codec) + caption/medium tag.

Compresses the image BOTH ways at matched bytes — VQ-VAE tokens (QP192) and the best of AVIF/WebP — then
keeps whichever guide is more faithful by degraded LPIPS, BIASED toward VQ: the codec is chosen only when
it beats VQ by more than --lpips-margin (default 0.05). Ties / close / VQ-better all keep VQ. Writes a
CSC2 (.nit, "neural image tiles") container with the winning payload + the gemini-flash caption/medium tag.

    python encode.py img.jpg                            # -> img.nit, auto guide select + tag
    python encode.py img.jpg -o out.nit --lpips-margin 0.05
    python encode.py img.jpg --force-vq                 # skip selection, always VQ
    python encode.py img.jpg --no-tag                   # skip flash tag (medium=photo)
    python encode.py img.jpg --mode native_ov           # overlap/native tile encoding (default)
    python encode.py img.jpg --mode black               # legacy letterbox encoding
"""
import argparse, io, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, torch
from PIL import Image

HERE = Path(__file__).resolve().parent
CODEC = HERE / "codec"
MODEL = HERE / "model"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(CODEC))
from pipeline import classify_and_caption, pack, CODEC_FMTS
_LP = None


def _run_codec(label, cmd, expected):
    r = subprocess.run(cmd, cwd=str(CODEC), capture_output=True, text=True)
    if r.returncode != 0:
        detail = (r.stderr or r.stdout).strip()
        sys.exit(f"{label} failed with exit code {r.returncode}:\n{detail}")
    if not expected.exists():
        detail = (r.stderr or r.stdout).strip()
        msg = f"{label} did not create expected output: {expected}"
        if detail:
            msg += f"\n{detail}"
        sys.exit(msg)
    return r


def _fmt_bytes(n):
    return f"{n:,} B ({n / 1024:.2f} KB)"


def _lpips(pil, ref_t, MW, MH):
    global _LP
    if _LP is None:
        import piq
        _LP = piq.LPIPS()
    x = torch.from_numpy(np.asarray(pil.convert("RGB").resize((MW, MH), Image.LANCZOS)).copy()).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad():
        return _LP(x, ref_t).item()


def _codec_to_budget(orig, fmt, budget):
    """Largest downscale of orig in `fmt` whose bytes fit `budget`. Returns (blob_bytes, recon_full)."""
    W, H = orig.size; lo, hi = 0.03, 1.0; blob = None
    for _ in range(16):
        sc = (lo + hi) / 2; sm = orig.resize((max(1, round(W * sc)), max(1, round(H * sc))), Image.LANCZOS)
        b = io.BytesIO(); sm.save(b, fmt, quality=90)
        if b.tell() < budget: lo = sc; blob = b.getvalue()
        else: hi = sc
    if blob is None:  # even tiny didn't fit; use smallest
        sm = orig.resize((max(1, round(W * lo)), max(1, round(H * lo))), Image.LANCZOS)
        b = io.BytesIO(); sm.save(b, fmt, quality=90); blob = b.getvalue()
    recon = Image.open(io.BytesIO(blob)).convert("RGB").resize((W, H), Image.LANCZOS)
    return blob, recon


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("-o", "--output", default=None, help="output .nit (default: <input>.nit)")
    ap.add_argument("--qp", type=int, default=192,
                    help="VQ tile encode resolution; multiple of 32 in 32..512 (default 192)")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="[legacy alias] use codec only if it beats VQ degraded LPIPS by > this")
    ap.add_argument("--lpips-margin", type=float, default=None,
                    help="use codec only if it beats VQ degraded LPIPS by > this (default 0.05; "
                         "overrides --margin if both given)")
    ap.add_argument("--mode", choices=["black", "neighbor", "native", "native_ov"], default="native_ov",
                    help="tile encoding mode passed to inner VQ codec (default: native_ov)")
    ap.add_argument("--tile-margin", type=int, default=32,
                    help="overlap margin in pixels for native_ov tile mode; 0..255 (default 32)")
    ap.add_argument("--force-vq", action="store_true"); ap.add_argument("--no-tag", action="store_true")
    args = ap.parse_args()
    if args.qp % 32 != 0 or not 32 <= args.qp <= 512:
        ap.error("--qp must be a multiple of 32 in the range 32..512")
    if not 0 <= args.tile_margin <= 255:
        ap.error("--tile-margin must be in the range 0..255")
    lpips_margin = args.lpips_margin if args.lpips_margin is not None else args.margin

    orig = Image.open(args.input).convert("RGB"); W, H = orig.size
    MW, MH = (768, round(768 * H / W)) if W >= H else (round(768 * W / H), 768)
    ref_t = torch.from_numpy(np.asarray(orig.resize((MW, MH), Image.LANCZOS)).copy()).permute(2, 0, 1).float()[None] / 255

    # VQ-VAE guide + token payload
    with tempfile.NamedTemporaryFile(suffix=".nit", delete=False) as tf: nit = Path(tf.name)
    blur = nit.with_suffix(".b.png")
    try:
        _run_codec("VQ token encode",
                   [sys.executable, "encode.py", str(Path(args.input).resolve()), "-o", str(nit),
                    "--qp", str(args.qp), "--ckpt", str(MODEL / "vqvae_v05.pt"),
                    "--hp", str(MODEL / "hyperprior.pt"),
                    "--mode", args.mode, "--margin", str(args.tile_margin)],
                   nit)
        _run_codec("VQ guide decode",
                   [sys.executable, "decode.py", str(nit), "-o", str(blur), "--tokens-only",
                    "--ckpt", str(MODEL / "vqvae_v05.pt"), "--hp", str(MODEL / "hyperprior.pt")],
                   blur)
        inner = nit.read_bytes(); B = len(inner); vq_guide = Image.open(blur).convert("RGB")
    finally:
        nit.unlink(missing_ok=True); blur.unlink(missing_ok=True)
    vq_deg = _lpips(vq_guide, ref_t, MW, MH)

    guide_type, codec_fmt, payload, note = 0, 0, inner, f"VQ (deg {vq_deg:.3f})"
    if not args.force_vq:
        best = None
        for fi, fmt in enumerate(CODEC_FMTS):
            blob, recon = _codec_to_budget(orig, fmt, B)
            d = _lpips(recon, ref_t, MW, MH)
            if best is None or d < best[0]: best = (d, fi, blob)
        codec_deg, fi, blob = best
        if (vq_deg - codec_deg) > lpips_margin:                      # codec clearly better -> use it
            guide_type, codec_fmt, payload = 1, fi, blob
            note = f"{CODEC_FMTS[fi]} (deg {codec_deg:.3f} < VQ {vq_deg:.3f} by {vq_deg-codec_deg:.3f} > {lpips_margin})"
        else:
            note = f"VQ (deg {vq_deg:.3f}; best codec {codec_deg:.3f}, gap {vq_deg-codec_deg:+.3f} <= {lpips_margin})"

    medium, caption = ("photo", "") if args.no_tag else classify_and_caption(orig)
    out = Path(args.output) if args.output else Path(args.input).with_suffix(".nit")
    out.write_bytes(pack(payload, guide_type, medium, caption, W, H, codec_fmt))
    print(f"Encoded -> {out}")
    print(f"  output size: {W}x{H}")
    print(f"  .nit size: {_fmt_bytes(out.stat().st_size)}")
    print(f"  medium: {medium}")
    print(f"  guide selected: {note}")
    print(f"  description: {caption or '(none)'}")


if __name__ == "__main__":
    main()
