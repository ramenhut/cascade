#!/usr/bin/env python3
"""Cascade decoder — CSC2 (VQ tokens OR codec blob) -> medium-routed Gemini reconstruction.

  1. Unwrap CSC2 -> guide_type, (medium, caption), W, H, payload.
  2. Build the blurry GUIDE: VQ token decode (canonical, offline) OR decompress the codec blob + upscale.
  3. Route the Gemini prompt by MEDIUM + append CAPTION; one (or best-of-N) Gemini pass -> original size.

    python decode.py img.nit -o out.png
    python decode.py img.nit -o guide.png --tokens-only     # the guide only, no model (offline)
    python decode.py img.nit -o out.png --n-decode 4        # best-of-N: keep most guide-faithful draw
    python decode.py img.nit -o out.png --force-medium anime
    python decode.py img.nit -o out.png --generator sdxl    # open SDXL backend (up to 1344px long side)
"""
import argparse, io, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
CODEC = HERE / "codec"
MODEL = HERE / "model"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(CODEC))
from pipeline import unpack, routed_prompt, PROMPTS, CODEC_FMTS
from letterbox import letterbox, unletterbox, pick_tier
from decode import gemini_enhance, TIER_PX


def _fmt_bytes(n):
    return f"{n:,} B ({n / 1024:.2f} KB)"


def _select_best(candidates, guide):
    """Best-of-N: keep the draw most faithful to the GUIDE at low res (LPIPS@256) — rewards structure/
    color fidelity, ignores added detail, rejects hallucination-outlier draws. Validated near-oracle."""
    if len(candidates) == 1:
        return candidates[0]
    import torch, piq
    lp = piq.LPIPS(); s = (256, 256)
    def t(im): return torch.from_numpy(np.asarray(im.convert("RGB").resize(s, Image.LANCZOS)).copy()).permute(2, 0, 1).float()[None] / 255
    g = t(guide)
    with torch.no_grad():
        scores = [lp(t(c), g).item() for c in candidates]
    return candidates[int(np.argmin(scores))]


def build_guide(info):
    """Reconstruct the blurry guide at original W×H from either VQ tokens or a codec blob."""
    W, H = info["W"], info["H"]
    if info["guide_type"] == 0:                                   # VQ tokens -> canonical decode
        with tempfile.NamedTemporaryFile(suffix=".nit", delete=False) as tf:
            tmp = Path(tf.name); tmp.write_bytes(info["payload"])
        gp = tmp.with_suffix(".g.png")
        r = subprocess.run([sys.executable, "decode.py", str(tmp), "-o", str(gp), "--tokens-only",
                            "--ckpt", str(MODEL / "vqvae_v05.pt"), "--hp", str(MODEL / "hyperprior.pt")],
                           cwd=str(CODEC), capture_output=True, text=True)
        tmp.unlink()
        if r.returncode != 0 or not gp.exists():
            sys.exit(f"VQ token decode failed:\n{r.stderr or r.stdout}")
        g = Image.open(gp).convert("RGB"); gp.unlink(); return g, f"VQ tokens"
    # codec blob -> decompress + upscale to original size
    g = Image.open(io.BytesIO(info["payload"])).convert("RGB").resize((W, H), Image.LANCZOS)
    return g, f"{CODEC_FMTS[info['codec_fmt']]} codec"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--tokens-only", action="store_true", help="output the blurry guide only, no Gemini")
    ap.add_argument("--resolution", choices=["auto", "1K", "2K", "4K"], default="auto")
    ap.add_argument("--force-medium", choices=list(PROMPTS), default=None)
    ap.add_argument("--n-decode", type=int, default=1, help="best-of-N draws, keep most guide-faithful (default 1)")
    ap.add_argument("--generator", choices=["gemini", "sdxl"], default="gemini",
                    help="restoration backend (default gemini). 'sdxl' is open/offline up to 1344px long side.")
    args = ap.parse_args()

    input_path = Path(args.input)
    input_bytes = input_path.stat().st_size
    info = unpack(input_path.read_bytes())
    medium = args.force_medium or info["medium"]
    guide, gtag = build_guide(info); W, H = info["W"], info["H"]

    if args.tokens_only:
        guide.save(args.output)
        print(f"Decoded guide -> {args.output}")
        print(f"  input .nit size: {_fmt_bytes(input_bytes)}")
        print(f"  output size: {W}x{H}")
        print(f"  medium: {medium}")
        print(f"  guide: {gtag}")
        print(f"  description: {info['caption'] or '(none)'}")
        return

    # SDXL backend: open/reproducible. The bundled LoRA is the aspect-flexible 1024 model, so it runs at
    # the nearest ~1MP aspect bucket (<=1344 long side). Larger inputs are rejected (use Gemini).
    if args.generator == "sdxl":
        if max(W, H) > 1344:
            sys.exit(f"sdxl max input size is 1344px on the long side (this image is {W}x{H}). "
                     f"Use --generator gemini for larger images.")
        from sdxl_decode import decode_sdxl, nearest_bucket
        bh, bw = nearest_bucket(W, H)
        out = decode_sdxl(guide, W, H, str(MODEL / "sdxl_lora.pt"), target_hw=(bh, bw))
        out.save(args.output)
        print(f"Decoded -> {args.output}")
        print(f"  input .nit size: {_fmt_bytes(input_bytes)}")
        print(f"  output size: {W}x{H}")
        print(f"  medium: {medium}")
        print(f"  guide: {gtag}")
        print(f"  generator: SDXL @ {bh}x{bw} bucket")
        print(f"  description: {info['caption'] or '(none)'}")
        return

    tier = pick_tier(W, H)[0] if args.resolution == "auto" else args.resolution
    G = TIER_PX[tier]
    sq, _ = letterbox(guide, size=max(W, H)); sq = sq.resize((G, G), Image.LANCZOS)
    prompt = routed_prompt(medium, info["caption"])
    n = max(1, args.n_decode); candidates = []
    for _ in range(n):
        try:
            candidates.append(gemini_enhance(sq, image_size=tier, prompt=prompt))
        except Exception as e:
            print(f"  [draw] Gemini failed: {str(e)[:60]}")
    if not candidates:
        sys.exit("all Gemini draws failed")
    best = _select_best(candidates, guide)
    unletterbox(best, W, H).save(args.output)
    tag = f", best-of-{len(candidates)}" if n > 1 else ""
    print(f"Decoded -> {args.output}")
    print(f"  input .nit size: {_fmt_bytes(input_bytes)}")
    print(f"  output size: {W}x{H}")
    print(f"  medium: {medium}")
    print(f"  guide: {gtag}")
    print(f"  generator: Gemini @ {tier}{tag}")
    print(f"  description: {info['caption'] or '(none)'}")


if __name__ == "__main__":
    main()
