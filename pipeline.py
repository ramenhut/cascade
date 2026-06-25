"""Cascade container (CSC2, "Cascade v2") + prompt routing. File extension: .nit (neural image tiles).

Beta adds content-adaptive GUIDE SELECTION over DiffusionGemini: at encode the image is compressed BOTH
ways (VQ-VAE neural tiles and the best modern codec at matched bytes), and whichever produces the more
faithful guide (degraded LPIPS) is kept — biased toward VQ (the codec is used only when CLEARLY better by
a margin). The CSC2 container therefore carries EITHER a VQ neural-tile payload OR a codec blob, plus the
caption/medium tag that routes the Gemini prompt. The CSC2 magic distinguishes it from the canonical
CSC1 .nit (which a CSC2 decoder will reject, and vice-versa).

CSC2 (.nit) layout (little-endian):
  b"CSC2" | version(1) | guide_type(1: 0=VQ,1=codec) | medium_idx(1) | W(u32) | H(u32) |
  caption_len(u16) | caption | [if codec: codec_fmt(1: 0=AVIF,1=WEBP)] | payload
"""
import io, json, os, struct, sys
from pathlib import Path

CODEC = (Path(__file__).resolve().parent / "codec")
sys.path.insert(0, str(CODEC))
from decode import ENHANCE_PROMPT, STYLE_PROMPT  # canonical photo + preserve-style prompts

CSC2_MAGIC = b"CSC2"
CSC2_VERSION = 1
MEDIA = ["photo", "anime", "screenshot"]
CODEC_FMTS = ["AVIF", "WEBP"]
FLASH_MODEL = "gemini-flash-latest"

SCREENSHOT_PROMPT = (
    "This is a low-resolution, blurry compressed screenshot or text/UI image. Reconstruct it crisply "
    "while preserving the EXACT layout, sharp straight edges, solid color fills, icons, and especially "
    "any TEXT: keep text legible, in the same place, same wording — do not paraphrase, translate, or "
    "invent text, and do not add photographic texture, grain, or 3D shading. Keep the same composition "
    "and colors. If solid black letterbox bars (padding) are present, keep those areas solid black."
)
PROMPTS = {"photo": ENHANCE_PROMPT, "anime": STYLE_PROMPT, "screenshot": SCREENSHOT_PROMPT}


def classify_and_caption(pil):
    """Return (medium:str, caption:str) for the ORIGINAL. Fail-safe -> ('photo', '')."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return "photo", ""
    try:
        from google import genai
        from google.genai import types
        sm = pil.copy(); sm.thumbnail((512, 512))
        buf = io.BytesIO(); sm.convert("RGB").save(buf, "PNG")
        client = genai.Client(api_key=key)
        r = client.models.generate_content(
            model=FLASH_MODEL,
            contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
                      types.Part.from_text(text=(
                          "Classify this image's medium as exactly one of: photo, anime, screenshot "
                          "(use 'anime' for any flat 2D illustration/cartoon/art; 'screenshot' for "
                          "UI/text/diagrams). Then write one concise factual caption (main objects, "
                          "colors, layout). Reply ONLY as compact JSON: "
                          '{"medium":"...","caption":"..."}'))])
        txt = r.text.strip()
        if txt.startswith("```"):
            txt = txt.strip("`"); txt = txt[txt.find("{"):]
        obj = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        medium = obj.get("medium", "photo").lower().strip()
        if medium not in PROMPTS:
            medium = "photo"
        return medium, str(obj.get("caption", "")).replace("\n", " ").strip()
    except Exception as e:
        print(f"[pipeline] classify failed ({str(e)[:60]}); defaulting to photo", file=sys.stderr)
        return "photo", ""


def routed_prompt(medium, caption):
    base = PROMPTS.get(medium, ENHANCE_PROMPT)
    return base + (f" The original image depicts: {caption}" if caption else "")


def pack(payload: bytes, guide_type: int, medium: str, caption: str, W: int, H: int, codec_fmt=0) -> bytes:
    cap = caption.encode("utf-8")[:65535]
    mi = MEDIA.index(medium) if medium in MEDIA else 0
    head = CSC2_MAGIC + bytes([CSC2_VERSION, guide_type & 1, mi]) + struct.pack("<IIH", W, H, len(cap)) + cap
    if guide_type == 1:
        head += bytes([codec_fmt & 1])
    return head + payload


def unpack(data: bytes):
    """-> dict(guide_type, medium, caption, W, H, codec_fmt, payload)."""
    assert data[:4] == CSC2_MAGIC, "not a CSC2 .nit (Cascade) file — wrong magic (a canonical CSC1 .nit is not a CSC2 file)"
    gt = data[5]; medium = MEDIA[data[6]] if data[6] < len(MEDIA) else "photo"
    W, H, clen = struct.unpack("<IIH", data[7:17]); off = 17
    caption = data[off:off + clen].decode("utf-8", "replace"); off += clen
    codec_fmt = 0
    if gt == 1:
        codec_fmt = data[off]; off += 1
    return dict(guide_type=gt, medium=medium, caption=caption, W=W, H=H,
                codec_fmt=codec_fmt, payload=data[off:])
