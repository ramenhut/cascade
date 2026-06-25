"""Hyperprior entropy coding for Cascade tokens — resolution-aware (QP) variant.

Same lossless arithmetic coding as release2's hp_codec, but the token-grid shapes are derived from
the per-tile **encode resolution** (the QP knob) instead of being hardcoded to 192². The resolution
(in units of 32px) is stored in the header so decode can reconstruct the grid.

Coding order per tile (chain rule): L1 (marginal) → L2|L1 → L3|L1L2 → L4|L1L2L3, one range coder
per tile; decode regenerates each level's probabilities autoregressively from already-decoded
coarser levels (nothing about the distribution is stored).

Formats:
  v4 = legacy release2 hyperprior (assumed 192²).
  v5 = QP hyperprior (header carries the resolution byte, square letterbox tiles).
  v6 = native-aspect hyperprior (header carries res, mode, margin; per-tile dims computed from box).

Image I/O / tiling / letterboxing / Gemini decode live in encode.py / decode.py.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import constriction as cs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vqvae import VQVAE4L, VQVAE4LConfig
from hyperprior import Hyperprior
from letterbox import letterbox, tile_grid, tile_boxes, lb_dims
from torchvision import transforms

MAGIC = b"CSC1"
VERSION_HP = 4          # legacy release2 hyperprior (192² only)
VERSION_QP = 5          # resolution-aware hyperprior (header stores res//32, square tiles)
VERSION_NATIVE = 6      # native-aspect hyperprior (header stores res//32, mode, margin)
LEVELS = ["l1", "l2", "l3", "l4"]
STRIDES = {"l1": 32, "l2": 16, "l3": 8, "l4": 4}
_NORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])

MODES = {"black": 0, "neighbor": 1, "native": 2, "native_ov": 3}
MODE_BY_ID = {v: k for k, v in MODES.items()}
QP_MIN = 32
QP_MAX = 512


def _validate_encode_params(res, mode, margin):
    if mode not in MODES:
        raise ValueError(f"unknown tile mode {mode!r}; expected one of {', '.join(MODES)}")
    if res % 32 != 0 or not QP_MIN <= res <= QP_MAX:
        raise ValueError(f"encode resolution (QP) must be a multiple of 32 in {QP_MIN}..{QP_MAX}")
    if not 0 <= margin <= 255:
        raise ValueError("tile overlap margin must be in the range 0..255")


# ----------------------------------------------------------------- geometry helpers

def native_dims(tw, th, qp):
    """Multiples of 32, long side = qp, aspect ~preserved (nearest 32 on the short side)."""
    if tw >= th:
        return qp, max(32, round(th / tw * qp / 32) * 32)
    return max(32, round(tw / th * qp / 32) * 32), qp


def _expanded_box(box, margin, W, H):
    l, t, r, b = box
    return (max(0, l - margin), max(0, t - margin), min(W, r + margin), min(H, b + margin))


def _shapes_for_dims(nw, nh):
    """Per-tile token-grid shapes from encode square dimensions (nw, nh)."""
    return {lv: (nh // STRIDES[lv], nw // STRIDES[lv]) for lv in LEVELS}


def shapes_for(res):
    """Token-grid (h,w) per level for a square encode resolution `res` (a multiple of 32)."""
    return {lv: (res // STRIDES[lv], res // STRIDES[lv]) for lv in LEVELS}


def _tile_encode_dims(box, mode, qp, margin, W, H):
    """Return (nw, nh) of the encode square for this tile, without needing image pixels."""
    l, t, r, b = box
    tw, th = r - l, b - t
    if mode == "native":
        return native_dims(tw, th, qp)
    if mode == "native_ov":
        e = _expanded_box(box, margin, W, H)
        return native_dims(e[2] - e[0], e[3] - e[1], qp)
    # black / neighbor: square at qp
    return qp, qp


def _tile_square(pil, box, mode, qp, margin):
    """Build the PIL image fed to the encoder for this tile. Returns (square_pil, (nw, nh))."""
    from PIL import Image as _Image
    import numpy as _np
    l, t, r, b = box
    W, H = pil.size
    tw, th = r - l, b - t

    if mode == "native":
        nw, nh = native_dims(tw, th, qp)
        return pil.crop(box).resize((nw, nh), _Image.LANCZOS), (nw, nh)

    if mode == "native_ov":
        e = _expanded_box(box, margin, W, H)
        etw, eth = e[2] - e[0], e[3] - e[1]
        nw, nh = native_dims(etw, eth, qp)
        return pil.crop(e).resize((nw, nh), _Image.LANCZOS), (nw, nh)

    # black / neighbor: qp×qp letterbox (content byte-identical across the two)
    nw, nh = lb_dims(tw, th, qp)
    scale = nw / tw
    left = (qp - nw) // 2
    top = (qp - nh) // 2
    content = _np.asarray(pil.crop(box).resize((nw, nh), _Image.LANCZOS))
    if mode == "black":
        canvas = _np.zeros((qp, qp, 3), _np.uint8)
    else:  # neighbor — fill bars from real neighbours via an aligned expanded crop
        right = qp - nw - left
        bot = qp - nh - top
        ex0, ey0 = l - left / scale, t - top / scale
        ex1, ey1 = r + right / scale, b + bot / scale
        cx0, cy0, cx1, cy1 = round(ex0), round(ey0), round(ex1), round(ey1)
        cx0c, cy0c, cx1c, cy1c = max(0, cx0), max(0, cy0), min(W, cx1), min(H, cy1)
        crop = _np.asarray(pil.crop((cx0c, cy0c, cx1c, cy1c)))
        pl, pt, pr, pb = cx0c - cx0, cy0c - cy0, cx1 - cx1c, cy1 - cy1c
        if pl or pt or pr or pb:
            crop = _np.pad(crop, ((pt, pb), (pl, pr), (0, 0)), mode="edge")
        canvas = _np.asarray(_Image.fromarray(crop).resize((qp, qp), _Image.LANCZOS)).copy()
    canvas[top:top + nh, left:left + nw] = content
    return _Image.fromarray(canvas), (qp, qp)


# ----------------------------------------------------------------- model loading

def load_models(vqvae_path, hp_path, device=None):
    # CPU by default: range coding desyncs on sub-ULP probability differences, and GPU/MPS
    # neural-net forwards are not bit-reproducible. CPU token coding is fast and deterministic.
    device = device or torch.device("cpu")
    ck = torch.load(vqvae_path, map_location=device, weights_only=False)
    cfg = VQVAE4LConfig(**{k: v for k, v in ck["model_config"].items()
                          if k in VQVAE4LConfig.__dataclass_fields__})
    m = VQVAE4L(cfg).to(device).eval(); m.load_state_dict(ck["model"])
    hp = Hyperprior(cfg).to(device).eval()
    hp.load_state_dict(torch.load(hp_path, map_location=device)["hyperprior"])
    return m, hp, cfg, device


# ----------------------------------------------------------------- probability tables

def _probs_from_logits(logits):
    """(1,K,h,w) logits -> (h*w, K) float64 probabilities, row-major over (h,w)."""
    p = F.softmax(logits[0].float(), dim=0)          # (K,h,w)
    return p.permute(1, 2, 0).reshape(-1, p.shape[0]).cpu().numpy().astype(np.float64)


def _q(m, lv, idx):
    return getattr(m, f"vq_{lv}").lookup(idx)


# Per-level probability tables. Encode and decode MUST call these identical expressions (the
# float math has to be byte-for-byte the same on both sides; run on CPU). The target spatial sizes
# are passed in so the same code works at any encode resolution.

def _p_l1(hp, n1):
    p = F.softmax(hp.l1_prior.float(), dim=0).cpu().numpy().astype(np.float64)
    return np.ascontiguousarray(np.tile(p, (n1, 1)))


def _p_l2(hp, q1, s2):
    return np.ascontiguousarray(_probs_from_logits(hp.h_l2(hp._up(q1, s2))))


def _p_l3(hp, q1, q2, s3):
    f = torch.cat([hp._up(q1, s3), hp._up(q2, s3)], 1)
    return np.ascontiguousarray(_probs_from_logits(hp.h_l3(f)))


def _p_l4(hp, q1, q2, q3, s4):
    f = torch.cat([hp._up(q1, s4), hp._up(q2, s4), hp._up(q3, s4)], 1)
    return np.ascontiguousarray(_probs_from_logits(hp.h_l4(f)))


@torch.no_grad()
def _level_probs_encode(m, hp, idx):
    """All four levels' (n,K) prob tables. Shapes derived from the actual index maps."""
    q1, q2, q3 = (_q(m, "l1", idx["l1"]), _q(m, "l2", idx["l2"]), _q(m, "l3", idx["l3"]))
    s = {lv: tuple(idx[lv].shape[-2:]) for lv in LEVELS}
    n1 = s["l1"][0] * s["l1"][1]
    return {"l1": _p_l1(hp, n1), "l2": _p_l2(hp, q1, s["l2"]),
            "l3": _p_l3(hp, q1, q2, s["l3"]), "l4": _p_l4(hp, q1, q2, q3, s["l4"])}


def _encode_tile(m, hp, idx):
    probs = _level_probs_encode(m, hp, idx)
    enc = cs.stream.queue.RangeEncoder()
    model = cs.stream.model.Categorical(lazy=True)
    for lv in LEVELS:
        syms = idx[lv][0].reshape(-1).cpu().numpy().astype(np.int32)
        enc.encode(syms, model, np.ascontiguousarray(probs[lv]))
    return enc.get_compressed().tobytes()


@torch.no_grad()
def _decode_tile(m, hp, blob, device, shapes):
    """compressed bytes -> dict of (1,h,w) index tensors (autoregressive over levels)."""
    comp = np.frombuffer(blob, dtype=np.uint32).copy()
    dec = cs.stream.queue.RangeDecoder(comp)
    model = cs.stream.model.Categorical(lazy=True)

    def _take(lv, probs):
        s = dec.decode(model, probs)
        return torch.from_numpy(s.astype(np.int64)).reshape(1, *shapes[lv]).to(device)

    n1 = shapes["l1"][0] * shapes["l1"][1]
    idx = {}
    idx["l1"] = _take("l1", _p_l1(hp, n1)); q1 = _q(m, "l1", idx["l1"])
    idx["l2"] = _take("l2", _p_l2(hp, q1, shapes["l2"])); q2 = _q(m, "l2", idx["l2"])
    idx["l3"] = _take("l3", _p_l3(hp, q1, q2, shapes["l3"])); q3 = _q(m, "l3", idx["l3"])
    idx["l4"] = _take("l4", _p_l4(hp, q1, q2, q3, shapes["l4"]))
    return idx


# ----------------------------------------------------------------- encode / decode (public API)

@torch.no_grad()
def encode_hp(m, hp, device, pil_image, tile_size=512, res=192, mode="native", margin=32):
    """image -> hyperprior-coded CSC1 bytes.

    mode in {"black","neighbor","native","native_ov"} controls how each tile is encoded:
      black      — v5-compatible letterbox (square, black bars).  Written as VERSION_NATIVE for
                   uniformity; decoders only need VERSION_NATIVE path.
      neighbor   — like black but bars filled with real neighbour pixels (content byte-identical).
      native     — true-aspect crop, resized to multiples-of-32 (long side = res).
      native_ov  — native-aspect of an expanded crop (margin px overlap).

    For black/neighbor the file is VERSION_NATIVE (v6) — NOT v5 — so the mode byte is present.
    This keeps the format uniform; old v4/v5 files can still be decoded by decode_hp_tokens.
    """
    _validate_encode_params(res, mode, margin)
    pil = pil_image.convert("RGB"); W, H = pil.size
    cols, rows = tile_grid(W, H, tile_size)
    # v6 header: MAGIC + VERSION_NATIVE + W(u32) + H(u32) + cols(u16) + rows(u16) +
    #            res//32(u8) + mode(u8) + margin(u8)  = 4+1+4+4+2+2+1+1+1 = 20 bytes total
    header = bytearray(MAGIC)
    header.append(VERSION_NATIVE)
    header += struct.pack("<IIHHBBB", W, H, cols, rows, res // 32, MODES[mode], margin)
    body = bytearray()
    for box in tile_boxes(W, H, cols, rows):
        square, _ = _tile_square(pil, box, mode, res, margin)
        x = _NORM(square).unsqueeze(0).to(device)
        blob = _encode_tile(m, hp, m.encode(x))
        body += struct.pack("<I", len(blob)) + blob
    return bytes(header) + bytes(body), (cols, rows)


@torch.no_grad()
def decode_hp_tokens(m, hp, data, device):
    """hp-coded CSC1 bytes -> (tiles, (W,H), (cols,rows), mode, margin).

    Handles v4 (legacy 192²), v5 (QP square), and v6 (native-aspect) formats.
    v4 and v5 return mode="black", margin=0 for caller uniformity.
    """
    assert data[:4] == MAGIC and data[4] in (VERSION_HP, VERSION_QP, VERSION_NATIVE), \
        "not a CSC1 hyperprior file"
    version = data[4]
    off = 5

    if version == VERSION_NATIVE:
        # v6: W(u32) H(u32) cols(u16) rows(u16) res32(u8) mode_id(u8) margin(u8) = 15 bytes
        W, H, cols, rows, res32, mode_id, margin = struct.unpack("<IIHHBBB", data[off:off + 15])
        off += 15
        mode = MODE_BY_ID[mode_id]
        res = res32 * 32
        # Decode each tile with per-tile shapes derived from mode/box geometry
        tiles = []
        for box in tile_boxes(W, H, cols, rows):
            (blen,) = struct.unpack("<I", data[off:off + 4]); off += 4
            blob = data[off:off + blen]; off += blen
            nw, nh = _tile_encode_dims(box, mode, res, margin, W, H)
            shapes = _shapes_for_dims(nw, nh)
            tiles.append(_decode_tile(m, hp, blob, device, shapes))
        return tiles, (W, H), (cols, rows), mode, margin

    elif version == VERSION_QP:
        # v5: W(u32) H(u32) cols(u8) rows(u8) res32(u8) = 11 bytes
        W, H, cols, rows, res32 = struct.unpack("IIBBB", data[off:off + 11]); off += 11
        shapes = shapes_for(res32 * 32)
    else:
        # v4 legacy: W(u32) H(u32) cols(u8) rows(u8) = 10 bytes, assume 192²
        W, H, cols, rows = struct.unpack("IIBB", data[off:off + 10]); off += 10
        shapes = shapes_for(192)

    # v4/v5: uniform shapes, square letterbox tiles
    tiles = []
    for _ in range(cols * rows):
        (blen,) = struct.unpack("I", data[off:off + 4]); off += 4
        blob = data[off:off + blen]; off += blen
        tiles.append(_decode_tile(m, hp, blob, device, shapes))
    return tiles, (W, H), (cols, rows), "black", 0
