"""Optional SDXL generator backend for Cascade (--generator sdxl).

An open / reproducible alternative to Gemini for images up to 1344px on the long side: feeds the
blurry guide into SDXL via the validated channel-concat + LoRA recipe (widen conv_in to accept the
blurry latent concatenated to the noisy latent; conditioned DDIM). The bundled LoRA is trained for
aspect-flexible ~1MP SDXL buckets up to 1344px, and callers enforce max(W,H) <= 1344. Weights:
model/sdxl_lora.pt (LoRA + widened conv_in); the SDXL base is fetched from HF on first use.
"""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image

SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
_NEG = "blurry, low quality"

# SDXL ~1MP aspect buckets (h, w) the 1024 aspect-flexible LoRA was trained on.
BUCKETS = [(1024, 1024), (896, 1152), (1152, 896), (832, 1216), (1216, 832),
           (768, 1344), (1344, 768), (960, 1088), (1088, 960)]


def nearest_bucket(W, H):
    ar = W / H
    return min(BUCKETS, key=lambda b: abs((b[1] / b[0]) - ar))  # returns (h, w)


def _dev():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _enc_prompt(prompt, toks, tes, dev):
    e = []
    for tok, te in zip(toks, tes):
        ids = tok(prompt, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids.to(dev)
        o = te(ids, output_hidden_states=True); pooled = o[0]; e.append(o.hidden_states[-2])
    return torch.cat(e, -1), pooled


_CACHE = {}


def _load(ckpt_path, dev):
    if ckpt_path in _CACHE:
        return _CACHE[ckpt_path]
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
    from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
    from peft import LoraConfig, set_peft_model_state_dict
    ck = torch.load(ckpt_path, map_location="cpu")
    # Force float32 everywhere — diffusers can load mixed fp16/fp32 shards, which breaks attention on
    # MPS/CPU ("query float, key/value Half"). float32 is consistent and correct on all devices.
    vae = AutoencoderKL.from_pretrained(SDXL, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
    tok1 = CLIPTokenizer.from_pretrained(SDXL, subfolder="tokenizer"); tok2 = CLIPTokenizer.from_pretrained(SDXL, subfolder="tokenizer_2")
    te1 = CLIPTextModel.from_pretrained(SDXL, subfolder="text_encoder", torch_dtype=torch.float32).to(dev).eval()
    te2 = CLIPTextModelWithProjection.from_pretrained(SDXL, subfolder="text_encoder_2", torch_dtype=torch.float32).to(dev).eval()
    unet = UNet2DConditionModel.from_pretrained(SDXL, subfolder="unet", torch_dtype=torch.float32).to(dev)
    old = unet.conv_in
    unet.conv_in = nn.Conv2d(old.in_channels * 2, old.out_channels, old.kernel_size, padding=old.padding)
    unet.add_adapter(LoraConfig(r=ck["rank"], lora_alpha=ck["rank"], target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    set_peft_model_state_dict(unet, ck["lora"]); unet.conv_in.load_state_dict(ck["conv_in"]); unet.conv_in.to(dev); unet.eval()
    sched = DDIMScheduler.from_pretrained(SDXL, subfolder="scheduler")
    with torch.no_grad():
        emb, pooled = _enc_prompt(ck.get("prompt", "a sharp, high-quality, detailed photograph"), [tok1, tok2], [te1, te2], dev)
        nemb, npooled = _enc_prompt(_NEG, [tok1, tok2], [te1, te2], dev)
    _CACHE[ckpt_path] = (vae, unet, sched, emb, pooled, nemb, npooled, vae.config.scaling_factor)
    return _CACHE[ckpt_path]


@torch.no_grad()
def decode_sdxl(blur_pil, W, H, ckpt_path, strength=0.5, steps=40, guidance=1.5, target_hw=(512, 512)):
    """Blurry guide -> SDXL-sharpened reconstruction at original (W,H). target_hw = the (h,w) the model
    runs at: (512,512) for the 512 LoRA; an aspect bucket (mult of 64) for the 1024 aspect-flex LoRA,
    with matching SDXL time_ids. Must be multiples of 8 (VAE stride)."""
    dev = _dev()
    th, tw = target_hw
    vae, unet, sched, emb, pooled, nemb, npooled, sf = _load(ckpt_path, dev)
    tid = torch.tensor([[th, tw, 0, 0, th, tw]], device=dev, dtype=emb.dtype)
    x = torch.from_numpy(np.asarray(blur_pil.convert("RGB").resize((tw, th), Image.LANCZOS)).copy()).permute(2, 0, 1).float()[None].to(dev) / 127.5 - 1
    zc = vae.encode(x).latent_dist.mean * sf
    sched.set_timesteps(steps, device=dev); ts = sched.timesteps[int(steps * (1 - strength)):]
    g = torch.Generator(device=dev).manual_seed(0)
    z = sched.add_noise(zc, torch.randn(zc.shape, generator=g, device=dev), ts[:1])
    use_amp = (dev == "cuda")
    for t in ts:
        inp = torch.cat([z, zc], 1)
        if use_amp:
            with torch.autocast("cuda", torch.bfloat16):
                ec = unet(inp, t, encoder_hidden_states=emb, added_cond_kwargs={"text_embeds": pooled, "time_ids": tid}).sample
                eu = unet(inp, t, encoder_hidden_states=nemb, added_cond_kwargs={"text_embeds": npooled, "time_ids": tid}).sample
        else:
            ec = unet(inp, t, encoder_hidden_states=emb, added_cond_kwargs={"text_embeds": pooled, "time_ids": tid}).sample
            eu = unet(inp, t, encoder_hidden_states=nemb, added_cond_kwargs={"text_embeds": npooled, "time_ids": tid}).sample
        z = sched.step((eu + guidance * (ec - eu)).float(), t, z).prev_sample
    img = vae.decode(z / sf).sample
    arr = ((img.clamp(-1, 1) + 1) / 2 * 255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr).resize((W, H), Image.LANCZOS)
