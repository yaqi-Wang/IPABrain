"""Decode synthesised NSD-perception fMRI latents with SDXL-UnCLIP.

This is the matching decoder for the saved BrainVAE / S2N latents shaped
``(N, 256, 1664)``. It uses ``unclip6.yaml`` whose UNet cross-attention
``context_dim`` is 1664, instead of SDXL Refiner which expects 1280-d text
tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.utils.config import load_yaml
from src.utils.paths import ProjectPaths


DEFAULT_UNCLIP_CKPT = str(ROOT / "huggingface" / "unclip6_epoch0_step110000.ckpt")
DEFAULT_OPENCLIP = "huggingface/open_clip_pytorch_model.bin"


def _set_first_stage_vanilla_attention(config: dict) -> None:
    """Avoid xformers-only VAE attention when xformers is not installed."""
    first_stage = config["model"]["params"]["first_stage_config"]["params"]
    ddconfig = first_stage.get("ddconfig", {})
    if ddconfig.get("attn_type") == "vanilla-xformers":
        ddconfig["attn_type"] = "vanilla"


def _set_openclip_local_path(config: dict, openclip_path: Path) -> None:
    emb_models = config["model"]["params"]["conditioner_config"]["params"]["emb_models"]
    for emb in emb_models:
        if emb.get("target", "").endswith("FrozenOpenCLIPImageEmbedder"):
            params = emb.setdefault("params", {})
            params["version"] = str(openclip_path)
            params["cache_dir"] = str(openclip_path.parent)


def load_sdxl_unclip(
    generative_models_path: Path,
    unclip_ckpt: Path,
    openclip_path: Path,
    device: torch.device,
    num_steps: int,
) -> tuple[torch.nn.Module, torch.Tensor, float]:
    sys.path.append(str(generative_models_path))
    sys.path.append(str(generative_models_path.parent))

    from omegaconf import OmegaConf
    from sgm.models.diffusion import DiffusionEngine

    config_path = generative_models_path / "configs" / "unclip6.yaml"
    cfg = OmegaConf.load(str(config_path))
    cfg = OmegaConf.to_container(cfg, resolve=True)
    _set_openclip_local_path(cfg, openclip_path)
    _set_first_stage_vanilla_attention(cfg)

    params = cfg["model"]["params"]
    first_stage_config = params["first_stage_config"]
    first_stage_config["target"] = "sgm.models.autoencoder.AutoencoderKL"
    params["sampler_config"]["params"]["num_steps"] = int(num_steps)
    params["sampler_config"]["params"]["device"] = str(device)

    engine = DiffusionEngine(
        network_config=params["network_config"],
        denoiser_config=params["denoiser_config"],
        first_stage_config=first_stage_config,
        conditioner_config=params["conditioner_config"],
        sampler_config=params["sampler_config"],
        scale_factor=params["scale_factor"],
        disable_first_stage_autocast=params["disable_first_stage_autocast"],
    )
    engine.eval().requires_grad_(False).to(device)

    ckpt = torch.load(unclip_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = engine.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[unclip] {len(unexpected)} unexpected keys")
    if missing:
        print(f"[unclip] {len(missing)} missing keys")

    batch = {
        "jpg": torch.randn(1, 3, 1, 1, device=device),
        "original_size_as_tuple": torch.ones(1, 2, device=device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2, device=device),
    }
    with torch.no_grad():
        vector_suffix = engine.conditioner(batch)["vector"].to(device)

    offset_noise_level = float(params["loss_fn_config"]["params"].get("offset_noise_level", 0.04))
    print("vector_suffix", tuple(vector_suffix.shape))
    return engine, vector_suffix, offset_noise_level


def decoded_to_rgb(decoded: torch.Tensor, mode: str = "legacy") -> torch.Tensor:
    """Convert VAE's signed RGB to image coordinates; retain old callers' mapping."""
    if mode == "standard":
        return (decoded + 1.0).mul(0.5).clamp(0.0, 1.0)
    if mode == "legacy":
        return (decoded * 0.8 + 0.2).clamp(0.0, 1.0)
    raise ValueError(f"Unknown RGB mode: {mode}")


def unclip_decode_batch(
    tokens: torch.Tensor,
    engine: torch.nn.Module,
    vector_suffix: torch.Tensor,
    offset_noise_level: float,
    init_latent: torch.Tensor | None = None,
    img2img_strength: float = 1.0,
    seed: int | None = None,
    init_latent_space: str = "diffusion",
    rgb_mode: str = "legacy",
) -> torch.Tensor:
    from sgm.util import append_dims

    if tokens.ndim != 3 or tokens.shape[1:] != (256, 1664):
        raise ValueError(f"Expected tokens (B, 256, 1664), got {tuple(tokens.shape)}")
    if init_latent_space not in {"raw", "diffusion"}:
        raise ValueError("init_latent_space must be raw or diffusion")
    if rgb_mode not in {"standard", "legacy"}:
        raise ValueError("rgb_mode must be standard or legacy")

    device = next(engine.parameters()).device
    tokens = tokens.to(device)
    batch = tokens.shape[0]

    if seed is not None:
        torch.manual_seed(seed)

    autocast_enabled = device.type == "cuda"
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled), engine.ema_scope():
        if init_latent is None:
            z = torch.randn(batch, 4, 96, 96, device=device)
            discretization = engine.sampler.discretization
        else:
            if tuple(init_latent.shape) != (batch, 4, 96, 96):
                raise ValueError(f"Expected init_latent {(batch, 4, 96, 96)}, got {tuple(init_latent.shape)}")
            if not 0.0 < img2img_strength <= 1.0:
                raise ValueError("img2img_strength must be in (0, 1]")
            z = init_latent.to(device=device, dtype=tokens.dtype)
            if init_latent_space == "raw":
                z = z * float(engine.scale_factor)

            class StrengthDiscretization:
                def __init__(self, base, strength: float) -> None:
                    self.base = base
                    self.strength = strength

                def __call__(self, *args, **kwargs):
                    values = self.base(*args, **kwargs)
                    keep = max(2, int(round(self.strength * (len(values) - 1))) + 1)
                    return values[-keep:]

            discretization = StrengthDiscretization(engine.sampler.discretization, img2img_strength)
        c = {
            "crossattn": tokens,
            "vector": vector_suffix.repeat(batch, 1),
        }
        uc = {
            "crossattn": torch.randn_like(tokens),
            "vector": vector_suffix.repeat(batch, 1),
        }

        noise = torch.randn_like(z)
        sigmas = discretization(engine.sampler.num_steps, device=str(device))
        sigma = sigmas[0].to(z.device)
        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(torch.randn(z.shape[0], device=z.device), z.ndim)

        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(1.0 + sigmas[0].to(z.device) ** 2.0)

        def denoiser(x, sigma_, cond):
            return engine.denoiser(engine.model, x, sigma_, cond)

        original_discretization = engine.sampler.discretization
        engine.sampler.discretization = discretization
        try:
            samples_z = engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        finally:
            engine.sampler.discretization = original_discretization
        samples_x = engine.decode_first_stage(samples_z)
        return decoded_to_rgb(samples_x, rgb_mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode synthesised fMRI latents with SDXL-UnCLIP.")
    parser.add_argument("--paths", default="configs/paths.example.yaml")
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--generation-name", default="nsd_perception")
    parser.add_argument("--latent-source", choices=("recon", "true", "fm"), default="recon")
    parser.add_argument("--unclip-ckpt", default=DEFAULT_UNCLIP_CKPT)
    parser.add_argument("--openclip-path", default=DEFAULT_OPENCLIP)
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=38)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--low-level-latents",
        default="",
        help="predicted_sdxl_vae.pt from predict_low_level_vae.py. Enables low-level img2img initialization.",
    )
    parser.add_argument("--img2img-strength", type=float, default=0.4)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--save-thumbnails", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths.from_mapping(load_yaml(args.paths), ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.dry_run else "cpu")

    gen_dir = paths.output_root / "nsd_generation" / f"subj{args.subject:02d}" / args.generation_name
    latent_file_map = {
        "recon": "all_clipvoxels_recon.pt",
        "true": "all_clipvoxels.pt",
        "fm": "all_clipvoxels_fm.pt",
    }
    latent_path = gen_dir / latent_file_map[args.latent_source]
    image_id_path = gen_dir / "all_image_id.pt"
    if not latent_path.exists():
        raise FileNotFoundError(latent_path)
    if not image_id_path.exists():
        raise FileNotFoundError(image_id_path)

    print(f"Loading latents: {latent_path}")
    latents = torch.load(latent_path, map_location="cpu").float()
    image_ids = torch.load(image_id_path, map_location="cpu")
    if latents.ndim != 3 or latents.shape[1:] != (256, 1664):
        raise ValueError(f"Expected latents (N, 256, 1664), got {tuple(latents.shape)}")

    n_total = latents.shape[0]
    if args.num_images > 0:
        n_total = min(n_total, args.num_images)
        latents = latents[:n_total]
        image_ids = image_ids[:n_total]
    low_level_latents = None
    if args.low_level_latents:
        low_payload = torch.load(Path(args.low_level_latents), map_location="cpu")
        if not isinstance(low_payload, dict) or "latent" not in low_payload or "image_id" not in low_payload:
            raise ValueError("Low-level latent file must contain latent and image_id")
        if not bool(low_payload.get("scaled", False)):
            raise ValueError("Low-level latents must be scaled SDXL latents")
        low_level_latents = low_payload["latent"][:n_total].float()
        low_image_ids = low_payload["image_id"][:n_total].reshape(-1).long()
        if tuple(low_level_latents.shape) != (n_total, 4, 96, 96):
            raise ValueError(f"Expected low-level latents {(n_total, 4, 96, 96)}, got {tuple(low_level_latents.shape)}")
        if not torch.equal(low_image_ids, image_ids.reshape(-1).long()):
            raise ValueError("Low-level latent image ids do not match UnCLIP conditioning image ids")
        print(f"Using low-level SDXL latents: {args.low_level_latents}, strength={args.img2img_strength}")
    print(f"Decoding {n_total} trials with UnCLIP, latent_source={args.latent_source}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        output_name = args.latent_source
        if low_level_latents is not None:
            strength_tag = str(args.img2img_strength).replace(".", "p")
            output_name = f"{args.latent_source}_low_s{strength_tag}"
        out_dir = paths.output_root / "synthesised_decoding_unclip" / f"subj{args.subject:02d}" / output_name
    images_dir = out_dir / "images"
    thumbs_dir = out_dir / "thumbnails"

    if args.dry_run:
        print("[dry-run] UnCLIP decode pipeline ready; skipping model load and sampling.")
        print(f"Would write {n_total} images to {images_dir}")
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    if args.save_thumbnails:
        thumbs_dir.mkdir(parents=True, exist_ok=True)

    engine, vector_suffix, offset_noise_level = load_sdxl_unclip(
        Path(paths.generative_models_path),
        Path(args.unclip_ckpt),
        Path(args.openclip_path),
        device,
        args.num_steps,
    )

    manifest = []
    written = 0
    for start in range(0, n_total, args.batch_size):
        stop = min(start + args.batch_size, n_total)
        batch_tokens = latents[start:stop]
        batch_low_level = low_level_latents[start:stop] if low_level_latents is not None else None
        samples = unclip_decode_batch(
            batch_tokens,
            engine,
            vector_suffix,
            offset_noise_level,
            init_latent=batch_low_level,
            img2img_strength=args.img2img_strength,
            seed=args.seed + start,
        ).cpu()

        ids = image_ids[start:stop].tolist()
        for offset, image_id in enumerate(ids):
            trial_idx = start + offset
            stem = f"trial{trial_idx:05d}_img{int(image_id):05d}"
            png_path = images_dir / f"{stem}.png"
            save_image(samples[offset], png_path)
            if args.save_thumbnails:
                thumb = torch.nn.functional.interpolate(
                    samples[offset : offset + 1], size=(256, 256), mode="bilinear", align_corners=False
                )
                save_image(thumb[0], thumbs_dir / f"{stem}.png")
            manifest.append({"trial_index": trial_idx, "image_id": int(image_id), "path": str(png_path.relative_to(out_dir))})
            written += 1
        print(f"  decoded {written}/{n_total}", flush=True)

    with (out_dir / "manifest.json").open("w") as fh:
        json.dump(
            {
                "subject": args.subject,
                "latent_source": args.latent_source,
                "decoder": "sdxl_unclip",
                "num_steps": args.num_steps,
                "low_level_latents": args.low_level_latents or None,
                "img2img_strength": args.img2img_strength if low_level_latents is not None else None,
                "n_trials": written,
                "items": manifest,
            },
            fh,
            indent=2,
        )
    print(f"Saved {written} decoded images to {images_dir}")


if __name__ == "__main__":
    main()
