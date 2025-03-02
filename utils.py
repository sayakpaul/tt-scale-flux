import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers import FluxPipeline, LTXPipeline
import base64
import re
import hashlib
from typing import Dict
import json
from typing import Union
from PIL import Image
import requests
import argparse
import io
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from diffusers.utils import export_to_video


TORCH_DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
MODEL_NAME_MAP = {
    "black-forest-labs/FLUX.1-dev": "flux.1-dev",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": "pixart-sigma-1024-ms",
    "stabilityai/stable-diffusion-xl-base-1.0": "sdxl-base",
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "sd-v1.5",
}
MANDATORY_CONFIG_KEYS = [
    "pretrained_model_name_or_path",
    "torch_dtype",
    "pipeline_call_args",
    "verifier_args",
    "search_args",
]


def parse_cli_args():
    """
    Parse and return CLI arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_config_path",
        type=str,
        default="configs/flux.1_dev.json",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Use your own prompt.")
    parser.add_argument(
        "--num_prompts",
        type=lambda x: None if x.lower() == "none" else x if x.lower() == "all" else int(x),
        default=2,
        help="Number of prompts to use (or 'all' to use all prompts from file).",
    )
    parser.add_argument(
        "--batch_size_for_img_gen",
        type=int,
        default=1,
        help="Controls the batch size of noises during image generation. Increasing it reduces the total time at the cost of more memory.",
    )
    parser.add_argument(
        "--use_low_gpu_vram",
        action="store_true",
        help="Flag to use low GPU VRAM mode (moves models between cpu and cuda as needed). Ignored when using Gemini.",
    )

    args = parser.parse_args()

    validate_args(args)
    return args


def validate_args(args):
    if args.prompt and args.num_prompts:
        raise ValueError("Both `prompt` and `num_prompts` cannot be specified.")
    if not args.prompt and not args.num_prompts:
        raise ValueError("Both `prompt` and `num_prompts` cannot be None.")

    with open(args.pipeline_config_path, "r") as f:
        config = json.load(f)

    config_keys = list(config.keys())
    assert all(element in config_keys for element in MANDATORY_CONFIG_KEYS), (
        f"Expected the following keys to be present: {MANDATORY_CONFIG_KEYS} but got: {config_keys}."
    )

    _validate_verifier_args(config)
    _validate_search_args(config)


def _validate_verifier_args(config):
    from verifiers import SUPPORTED_VERIFIERS, SUPPORTED_METRICS

    verifier_args = config["verifier_args"]
    supported_verifiers = list(SUPPORTED_VERIFIERS.keys())
    verifier = verifier_args["name"]
    assert verifier in supported_verifiers, (
        f"Unknown verifier provided: {verifier}, supported ones are: {supported_metrics}."
    )

    supported_metrics = SUPPORTED_METRICS[verifier_args["name"]]
    choice_of_metric = verifier_args["choice_of_metric"]
    assert choice_of_metric in supported_metrics, (
        f"Unsupported metric provided: {choice_of_metric}, supported ones are: {supported_metrics}."
    )


def _validate_search_args(config):
    search_args = config["search_args"]
    search_method = search_args["search_method"]
    supported_search_methods = ["random", "zero-order"]

    assert search_method in supported_search_methods, (
        f"Unsupported search method provided: {search_method}, supported ones are: {supported_search_methods}."
    )


# Adapted from Diffusers.
def prepare_latents_for_flux(
    batch_size: int,
    height: int,
    width: int,
    generator: torch.Generator,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_latent_channels = 16
    vae_scale_factor = 8

    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    shape = (batch_size, num_latent_channels, height, width)
    latents = randn_tensor(shape, generator=generator, device=torch.device(device), dtype=dtype)
    latents = FluxPipeline._pack_latents(latents, batch_size, num_latent_channels, height, width)
    return latents


# Adapted from Diffusers.
def prepare_latents_ltx(
    batch_size: int = 1,
    num_channels_latents: int = 128,
    height: int = 512,
    width: int = 704,
    num_frames: int = 161,
    dtype: torch.dtype = None,
    device: torch.device = None,
    generator: torch.Generator = None,
) -> torch.Tensor:
    vae_spatial_compression_ratio = 32
    vae_temporal_compression_ratio = 8
    transformer_spatial_patch_size = 1
    transformer_temporal_patch_size = 1

    height = height // vae_spatial_compression_ratio
    width = width // vae_spatial_compression_ratio
    num_frames = (num_frames - 1) // vae_temporal_compression_ratio + 1

    shape = (batch_size, num_channels_latents, num_frames, height, width)

    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latents = LTXPipeline._pack_latents(latents, transformer_spatial_patch_size, transformer_temporal_patch_size)
    return latents


# Adapted from Diffusers.
def prepare_latents(
    batch_size: int, height: int, width: int, generator: torch.Generator, device: str, dtype: torch.dtype
):
    num_channels_latents = 4
    vae_scale_factor = 8
    shape = (
        batch_size,
        num_channels_latents,
        int(height) // vae_scale_factor,
        int(width) // vae_scale_factor,
    )
    latents = randn_tensor(shape, generator=generator, device=torch.device(device), dtype=dtype)
    return latents


def get_latent_prep_fn(pretrained_model_name_or_path: str) -> callable:
    fn_map = {
        "black-forest-labs/FLUX.1-dev": prepare_latents_for_flux,
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": prepare_latents,
        "stabilityai/stable-diffusion-xl-base-1.0": prepare_latents,
        "stable-diffusion-v1-5/stable-diffusion-v1-5": prepare_latents,
        "Lightricks/LTX-Video": prepare_latents_ltx,
    }[pretrained_model_name_or_path]
    return fn_map


def get_noises(
    max_seed: int,
    num_samples: int,
    height: int,
    width: int,
    device="cuda",
    dtype: torch.dtype = torch.bfloat16,
    fn: callable = prepare_latents_for_flux,
    **kwargs,
) -> Dict[int, torch.Tensor]:
    seeds = torch.randint(0, high=max_seed, size=(num_samples,))

    noises = {}
    for noise_seed in seeds:
        latents = fn(
            batch_size=1,
            height=height,
            width=width,
            generator=torch.manual_seed(int(noise_seed)),
            device=device,
            dtype=dtype,
            **kwargs,
        )
        noises.update({int(noise_seed): latents})

    assert len(noises) == len(seeds)
    return noises


def generate_neighbors(x, threshold=0.95, num_neighbors=4):
    """Courtesy: Willis Ma"""
    rng = np.random.Generator(np.random.PCG64())
    x_f = x.flatten(1)
    x_norm = torch.linalg.norm(x_f, dim=-1, keepdim=True, dtype=torch.float64).unsqueeze(-2)
    u = x_f.unsqueeze(-2) / x_norm.clamp_min(1e-12)
    v = torch.from_numpy(rng.standard_normal(size=(u.shape[0], num_neighbors, u.shape[-1]), dtype=np.float64)).to(
        u.device
    )
    w = F.normalize(v - (v @ u.transpose(-2, -1)) * u, dim=-1)
    return (
        (x_norm * (threshold * u + np.sqrt(1 - threshold**2) * w))
        .reshape(x.shape[0], num_neighbors, *x.shape[1:])
        .to(x.dtype)
    )


def load_verifier_prompt(path: str) -> str:
    with open(path, "r") as f:
        verifier_prompt = f.read().replace('"""', "")

    return verifier_prompt


def prompt_to_filename(prompt, max_length=100):
    """Thanks ChatGPT."""
    filename = re.sub(r"[^a-zA-Z0-9]", "_", prompt.strip())
    filename = re.sub(r"_+", "_", filename)
    hash_digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    base_filename = f"prompt@{filename}_hash@{hash_digest}"

    if len(base_filename) > max_length:
        base_length = max_length - len(hash_digest) - 7
        base_filename = f"prompt@{filename[:base_length]}_hash@{hash_digest}"

    return base_filename


def load_image(path_or_url: Union[str, Image.Image]) -> Image.Image:
    """
    Load an image from a local path or a URL and return a PIL Image object.

    `path_or_url` is returned as is if it's an `Image` already.
    """
    if isinstance(path_or_url, Image.Image):
        return path_or_url
    elif path_or_url.startswith("http"):
        response = requests.get(path_or_url, stream=True)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    return Image.open(path_or_url)


def convert_to_bytes(path_or_url: Union[str, Image.Image], b64_encode: bool = False) -> bytes:
    """Load an image from a path or URL and convert it to bytes."""
    image = load_image(path_or_url).convert("RGB")
    image_bytes_io = io.BytesIO()
    image.save(image_bytes_io, format="PNG")
    image_bytes = image_bytes_io.getvalue()
    if not b64_encode:
        return image_bytes
    else:
        return base64.b64encode(image_bytes).decode("utf-8")


def recover_json_from_output(output: str):
    start = output.find("{")
    end = output.rfind("}") + 1
    json_part = output[start:end]
    return json.loads(json_part)


def get_key_frames(path: Union[Path, str]) -> list[Image.Image]:
    import av

    frames = []
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = "NONKEY"
    for _, frame in enumerate(container.decode(stream)):
        frames.append(frame.to_image())
    container.close()
    return frames


def prepare_video_frames(vid_path):
    vid_frames = get_key_frames(vid_path)
    first = vid_frames[0]
    mid = None
    last = None
    if len(vid_frames) == 2:
        last = vid_frames[1]
    elif len(vid_frames) > 2:
        mid = vid_frames[len(vid_frames) // 2]
        last = vid_frames[-1]
    frames = []
    for idx, frame in enumerate([first, mid, last]):
        if frame is None:
            continue
        frames.append(frame)

    return frames


def serialize_artifacts(
    images_info: list[tuple[int, torch.Tensor, Image.Image, str]],
    prompt: str,
    search_round: int,
    root_dir: str,
    datapoint: dict,
    **kwargs,
) -> None:
    """
    Serialize generated images and the best datapoint JSON configuration.
    """
    # Save each image.
    for seed, noise, image, filename in images_info:
        if filename.endswith(".png"):
            image.save(filename)
        elif filename.endswith(".mp4"):
            export_to_video(image, filename, fps=kwargs.get("fps", 24))

    # Save the best datapoint config as a JSON file.
    best_json_filename = datapoint["best_img_path"].replace(".png", ".json")
    with open(best_json_filename, "w") as f:
        # Remove the noise tensor (or any non-serializable object) from the JSON.
        datapoint_copy = datapoint.copy()
        datapoint_copy.pop("best_noise", None)
        json.dump(datapoint_copy, f, indent=4)
    print(f"Serialized JSON configuration and images to {root_dir}.")
