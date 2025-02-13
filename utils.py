import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers import FluxPipeline
import re
import hashlib
from typing import Dict
import json
from typing import Union
from PIL import Image
import requests
import io


# Adapted from Diffusers.
def prepare_latents(
    batch_size: int,
    height: int,
    width: int,
    num_latent_channels: int,
    vae_scale_factor,
    generator: torch.Generator,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    shape = (batch_size, num_latent_channels, height, width)
    latents = randn_tensor(shape, generator=generator, device=torch.device(device), dtype=dtype)
    latents = FluxPipeline._pack_latents(latents, batch_size, num_latent_channels, height, width)
    return latents


def get_noises(
    max_seed: int,
    num_samples: int,
    height: int,
    width: int,
    num_latent_channels: int,
    vae_scale_factor: int,
    device="cuda",
    dtype=torch.bfloat16,
) -> Dict[int, torch.Tensor]:
    seeds = torch.randint(0, high=max_seed, size=(num_samples,))
    print(f"{seeds=}")
    noises = {
        int(noise_seed): prepare_latents(
            batch_size=1,
            height=height,
            width=width,
            num_latent_channels=num_latent_channels,
            vae_scale_factor=vae_scale_factor,
            generator=torch.manual_seed(int(noise_seed)),
            device=device,
            dtype=dtype,
        )
        for noise_seed in seeds
    }
    assert len(noises) == len(seeds)
    return noises


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


def convert_to_bytes(path_or_url: Union[str, Image.Image]) -> bytes:
    """Load an image from a path or URL and convert it to bytes."""
    image = load_image(path_or_url).convert("RGB")
    image_bytes_io = io.BytesIO()
    image.save(image_bytes_io, format="PNG")
    return image_bytes_io.getvalue()


def recover_json_from_output(output: str):
    start = output.find("{")
    end = output.rfind("}") + 1
    json_part = output[start:end]
    return json.loads(json_part)
