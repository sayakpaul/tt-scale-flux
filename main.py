import argparse
import os
import json
from datetime import datetime

import numpy as np
import torch
from diffusers import FluxPipeline
from tqdm.auto import tqdm

from utils import prompt_to_filename, get_noises

# Non-configurable constants
NUM_LATENT_CHANNELS = 16
VAE_SCALE_FACTOR = 8
TOPK = 1  # Always selecting the top-1 noise for the next round
MAX_SEED = np.iinfo(np.int32).max  # To generate random seeds


def parse_cli_args():
    """
    Parse and return CLI arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search_rounds",
        type=int,
        default=4,
        help="Number of search rounds (each round scales the number of noise samples).",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Use your own prompt.")
    parser.add_argument(
        "--num_prompts",
        type=lambda x: None if x.lower() == "none" else x if x.lower() == "all" else int(x),
        default=2,
        help="Number of prompts to use (or 'all' to use all prompts from file).",
    )
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated images.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated images.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=300,
        help="Maximum number of tokens for the verifier. Ignored when using Gemini.",
    )
    parser.add_argument(
        "--use_low_gpu_vram",
        action="store_true",
        help="Flag to use low GPU VRAM mode (moves models between cpu and cuda as needed). Ignored when using Gemini.",
    )
    parser.add_argument(
        "--choice_of_metric",
        type=str,
        default="overall_score",
        choices=[
            "accuracy_to_prompt",
            "creativity_and_originality",
            "visual_quality_and_realism",
            "consistency_and_cohesion",
            "emotional_or_thematic_resonance",
            "overall_score",
        ],
        help="Metric to use from the LLM grading.",
    )
    parser.add_argument(
        "--verifier_to_use",
        type=str,
        default="gemini",
        choices=["gemini", "qwen"],
        help="Verifier to use; must be one of 'gemini' or 'qwen'.",
    )
    args = parser.parse_args()

    if args.prompt and args.num_prompts:
        raise ValueError("Both `prompt` and `num_prompts` cannot be specified.")
    if not args.prompt and not args.num_prompts:
        raise ValueError("Both `prompt` and `num_prompts` cannot be None.")
    return args


def sample(
    noises: dict[int, torch.Tensor],
    prompt: str,
    search_round: int,
    pipe: FluxPipeline,
    verifier,
    topk: int,
    root_dir: str,
    config: dict,
) -> dict:
    """
    For a given prompt, generate images using all provided noises,
    score them with the verifier, and select the top-K noise.
    The images and JSON artifacts are saved under `root_dir`.
    """
    images_for_prompt = []
    noises_used = []
    seeds_used = []
    prompt_filename = prompt_to_filename(prompt)

    for i, (seed, noise) in enumerate(noises.items()):
        # Build the output filename inside the provided root directory.
        filename = os.path.join(root_dir, f"{prompt_filename}_i@{search_round}_s@{seed}.png")

        # If using low GPU VRAM (and not Gemini) move the pipeline to cuda before generating.
        if config["use_low_gpu_vram"] and config["verifier_to_use"] != "gemini":
            pipe = pipe.to("cuda:0")
        print(f"Generating images.")
        image = pipe(
            prompt=prompt,
            latents=noise,
            height=config["height"],
            width=config["width"],
            max_sequence_length=512,
            guidance_scale=3.5,
            num_inference_steps=50,
        ).images[0]
        if config["use_low_gpu_vram"] and config["verifier_to_use"] != "gemini":
            pipe = pipe.to("cpu")

        images_for_prompt.append(image)
        noises_used.append(noise)
        seeds_used.append(seed)

        # Save the intermediate image to the output folder.
        image.save(filename)

    # Prepare verifier inputs and perform inference.
    verifier_inputs = verifier.prepare_inputs(
        images=images_for_prompt,
        prompts=[prompt] * len(images_for_prompt),
        use_low_gpu_vram=config["use_low_gpu_vram"],  # Ignored when using Gemini.
    )
    print("Scoring with the verifier.")
    outputs = verifier.score(
        inputs=verifier_inputs,
        max_new_tokens=config["max_new_tokens"],  # Ignored when using Gemini for now.
    )
    for o in outputs:
        assert config["choice_of_metric"] in o, o.keys()

    assert (
        len(outputs) == len(images_for_prompt)
    ), f"Expected len(outputs) to be same as len(images_for_prompt) but got {len(outputs)=} & {len(images_for_prompt)=}"

    results = []
    for json_dict, seed_val, noise in zip(outputs, seeds_used, noises_used):
        # Attach the noise tensor so we can select top-K
        merged = {**json_dict, "noise": noise, "seed": seed_val}
        results.append(merged)

    # Sort by the chosen metric descending and pick top-K.
    for x in results:
        assert config["choice_of_metric"] in x, (
            f"Expected all dicts in `results` to contain the " f"`{config['choice_of_metric']}` key; got {x.keys()}."
        )

    def f(x):
        if isinstance(x[config["choice_of_metric"]], dict):
            return x[config["choice_of_metric"]]["score"]
        return x[config["choice_of_metric"]]

    sorted_list = sorted(results, key=lambda x: f(x), reverse=True)
    topk_scores = sorted_list[:topk]

    # Print debug information.
    for ts in topk_scores:
        print(f"Prompt='{prompt}' | Best seed={ts['seed']} | " f"Score={ts[config['choice_of_metric']]}")

    best_img_path = os.path.join(root_dir, f"{prompt_filename}_i@{search_round}_s@{topk_scores[0]['seed']}.png")
    datapoint = {
        "prompt": prompt,
        "search_round": search_round,
        "num_noises": len(noises),
        "best_noise_seed": topk_scores[0]["seed"],
        "best_score": topk_scores[0][config["choice_of_metric"]],
        "choice_of_metric": config["choice_of_metric"],
        "best_img_path": best_img_path,
    }
    # Save the best config JSON file alongside the images.
    best_json_filename = best_img_path.replace(".png", ".json")
    with open(best_json_filename, "w") as f:
        json.dump(datapoint, f, indent=4)
    return datapoint


@torch.no_grad()
def main():
    """
    Main function:
      - Parses CLI arguments.
      - Creates an output directory based on verifier and current datetime.
      - Loads prompts.
      - Loads the image-generation pipeline.
      - Loads the verifier model.
      - Runs several search rounds where for each prompt a pool of random noises is generated,
        candidate images are produced and verified, and the best noise is chosen.
    """
    args = parse_cli_args()

    # Build a config dictionary for parameters that need to be passed around.
    config = {
        "height": args.height,
        "width": args.width,
        "max_new_tokens": args.max_new_tokens,
        "use_low_gpu_vram": args.use_low_gpu_vram,
        "choice_of_metric": args.choice_of_metric,
        "verifier_to_use": args.verifier_to_use,
    }

    search_rounds = args.search_rounds
    num_prompts = args.num_prompts

    # Create a root output directory: output/{verifier_to_use}/{current_datetime}
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join("output", config["verifier_to_use"], config["choice_of_metric"], current_datetime)
    os.makedirs(root_dir, exist_ok=True)
    print(f"Artifacts will be saved to: {root_dir}")

    # Load prompts from file.
    if args.prompt is None:
        with open("prompts_open_image_pref_v1.txt", "r") as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        if num_prompts != "all":
            prompts = prompts[:num_prompts]
        print(f"Using {len(prompts)} prompt(s).")
    else:
        prompts = [args.prompt]

    # Set up the image-generation pipeline (on the first GPU if available).
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    if not config["use_low_gpu_vram"]:
        pipe = pipe.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)

    # Load the verifier model.
    if config["verifier_to_use"] == "gemini":
        from verifiers.gemini_verifier import GeminiVerifier

        verifier = GeminiVerifier()
    else:
        from verifiers.qwen_verifier import QwenVerifier

        verifier = QwenVerifier(use_low_gpu_vram=config["use_low_gpu_vram"])

    # Main loop: For each search round and each prompt, generate images, verify, and save artifacts.
    for round in range(1, search_rounds + 1):
        print(f"\n=== Round: {round} ===")
        num_noises_to_sample = 2**round  # scale noise pool.
        for prompt in tqdm(prompts, desc="Sampling prompts"):
            noises = get_noises(
                max_seed=MAX_SEED,
                height=config["height"],
                width=config["width"],
                num_latent_channels=NUM_LATENT_CHANNELS,
                vae_scale_factor=VAE_SCALE_FACTOR,
                num_samples=num_noises_to_sample,
            )
            print(f"Number of noise samples: {len(noises)}")
            datapoint_for_current_round = sample(
                noises=noises,
                prompt=prompt,
                search_round=round,
                pipe=pipe,
                verifier=verifier,
                topk=TOPK,
                root_dir=root_dir,
                config=config,
            )


if __name__ == "__main__":
    main()
