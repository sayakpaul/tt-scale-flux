import os
import json
from datetime import datetime

import numpy as np
import torch
from diffusers import DiffusionPipeline
from tqdm.auto import tqdm

from utils import (
    generate_neighbors,
    prompt_to_filename,
    get_noises,
    TORCH_DTYPE_MAP,
    get_latent_prep_fn,
    parse_cli_args,
    MODEL_NAME_MAP,
)

# Non-configurable constants
TOPK = 1  # Always selecting the top-1 noise for the next round
MAX_SEED = np.iinfo(np.int32).max  # To generate random seeds


def sample(
    noises: dict[int, torch.Tensor],
    prompt: str,
    search_round: int,
    pipe: DiffusionPipeline,
    verifier,
    topk: int,
    root_dir: str,
    config: dict,
) -> dict:
    """
    For a given prompt, generate images using all provided noises in batches,
    score them with the verifier, and select the top-K noise.
    The images and JSON artifacts are saved under `root_dir`.
    """
    use_low_gpu_vram = config.get("use_low_gpu_vram", False)
    batch_size_for_img_gen = config.get("batch_size_for_img_gen", 1)
    verifier_args = config.get("verifier_args")
    max_new_tokens = verifier_args.get("max_new_tokens", None)
    choice_of_metric = verifier_args.get("choice_of_metric", None)
    verifier_to_use = verifier_args.get("name", "gemini")
    search_args = config.get("search_args", None)

    images_for_prompt = []
    noises_used = []
    seeds_used = []
    prompt_filename = prompt_to_filename(prompt)

    # Convert the noises dictionary into a list of (seed, noise) tuples.
    noise_items = list(noises.items())

    # Process the noises in batches.
    for i in range(0, len(noise_items), batch_size_for_img_gen):
        batch = noise_items[i : i + batch_size_for_img_gen]
        seeds_batch, noises_batch = zip(*batch)
        filenames_batch = [
            os.path.join(root_dir, f"{prompt_filename}_i@{search_round}_s@{seed}.png") for seed in seeds_batch
        ]

        if use_low_gpu_vram and verifier_to_use != "gemini":
            pipe = pipe.to("cuda:0")
        print(f"Generating images for batch with seeds: {[s for s in seeds_batch]}.")

        # Create a batched prompt list and stack the latents.
        batched_prompts = [prompt] * len(noises_batch)
        batched_latents = torch.stack(noises_batch).squeeze(dim=1)

        batch_result = pipe(prompt=batched_prompts, latents=batched_latents, **config["pipeline_call_args"])
        batch_images = batch_result.images
        if use_low_gpu_vram and verifier_to_use != "gemini":
            pipe = pipe.to("cpu")

        # Iterate over the batch and save the images.
        for seed, noise, image, filename in zip(seeds_batch, noises_batch, batch_images, filenames_batch):
            images_for_prompt.append(image)
            noises_used.append(noise)
            seeds_used.append(seed)
            image.save(filename)

    # Prepare verifier inputs and perform inference.
    verifier_inputs = verifier.prepare_inputs(images=images_for_prompt, prompts=[prompt] * len(images_for_prompt))
    print("Scoring with the verifier.")
    outputs = verifier.score(
        inputs=verifier_inputs,
        max_new_tokens=max_new_tokens,  # Ignored when using Gemini for now.
    )
    for o in outputs:
        assert choice_of_metric in o, o.keys()

    assert len(outputs) == len(images_for_prompt), (
        f"Expected len(outputs) to be same as len(images_for_prompt) but got {len(outputs)=} & {len(images_for_prompt)=}"
    )

    results = []
    for json_dict, seed_val, noise in zip(outputs, seeds_used, noises_used):
        # Attach the noise tensor so we can select top-K.
        merged = {**json_dict, "noise": noise, "seed": seed_val}
        results.append(merged)

    # Sort by the chosen metric descending and pick top-K.
    for x in results:
        assert choice_of_metric in x, (
            f"Expected all dicts in `results` to contain the `{choice_of_metric}` key; got {x.keys()}."
        )

    def f(x):
        if isinstance(x[choice_of_metric], dict):
            return x[choice_of_metric]["score"]
        return x[choice_of_metric]

    sorted_list = sorted(results, key=lambda x: f(x), reverse=True)
    topk_scores = sorted_list[:topk]

    # Print debug information.
    for ts in topk_scores:
        print(f"Prompt='{prompt}' | Best seed={ts['seed']} | Score={ts[choice_of_metric]}")

    best_img_path = os.path.join(root_dir, f"{prompt_filename}_i@{search_round}_s@{topk_scores[0]['seed']}.png")
    datapoint = {
        "prompt": prompt,
        "search_round": search_round,
        "num_noises": len(noises),
        "best_noise_seed": topk_scores[0]["seed"],
        "best_noise": topk_scores[0]["noise"],
        "best_score": topk_scores[0][choice_of_metric],
        "choice_of_metric": choice_of_metric,
        "best_img_path": best_img_path,
    }

    # Check if the neighbors have any improvements.
    if search_args and search_args.get("search_method") == "zero-order":
        # `first_score` corresponds to the base noise.
        first_score = f(results[0])
        neighbors_with_better_score = any(f(item) > first_score for item in results[1:])
        if not neighbors_with_better_score:
            datapoint["neighbors_no_improvement"] = True
        else:
            datapoint["neighbors_no_improvement"] = False

    # Save the best config JSON file alongside the images.
    best_json_filename = best_img_path.replace(".png", ".json")
    with open(best_json_filename, "w") as f:
        datapoint_cp = datapoint.copy()
        datapoint_cp.pop("best_noise")
        json.dump(datapoint_cp, f, indent=4)

    return datapoint


@torch.no_grad()
def main():
    # === Load configuration and CLI arguments ===
    args = parse_cli_args()
    with open(args.pipeline_config_path, "r") as f:
        config = json.load(f)
    config.update(vars(args))

    search_args = config["search_args"]
    search_rounds = search_args["search_rounds"]
    num_prompts = config["num_prompts"]

    # === Create output directory ===
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_name = config.pop("pretrained_model_name_or_path")
    verifier_name = config["verifier_args"]["name"]
    choice_of_metric = config["verifier_args"]["choice_of_metric"]
    output_dir = os.path.join(
        "output",
        MODEL_NAME_MAP[pipeline_name],
        verifier_name,
        choice_of_metric,
        current_datetime,
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Artifacts will be saved to: {output_dir}")
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # === Load prompts ===
    if args.prompt is None:
        with open("prompts_open_image_pref_v1.txt", "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        if num_prompts != "all":
            prompts = prompts[:num_prompts]
    else:
        prompts = [args.prompt]
    print(f"Using {len(prompts)} prompt(s).")

    # === Set up the image-generation pipeline ===
    torch_dtype = TORCH_DTYPE_MAP[config.pop("torch_dtype")]
    pipe = DiffusionPipeline.from_pretrained(pipeline_name, torch_dtype=torch_dtype)
    if not config.get("use_low_gpu_vram", False):
        pipe = pipe.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)

    # === Load verifier model ===
    verifier_args = config["verifier_args"]
    if verifier_args["name"] == "gemini":
        from verifiers import GeminiVerifier

        verifier = GeminiVerifier()
    else:
        from verifiers.qwen_verifier import QwenVerifier

        verifier = QwenVerifier(use_low_gpu_vram=config.get("use_low_gpu_vram", False))

    # For zero-order search, we store the best datapoint per prompt.
    best_datapoint_for_prompt = {}

    # === Main loop: For each search round and each prompt ===
    search_round = 1
    search_method = search_args.get("search_method", "random")
    tolerance_count = 0  # Only used for zero-order

    while search_round <= search_rounds:
        # Determine the number of noise samples.
        if search_method == "zero-order":
            num_noises_to_sample = 1
        else:
            num_noises_to_sample = 2**search_round

        print(f"\n=== Round: {search_round} (tolerance_count: {tolerance_count}) ===")

        # Track if any prompt improved in this round.
        round_improved = False

        for prompt in tqdm(prompts, desc="Sampling prompts"):
            # --- Generate noise pool ---
            if search_method != "zero-order" or search_round == 1:
                # Standard noise sampling
                noises = get_noises(
                    max_seed=MAX_SEED,
                    num_samples=num_noises_to_sample,
                    height=config["pipeline_call_args"]["height"],
                    width=config["pipeline_call_args"]["width"],
                    dtype=torch_dtype,
                    fn=get_latent_prep_fn(pipeline_name),
                )
            else:
                # For subsequent rounds in zero-order: use best noise from previous round.
                prev_dp = best_datapoint_for_prompt[prompt]
                noises = {int(prev_dp["best_noise_seed"]): prev_dp["best_noise"]}

            if search_method == "zero-order":
                # Process the noise to generate neighbors.
                base_seed, base_noise = next(iter(noises.items()))
                neighbors = generate_neighbors(
                    base_noise, threshold=search_args["threshold"], num_neighbors=search_args["num_neighbors"]
                ).squeeze(0)
                # Concatenate the base noise with its neighbors.
                neighbors_and_noise = torch.cat([base_noise, neighbors], dim=0)
                new_noises = {}
                for i, noise_tensor in enumerate(neighbors_and_noise):
                    new_noises[base_seed + i] = noise_tensor.unsqueeze(0)
                noises = new_noises

            print(f"Number of noise samples for prompt '{prompt}': {len(noises)}")

            # --- Sampling, verifying, and saving artifacts ---
            datapoint = sample(
                noises=noises,
                prompt=prompt,
                search_round=search_round,
                pipe=pipe,
                verifier=verifier,
                topk=TOPK,
                root_dir=output_dir,
                config=config,
            )

            if search_method == "zero-order":
                # Update the best noise for zero-order.
                best_datapoint_for_prompt[prompt] = datapoint

                # If there was an improvement, flag this round as improved.
                if not datapoint.get("neighbors_no_improvement", False):
                    round_improved = True

        # --- Decide on round incrementation ---
        if search_method == "zero-order":
            if round_improved:
                tolerance_count = 0
                search_round += 1
            else:
                tolerance_count += 1
                if tolerance_count >= search_args["search_round_tolerance"]:
                    tolerance_count = 0
                    search_round += 1
        else:
            search_round += 1


if __name__ == "__main__":
    main()
