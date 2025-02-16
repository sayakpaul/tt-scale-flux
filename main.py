import os
import json
from datetime import datetime

import numpy as np
import torch
from diffusers import DiffusionPipeline
from tqdm.auto import tqdm
import copy

from utils import prompt_to_filename, get_noises, TORCH_DTYPE_MAP, get_latent_prep_fn, parse_cli_args, MODEL_NAME_MAP

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
    For a given prompt, generate images using all provided noises,
    score them with the verifier, and select the top-K noise.
    The images and JSON artifacts are saved under `root_dir`.
    """
    config_cp = copy.deepcopy(config)
    max_new_tokens = config_cp.pop("max_new_tokens", None)
    choice_of_metric = config_cp.pop("choice_of_metric", None)
    verifier_to_use = config_cp.pop("verifier_to_use", "gemini")
    use_low_gpu_vram = config_cp.pop("use_low_gpu_vram", False)

    images_for_prompt = []
    noises_used = []
    seeds_used = []
    prompt_filename = prompt_to_filename(prompt)

    for i, (seed, noise) in enumerate(noises.items()):
        # Build the output filename inside the provided root directory.
        filename = os.path.join(root_dir, f"{prompt_filename}_i@{search_round}_s@{seed}.png")

        # If using low GPU VRAM (and not Gemini) move the pipeline to cuda before generating.
        if use_low_gpu_vram and verifier_to_use != "gemini":
            pipe = pipe.to("cuda:0")
        print(f"Generating images.")
        image = pipe(prompt=prompt, latents=noise, **config_cp).images[0]
        if use_low_gpu_vram and verifier_to_use != "gemini":
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
        use_low_gpu_vram=use_low_gpu_vram,  # Ignored when using Gemini.
    )
    print("Scoring with the verifier.")
    outputs = verifier.score(
        inputs=verifier_inputs,
        max_new_tokens=max_new_tokens,  # Ignored when using Gemini for now.
    )
    for o in outputs:
        assert choice_of_metric in o, o.keys()

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
        assert choice_of_metric in x, (
            f"Expected all dicts in `results` to contain the " f"`{choice_of_metric}` key; got {x.keys()}."
        )

    def f(x):
        if isinstance(x[choice_of_metric], dict):
            return x[choice_of_metric]["score"]
        return x[choice_of_metric]

    sorted_list = sorted(results, key=lambda x: f(x), reverse=True)
    topk_scores = sorted_list[:topk]

    # Print debug information.
    for ts in topk_scores:
        print(f"Prompt='{prompt}' | Best seed={ts['seed']} | " f"Score={ts[choice_of_metric]}")

    best_img_path = os.path.join(root_dir, f"{prompt_filename}_i@{search_round}_s@{topk_scores[0]['seed']}.png")
    datapoint = {
        "prompt": prompt,
        "search_round": search_round,
        "num_noises": len(noises),
        "best_noise_seed": topk_scores[0]["seed"],
        "best_score": topk_scores[0][choice_of_metric],
        "choice_of_metric": choice_of_metric,
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
        "max_new_tokens": args.max_new_tokens,
        "use_low_gpu_vram": args.use_low_gpu_vram,
        "choice_of_metric": args.choice_of_metric,
        "verifier_to_use": args.verifier_to_use,
    }
    with open(args.pipeline_config_path, "r") as f:
        config.update(json.load(f))

    search_rounds = args.search_rounds
    num_prompts = args.num_prompts

    # Create a root output directory: output/{verifier_to_use}/{current_datetime}
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_name = config.pop("pretrained_model_name_or_path")
    root_dir = os.path.join(
        "output",
        MODEL_NAME_MAP[pipeline_name],
        config["verifier_to_use"],
        config["choice_of_metric"],
        current_datetime,
    )
    os.makedirs(root_dir, exist_ok=True)
    print(f"Artifacts will be saved to: {root_dir}")
    with open(os.path.join(root_dir, "config.json"), "w") as f:
        config_cp = copy.deepcopy(config)
        config_cp.update(vars(args))
        json.dump(config_cp, f)

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
    torch_dtype = TORCH_DTYPE_MAP[config.pop("torch_dtype")]
    pipe = DiffusionPipeline.from_pretrained(pipeline_name, torch_dtype=torch_dtype)
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
                num_samples=num_noises_to_sample,
                height=config["height"],
                width=config["width"],
                dtype=torch_dtype,
                fn=get_latent_prep_fn(pipeline_name),
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
