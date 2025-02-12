import numpy as np
import torch
from diffusers import FluxPipeline
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
import json
from typing import Tuple

from verifier import load_verifier, prepare_inputs, perform_inference
from utils import load_verifier_prompt, prompt_to_filename, get_noises, recover_json_from_output

# Constants
SEARCH_ROUNDS = 4  # Number of search rounds to scale the initial noise pool (2, 4, 8, 16, etc.)
NUM_PROMPTS = 2  # Number of prompts to use for experiments. Change to "all" to use all prompts.
HEIGHT, WIDTH = 1024, 1024
NUM_LATENT_CHANNELS = 16
VAE_SCALE_FACTOR = 8
MAX_SEED = np.iinfo(np.int32).max  # To generate random seeds
MAX_NEW_TOKENS = 300  # Maximum number of tokens the verifier can use
TOPK = 1  # Number of maximum noise(s) to start  the search with
USE_LOW_GPU_VRAM = False  # When using a single GPU set this to True.
CHOICE_OF_METRIC = "overall_score" # metric to use from the LLM grading.
# available options: `accuracy_to_prompt`, `creativity_and_originality`, 
# `visual_quality_and_realism`, `consistency_and_cohesion`, `emotional_or_thematic_resonance`,
# `overall_score`.

def sample(
    noises: dict[int, torch.Tensor],
    prompt: str,
    verifier_prompt: str,
    search_round: int,
    pipe: FluxPipeline,
    verifier: AutoModel,
    processor: AutoTokenizer,
    topk: int,
) -> Tuple[str, dict]:
    images_for_prompt = []
    noises_used = []
    seeds_used = []
    prompt_filename = prompt_to_filename(prompt)

    for i, (seed, noise) in enumerate(noises.items()):
        filename = f"{prompt_filename}_i@{search_round}_s@{seed}.png"

        if USE_LOW_GPU_VRAM:
            pipe = pipe.to("cuda")
        image = pipe(
            prompt=prompt,
            latents=noise,
            height=HEIGHT,
            width=WIDTH,
            max_sequence_length=512,
            guidance_scale=3.5,
            num_inference_steps=50,
        ).images[0]
        if USE_LOW_GPU_VRAM:
            pipe = pipe.to("cpu")

        images_for_prompt.append(image)
        noises_used.append(noise)
        seeds_used.append(seed)

        # Save the intermediate image
        image.save(filename)

    # Prepare verifier inputs and perform inference
    verifier_inputs = prepare_inputs(
        system_prompt=verifier_prompt,
        images=images_for_prompt,
        prompts=[prompt] * len(images_for_prompt),
        processor=processor,
        use_low_gpu_vram=USE_LOW_GPU_VRAM,
    )
    outputs = perform_inference(
        model=verifier,
        processor=processor,
        inputs=verifier_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # Convert raw output to JSON and attach noise
    outputs = [recover_json_from_output(o) for o in outputs]
    for o in outputs:
        assert CHOICE_OF_METRIC in o, o.keys()

    assert (
        len(outputs) == len(images_for_prompt)
    ), f"Expected len(outputs) to be same as len(images_for_prompt) but got {len(outputs)=} & {len(images_for_prompt)=}"

    results = []
    for json_dict, seed_val, noise in zip(outputs, seeds_used, noises_used):
        # Attach the noise tensor so we can select top-K
        merged = {**json_dict, "noise": noise, "seed": seed_val}
        results.append(merged)

    # Sort by 'overall_score' descending and pick top-K
    for x in results:
        assert CHOICE_OF_METRIC in x, f"Expected all dicts in `results` to contain the `overall_score` key {x.keys()=}."

    sorted_list = sorted(results, key=lambda x: x[CHOICE_OF_METRIC], reverse=True)
    topk_scores = sorted_list[:topk]

    # Update `starting_noises` with the new top-K so next iteration continues the search
    new_noises = {item["seed"]: item["noise"] for item in topk_scores}

    # Print some debug info
    for ts in topk_scores:
        print(f"Prompt='{prompt}' | Best seed={ts['seed']} | Score={ts[CHOICE_OF_METRIC]}")

    datapoint = {
        "prompt": prompt,
        "search_round": search_round,
        "num_noises": len(noises),
        "best_noise_seed": list(new_noises.keys())[0],
        "best_score": topk_scores[0][CHOICE_OF_METRIC],
        "choice_of_metric": CHOICE_OF_METRIC,
    }
    return filename, datapoint


@torch.no_grad()
def main():
    """
    - Samples a pool of random noises.
    - For each text prompt:
      - Generates candidate images with each noise.
      - Passes them through the 'verifier' model to get scores.
      - Saves the top-K noise(s) and updates 'starting_noises' so the search continues.
      - Saves the final, best image for each prompt.
    """
    # Load system prompt and text prompts
    verifier_prompt = load_verifier_prompt("verifier_prompt.txt")
    with open("prompts_open_image_pref_v1.txt", "r") as f:
        prompts = [line.strip() for line in f.readlines()]
        if isinstance(NUM_PROMPTS, int):
            prompts = prompts[:NUM_PROMPTS]

    print(f"Using {len(prompts)} prompt(s).")

    # Set up the image-generation pipeline (on the first GPU)
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    if not USE_LOW_GPU_VRAM:
        pipe = pipe.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)

    # Load the verifier model and its processor (on the second GPU)
    verifier, processor = load_verifier(use_low_gpu_vram=USE_LOW_GPU_VRAM)

    # Main loop: Generate images, verify, and update noise set
    for round in range(1, SEARCH_ROUNDS + 1):
        print(f"Round: {round}")

        num_noises_to_sample = 2**round
        for prompt in tqdm(prompts, desc="Sampling ..."):
            noises = get_noises(
                max_seed=MAX_SEED,
                height=HEIGHT,
                width=WIDTH,
                num_latent_channels=NUM_LATENT_CHANNELS,
                vae_scale_factor=VAE_SCALE_FACTOR,
                num_samples=num_noises_to_sample,
            )

            print(f"{len(noises)=}")
            filename, datapoint_for_current_round = sample(
                noises=noises,
                prompt=prompt,
                verifier_prompt=verifier_prompt,
                search_round=round,
                pipe=pipe,
                verifier=verifier,
                processor=processor,
                topk=TOPK,
            )
            with open(filename.replace(".png", ".json"), "w") as f:
                json.dump(datapoint_for_current_round, f)


if __name__ == "__main__":
    main()
