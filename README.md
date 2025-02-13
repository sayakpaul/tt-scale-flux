# tt-scale-flux

Simple re-implementation of inference-time scaling Flux.1-Dev as introduced in [Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps](https://arxiv.org/abs/2501.09732) by Ma et al. We implement the random search strategy to scale the compute budget.

<div align="center">
<img src="">
</div>

## Getting started

Make sure to install the dependencies: `pip install -r requirements`. The codebase was tested using a single H100 and two H100s (both 80GB variants).

By default, we use [Gemini 2.0 Flash](https://deepmind.google/technologies/gemini/flash/) as the verifier. This requires two things:

* `GEMINI_API_KEY` (obtain it from [here](https://ai.google.dev/gemini-api/docs)).
* `google-genai` Python [library](https://pypi.org/project/google-genai/).

Now, fire up:

```bash
GEMINI_API_KEY=... python main.py --prompt="a tiny astronaut hatching from an egg on the moon" --num_prompts=None
```

If you want to use from the [data-is-better-together/open-image-preferences-v1-binarized](https://huggingface.co/datasets/data-is-better-together/open-image-preferences-v1-binarized) dataset, you can just run:

```bash
GEMINI_API_KEY=... python main.py
```

After this is done executing, you should expect a folder named `output` with the following structure:

```
TODO
```

Each JSON file should look like so:

```json
```

To limit the number of prompts, specify `--num_prompts`. By default, we use 2 prompts. Specify "--num_prompts=all" to use all.

## Controlling the "scale"

By default, we use 4 `search_rounds` and start with a noise pool size of 2. Each search round scales up the pool size like so: `2 ** current_seach_round` (with indexing starting from 1). This is where the "scale" in inference-time scaling comes from. You can increase the compute budget by specifying a larger `search_rounds`.

For each search round, we serialize the images and best datapoint (characterized by the best eval score) in a JSON file.

For other supported CLI args, run `python main.py -h`.

## Controlling the verifier

If you don't want to use Gemini, you can use [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) as an option. Simply specify `--verifier_to_use=qwen` for this. 

> [!IMPORTANT]  
> This setup was tested on 2 H100s. If you want to do this on a single GPU, specify `--use_low_gpu_vram`.

You can also bring in your own verifier by implementing a so-called `Verifier` class following the structure of either of `GeminiVerifier` or `QwenVerifier`. You will then have to make adjustments to the following places:

* TODO
* TODO

By default, we use "overall_score" as the metric to obtain the best samples in each search round. You can change it by specifying `--choice_of_metric`. Supported values are: 

* "accuracy_to_prompt"
* "creativity_and_originality"
* "visual_quality_and_realism"
* "consistency_and_cohesion"
* "emotional_or_thematic_resonance"
* "overall_score"

## More results

Changing `choice_of_metric` TODO

## Acknowledgements

* Thanks to [Willis](https://twitter.com/ma_nanye) for all the guidance and pair-coding.
* Thanks to Hugging Face for supporting the compute.
* Thanks to Google for providing Gemini credits.
