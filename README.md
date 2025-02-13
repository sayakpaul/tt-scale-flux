# tt-scale-flux

Simple re-implementation of inference-time scaling Flux.1-Dev as introduced in [Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps](https://arxiv.org/abs/2501.09732) by Ma et al. We implement the random search strategy to scale the inference compute budget.

<div align="center">
<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/collage_Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_i@1-4.png" width=600/>
<p><i>Photo of an athlete cat explaining it’s latest scandal at a press conference to journalists.</i></p>
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

<details>
<summary>Click to expand</summary>

```bash
output/gemini/overall_score/20250213_034054$ tree 
.
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@1_s@1039315023.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@1_s@77559330.json
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@1_s@77559330.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@2_s@1046091514.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@2_s@1388753168.json
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@2_s@1388753168.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@2_s@1527774201.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@2_s@1632020675.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@1648932110.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@2033640094.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@2056028012.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@510118118.json
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@510118118.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@544879571.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@722867022.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@951309743.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@973580742.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1169137714.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1271234848.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1327836930.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1589777351.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1592595351.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1654773907.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1901647417.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1916603945.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@209448213.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@2104826872.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@532500803.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@710122236.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@744797903.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@754998363.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@823891989.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@836183088.json
└── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@836183088.png
```

</details>

Each JSON file should look like so:

<details>
<summary>Click to expand</summary>

```json
{
    "prompt": "Photo of an athlete cat explaining it\u2019s latest scandal at a press conference to journalists.",
    "search_round": 4,
    "num_noises": 16,
    "best_noise_seed": 836183088,
    "best_score": {
        "score": 9.5,
        "explanation": "Considering all aspects, especially the high level of accuracy, creativity, and visual appeal, the overall score reflects the model's excellent performance in generating this image."
    },
    "choice_of_metric": "overall_score",
    "best_img_path": "output/gemini/overall_score/20250213_034054/prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@836183088.png"
}
```

</details>

To limit the number of prompts, specify `--num_prompts`. By default, we use 2 prompts. Specify "--num_prompts=all" to use all.

Once the results are generated, process the results by running:

```bash
python process_results.py --path=path_to_the_output_dir
```

This should output a collage of the best images generated in each search round, grouped by the same prompt.

## Controlling the "scale"

By default, we use 4 `search_rounds` and start with a noise pool size of 2. Each search round scales up the pool size like so: `2 ** current_seach_round` (with indexing starting from 1). This is where the "scale" in inference-time scaling comes from. You can increase the compute budget by specifying a larger `search_rounds`.

For each search round, we serialize the images and best datapoint (characterized by the best eval score) in a JSON file.

For other supported CLI args, run `python main.py -h`.

## Controlling the verifier

If you don't want to use Gemini, you can use [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) as an option. Simply specify `--verifier_to_use=qwen` for this. 

> [!IMPORTANT]  
> This setup was tested on 2 H100s. If you want to do this on a single GPU, specify `--use_low_gpu_vram`.

You can also bring in your own verifier by implementing a so-called `Verifier` class following the structure of either of `GeminiVerifier` or `QwenVerifier`. You will then have to make adjustments to the following places:

* [Scoring](https://github.com/sayakpaul/tt-scale-flux/blob/c654bc066171aee9c765fa42a322f65415529a77/main.py#L135)
* [Sorting](https://github.com/sayakpaul/tt-scale-flux/blob/c654bc066171aee9c765fa42a322f65415529a77/main.py#L163)

By default, we use "overall_score" as the metric to obtain the best samples in each search round. You can change it by specifying `--choice_of_metric`. Supported values are: 

* "accuracy_to_prompt"
* "creativity_and_originality"
* "visual_quality_and_realism"
* "consistency_and_cohesion"
* "emotional_or_thematic_resonance"
* "overall_score"

The verifier prompt that is used during grading/verification is specified in [this file](./verifiers/verifier_prompt.txt). The prompt is a slightly modified version of the one specified in the Figure 16 of
the paper (Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps). You are welcome to 
experiment with a different prompt.

## More results

<details>
<summary>Click to expand</summary>

<table>
  <tr>
    <th>Result</th>
  </tr>
  <tr>
    <td>
      <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/collage_a_bustling_manga_street_devoid_of_vehicles_detailed_with_vibrant_colors_and_dynamic_l_i@1-4.jpeg" alt="Manga" width="650">
      <br>
      <i>a bustling manga street, devoid of vehicles, detailed with vibrant colors and dynamic<br> line work, characters in the background adding life and movement, under a soft golden<br> hour light, with rich textures and a lively atmosphere, high resolution, sharp focus</i>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/collage_Alice_in_a_vibrant_dreamlike_digital_painting_inside_the_Nemo_Nautilus_submarine__i@1-4.jpeg" alt="Alice" width="650">
      <br>
      <i>Alice in a vibrant, dreamlike digital painting inside the Nemo Nautilus submarine.</i>
    </td>
  </tr>
</table>

</details><br>

Both searches were performed with "overall_score" as the metric. Below is example, presenting a comparison
between the outputs of different metrics -- "overall_score" vs. "emotional_or_thematic_resonance" for the prompt:
"a tiny astronaut hatching from an egg on the moon":

<details>
<summary>Click to expand</summary>

<table>
  <tr>
    <th>Metric</th>
    <th>Result</th>
  </tr>
  <tr>
    <td>"overall_score"</td>
    <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/collage_a_tiny_astronaut_hatching_from_an_egg_on_the_moon_i@1-4.png" alt="overall" width="350"></td>
  </tr>
  <tr>
    <td>"emotional_or_thematic_resonance"</td>
    <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/collage_a_tiny_astronaut_hatching_from_an_egg_on_the_moon_i@1-4_thematic.png" alt="Alicet" width="350"></td>
  </tr>
</table>

</details>

## Acknowledgements

* Thanks to [Willis Ma](https://twitter.com/ma_nanye) for all the guidance and pair-coding.
* Thanks to Hugging Face for supporting the compute.
* Thanks to Google for providing Gemini credits.
