from openai import OpenAI
from pydantic import BaseModel
import os
from typing import Union
from PIL import Image
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.insert(0, current_dir)
sys.path.insert(0, root_dir)

from base_verifier import BaseVerifier
from utils import convert_to_bytes


class Score(BaseModel):
    score: int
    explanation: str


class Grading(BaseModel):
    accuracy_to_prompt: Score
    creativity_and_originality: Score
    visual_quality_and_realism: Score
    consistency_and_cohesion: Score
    emotional_or_thematic_resonance: Score
    overall_score: Score


class OpenAIVerifier(BaseVerifier):
    SUPPORTED_METRIC_CHOICES = [
        "accuracy_to_prompt",
        "creativity_and_originality",
        "visual_quality_and_realism",
        "consistency_and_cohesion",
        "emotional_or_thematic_resonance",
        "overall_score",
    ]

    def __init__(self, seed=1994, model_name="gpt-4o-2024-11-20", **kwargs):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        super().__init__(seed=seed, prompt_path=kwargs.pop("prompt_path", None))
        self.model_name = model_name
        self.seed = seed

    def prepare_inputs(self, images: Union[list[Image.Image], Image.Image], prompts: Union[list[str], str], **kwargs):
        """Prepare inputs for the API from a given prompt and image."""
        inputs = []
        images = images if isinstance(images, list) else [images]
        prompts = prompts if isinstance(prompts, list) else [prompts]

        for prompt, image in zip(prompts, images):
            # Convert image to base64
            base64_image = convert_to_bytes(image, b64_encode=True)

            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
            inputs.append(message)

        return inputs

    def score(self, inputs, **kwargs) -> list[dict[str, float]]:
        system_message = {"role": "system", "content": self.verifier_prompt}

        def call_generate_content(parts):
            conversation = [system_message, parts]
            response = self.client.beta.chat.completions.parse(
                model=self.model_name, messages=conversation, temperature=1, response_format=Grading
            )
            return response.choices[0].message.parsed.model_dump()

        results = []
        max_workers = min(len(inputs), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_generate_content, group) for group in inputs]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    # Handle exceptions as appropriate.
                    print(f"An error occurred during API call: {e}")
        return results


# Define inputs
if __name__ == "__main__":
    verifier = OpenAIVerifier()
    image_urls = [
        (
            "realistic photo a shiny black SUV car with a mountain in the background.",
            "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/assets/car.jpg",
        ),
        (
            "photo a green and funny creature standing in front a lightweight forest.",
            "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/assets/green_creature.jpg",
        ),
    ]

    prompts = []
    images = []
    for text, path_or_url in image_urls:
        prompts.append(text)
        images.append(path_or_url)

    # # Single image
    # response = client.models.generate_content(
    #     model='gemini-2.0-flash',
    #     contents=[
    #         "realistic photo a shiny black SUV car with a mountain in the background.",
    #         load_image("https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/assets/car.jpg")
    #     ],
    #     config=generation_config
    # )
    inputs = verifier.prepare_inputs(images=images, prompts=prompts)
    response = verifier.score(inputs)

    with open("results.json", "w") as f:
        json.dump(response, f)

    print(json.dumps(response, indent=4))
