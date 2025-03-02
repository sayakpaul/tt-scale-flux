import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VERIFIER_PROMPT_PATH = os.path.join(SCRIPT_DIR, "verifier_prompt.txt")


class BaseVerifier:
    SUPPORTED_METRIC_CHOICES = None

    def __init__(self, seed=1994, prompt_path=None):
        prompt_path = prompt_path or DEFAULT_VERIFIER_PROMPT_PATH
        self.seed = seed
        self.verifier_prompt = self.load_verifier_prompt(prompt_path)

    @staticmethod
    def load_verifier_prompt(prompt_path: str) -> str:
        with open(prompt_path, "r") as f:
            return f.read()

    def prepare_inputs(self, images, prompts, **kwargs):
        """
        Prepare inputs for the verifier given images and prompts.
        """
        raise NotImplementedError

    def score(self, inputs, **kwargs):
        """
        Score the given inputs and return a list of results.
        """
        raise NotImplementedError
