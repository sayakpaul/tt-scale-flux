import torch
import numpy as np
from typing import Dict, List, Tuple
from diffusers import DiffusionPipeline
from tqdm.auto import tqdm


class PathSearch:
    def __init__(
        self,
        pipe: DiffusionPipeline,
        verifier,
        config: dict,
        N: int = 4,  # Number of initial paths
        M: int = 8,  # Number of noises per path
        sigma_start: float = 0.8,  # Starting noise level
        delta_b: float = 0.2,  # Backward step size
        delta_f: float = 0.1,  # Forward step size
        L: int = 20,  # Number of function evaluations per backward step
    ):
        """
        Initialize the path search algorithm.

        Args:
            pipe: The diffusion pipeline
            verifier: The verifier model for scoring
            config: Configuration dictionary
            N: Number of initial paths
            M: Number of noises per path
            sigma_start: Starting noise level
            delta_b: Backward step size
            delta_f: Forward step size
            L: Number of function evaluations per backward step
        """
        self.pipe = pipe
        self.verifier = verifier
        self.config = config

        # Validate parameters
        if delta_b <= delta_f:
            raise ValueError(
                "Backward step size (delta_b) must be greater than forward step size (delta_f)"
            )

        self.N = N
        self.M = M
        self.sigma_start = sigma_start
        self.delta_b = delta_b
        self.delta_f = delta_f
        self.L = L

        # Get pipeline call arguments
        self.pipeline_call_args = config["pipeline_call_args"].copy()
        self.height = self.pipeline_call_args.get("height", 512)
        self.width = self.pipeline_call_args.get("width", 512)
        self.dtype = self.pipeline_call_args.get("dtype", torch.float32)

        # Get verifier arguments
        self.verifier_args = config.get("verifier_args", {})
        self.choice_of_metric = self.verifier_args.get("choice_of_metric", None)

    def sample_initial_noises(self) -> Dict[int, torch.Tensor]:
        """Sample N initial noises using the pipeline's latent preparation function."""
        from utils import get_noises, get_latent_prep_fn

        fn = get_latent_prep_fn(self.config["pretrained_model_name_or_path"])
        noises = get_noises(
            max_seed=np.iinfo(np.int32).max,
            num_samples=self.N,
            height=self.height,
            width=self.width,
            device="cuda",
            dtype=self.dtype,
            fn=fn,
            **self.pipeline_call_args,
        )
        return noises

    def forward_noise(
        self, x: torch.Tensor, sigma: float, delta_f: float
    ) -> torch.Tensor:
        """
        Simulate forward noising process from sigma to sigma + delta_f.

        Args:
            x: Input tensor
            sigma: Current noise level
            delta_f: Forward step size

        Returns:
            Noised tensor at noise level sigma + delta_f
        """
        # Generate noise with proper scaling
        noise = torch.randn_like(x)
        # Scale noise according to the noise schedule
        noise_scale = np.sqrt(delta_f)
        return x + noise_scale * noise

    def backward_step(
        self, x: torch.Tensor, sigma: float, delta_b: float
    ) -> torch.Tensor:
        """
        Run ODE solver from sigma to sigma - delta_b.

        Args:
            x: Input tensor
            sigma: Current noise level
            delta_b: Backward step size

        Returns:
            Denoised tensor at noise level sigma - delta_b
        """
        # Convert sigma to timestep if needed (depends on the scheduler)
        timesteps = torch.linspace(sigma, sigma - delta_b, self.L, device=x.device)

        # Run ODE solver steps
        for t in timesteps:
            # Get model prediction
            model_output = self.pipe.unet(x, t).sample

            # Run scheduler step
            scheduler_output = self.pipe.scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=x,
                **self.pipeline_call_args,
            )
            x = scheduler_output.prev_sample

        return x

    def score_samples(self, samples: List[torch.Tensor], prompt: str) -> List[float]:
        """Score samples using the verifier."""
        # Convert samples to images if needed
        images = []
        for sample in samples:
            if hasattr(self.pipe, "vae"):
                image = self.pipe.vae.decode(
                    sample / self.pipe.vae.config.scaling_factor
                ).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = (image * 255).round().astype("uint8")
                images.append(image)
            else:
                images.append(sample)

        # Prepare verifier inputs and score
        verifier_inputs = self.verifier.prepare_inputs(
            images=images, prompts=[prompt] * len(images)
        )
        outputs = self.verifier.score(inputs=verifier_inputs)

        # Extract scores based on the chosen metric
        scores = []
        for output in outputs:
            if isinstance(output[self.choice_of_metric], dict):
                scores.append(output[self.choice_of_metric]["score"])
            else:
                scores.append(output[self.choice_of_metric])

        return scores

    def search(self, prompt: str) -> Tuple[torch.Tensor, float]:
        """
        Run the path search algorithm.

        Returns:
            Tuple of (best sample, best score)
        """
        # Step 1: Sample initial noises and run ODE solver to noise level σ
        print(
            "\nStep 1: Sampling initial noises and running ODE solver to noise level σ"
        )
        initial_noises = self.sample_initial_noises()
        current_samples = []
        for noise in tqdm(initial_noises.values(), desc="Running initial ODE solver"):
            # Run ODE solver from t=1 to t=σ
            sample = self.backward_step(noise, 1.0, 1.0 - self.sigma_start)
            current_samples.append(sample)
        current_sigma = self.sigma_start

        while current_sigma > 0:
            print(f"\nCurrent sigma: {current_sigma:.3f}")

            # Step 2: Forward noise each sample
            print("Step 2: Forward noising")
            forward_noised_samples = []
            for sample in current_samples:
                for _ in range(self.M):
                    forward_noised = self.forward_noise(
                        sample, current_sigma, self.delta_f
                    )
                    forward_noised_samples.append(forward_noised)

            # Step 3: Backward step on each forward-noised sample
            print("Step 3: Running backward steps")
            backward_samples = []
            for sample in tqdm(forward_noised_samples, desc="Running backward steps"):
                backward_sample = self.backward_step(
                    sample, current_sigma + self.delta_f, self.delta_b
                )
                backward_samples.append(backward_sample)

            # Score all backward samples
            print("Scoring samples")
            scores = self.score_samples(backward_samples, prompt)

            # Keep top N samples
            top_indices = np.argsort(scores)[-self.N :]
            current_samples = [backward_samples[i] for i in top_indices]

            # Update sigma
            current_sigma -= self.delta_b

        # Step 4: Final random search on remaining samples
        print("\nStep 4: Final random search")
        final_scores = self.score_samples(current_samples, prompt)
        best_idx = np.argmax(final_scores)

        return current_samples[best_idx], final_scores[best_idx]
