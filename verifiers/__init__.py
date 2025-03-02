try:
    from .gemini_verifier import GeminiVerifier
except Exception as e:
    GeminiVerifier = None

try:
    from .openai_verifier import OpenAIVerifier
except Exception as e:
    OpenAIVerifier = None

try:
    from .qwen_verifier import QwenVerifier
except Exception as e:
    QwenVerifier = None

try:
    from .laion_aesthetics import LAIONAestheticVerifier
except Exception as e:
    LAIONAestheticVerifier = None

SUPPORTED_VERIFIERS = {
    "qwen": QwenVerifier if QwenVerifier else None,
    "gemini": GeminiVerifier if GeminiVerifier else None,
    "openai": OpenAIVerifier if OpenAIVerifier else None,
    "laion_aesthetic": LAIONAestheticVerifier if LAIONAestheticVerifier else None,
}

SUPPORTED_METRICS = {k: getattr(v, "SUPPORTED_METRIC_CHOICES", None) for k, v in SUPPORTED_VERIFIERS.items()}
