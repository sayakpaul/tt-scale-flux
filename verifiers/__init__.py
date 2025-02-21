try:
    from .gemini_verifier import GeminiVerifier
except:
    GeminiVerifier = None

try:
    from .qwen_verifier import QwenVerifier
except:
    QwenVerifier = None

SUPPORTED_VERIFIERS = {
    "qwen": QwenVerifier if QwenVerifier else None,
    "gemini": GeminiVerifier if GeminiVerifier else None,
}

SUPPORTED_METRICS = {k: getattr(v, "SUPPORTED_METRIC_CHOICES", None) for k, v in SUPPORTED_VERIFIERS.items()}
