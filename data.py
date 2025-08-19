from dataclasses import dataclass


@dataclass
class TranslateReq:
    sentence: str
    target_lang: str
    resample: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = .0
    top_p: float = 1e-6
    top_k: int = -1
    min_p: float = 0.0
    seed: int = None
    max_tokens: int = 4096
    min_tokens: int = 0
