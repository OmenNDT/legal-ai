import os
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

@dataclass
class GeneratorConfig:
    base_model: str = "vinai/bartpho-syllable"
    lora_weights: str | None = None
    max_new_tokens: int = 256
    num_beams: int = 4
    device: str = "auto"

    @classmethod
    def from_env(cls) -> "GeneratorConfig":
        return cls(
            base_model = os.getenv("BARTPHO_MODEL", "vinai/bartpho-syllable"),
            lora_weights = os.getenv("LORA_WEIGHTS", None),
            max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", 256))
        )

class Generator:
    def __init__(self, config: GeneratorConfig) -> None:
        self._cfg = config
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        print(f"[Generator] Nạp model: {self._cfg.base_model}")
        self._tokenizer = AutoTokenizer.from_pretrained(self._cfg.base_model)

        base = AutoModelForSeq2SeqLM.from_pretrained(
            self._cfg.base_model,
            device_map = self._cfg.device,
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        if self._cfg.lora_weights:
            self._model = PeftModel.from_pretrained(base, self._cfg.lora_weights)
            print(f"[Generator] Đã nạp LoRA từ: {self._cfg.lora_weights}")
        else:
            self._model = base
        self._model.eval()

    def generate(self, prompt: str) -> str:
        self._load()
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model hoặc tokenizer chưa được nạp thành công.")
        inputs = self._tokenizer(
            prompt,
            return_tensors = "pt",
            truncation = True,
            max_length = 1024
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens = self._cfg.max_new_tokens,
                num_beams = self._cfg.num_beams,
                early_stopping = True
            )

        return self._tokenizer.decode(output_ids[0], skip_special_tokens = True)
