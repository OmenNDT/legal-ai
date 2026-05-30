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
    # Khi chưa có LoRA checkpoint, mô hình base BARTpho chưa được fine-tune
    # cho QA → câu sinh ra thường nhiễu. Trong trường hợp đó dùng extractive
    # (ghép trực tiếp các Điều/Khoản đã retrieve) để câu trả lời chính xác
    # với văn bản gốc. Set lazy_lora = True để KHÔNG load model nếu không có LoRA.
    lazy_lora: bool = True

    @classmethod
    def from_env(cls) -> "GeneratorConfig":
        return cls(
            base_model = os.getenv("BARTPHO_MODEL", "vinai/bartpho-syllable"),
            lora_weights = os.getenv("LORA_WEIGHTS", None) or None,
            max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", 256))
        )

class Generator:
    def __init__(self, config: GeneratorConfig) -> None:
        self._cfg = config
        self._tokenizer = None
        self._model = None

    @property
    def mode(self) -> str:
        return "lora" if self._cfg.lora_weights else "extractive"

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self._cfg.lora_weights and self._cfg.lazy_lora:
            # Chưa có LoRA → KHÔNG load BARTpho base (vô nghĩa cho QA).
            # Pipeline sẽ rơi vào nhánh extractive.
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
            raise RuntimeError("Mô hình BARTpho chưa được nạp (chưa có LoRA checkpoint).")
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
                num_beams = 6,
                no_repeat_ngram_size = 3,
                length_penalty = 1.2,
                repetition_penalty = 1.15,
                early_stopping = True
            )

        return self._tokenizer.decode(output_ids[0], skip_special_tokens = True)
