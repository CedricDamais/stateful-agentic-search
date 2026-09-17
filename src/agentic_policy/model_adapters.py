from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class TransformersCausalLM:
    """Thin shared runtime so multiple policies benchmark the exact same model instance."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the Qwen extra: pip install -e '.[qwen]'") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        kwargs = {"torch_dtype": "auto"}
        if device == "auto":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
        if device != "auto":
            self.model.to(device)

    @property
    def device(self):
        return self.model.device

    def chat_prompt(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            self.torch.mps.synchronize()

    def score_candidates(self, prompt: str, candidates: list[str]) -> tuple[list[float], int, float]:
        """Return sequence log-likelihoods using one batched forward pass."""
        self.synchronize()
        started = time.perf_counter()
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        suffixes = [self.tokenizer(candidate, add_special_tokens=False).input_ids for candidate in candidates]
        sequences = [prompt_ids + suffix for suffix in suffixes]
        max_length = max(map(len, sequences))
        padded = [ids + [self.tokenizer.pad_token_id] * (max_length - len(ids)) for ids in sequences]
        masks = [[1] * len(ids) + [0] * (max_length - len(ids)) for ids in sequences]
        input_ids = self.torch.tensor(padded, device=self.device)
        attention_mask = self.torch.tensor(masks, device=self.device)
        scores = []
        with self.torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = self.torch.log_softmax(logits, dim=-1)
            start = len(prompt_ids) - 1
            for row, suffix in enumerate(suffixes):
                positions = self.torch.arange(start, start + len(suffix), device=self.device)
                targets = input_ids[row, len(prompt_ids):len(prompt_ids) + len(suffix)]
                scores.append(float(log_probs[row, positions, targets].sum().cpu()))
        self.synchronize()
        return scores, len(prompt_ids), (time.perf_counter() - started) * 1000

    def score_next_tokens(self, prompt: str, candidates: list[str]) -> tuple[list[float], int, float]:
        """Score a finite set of single-token codes with one prompt forward pass."""
        token_ids = [self.tokenizer(candidate, add_special_tokens=False).input_ids for candidate in candidates]
        if any(len(ids) != 1 for ids in token_ids):
            raise ValueError(f"Finite action codes must each tokenize to one token: {token_ids}")
        self.synchronize()
        started = time.perf_counter()
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits[0, -1]
            scores = [float(logits[ids[0]].cpu()) for ids in token_ids]
        self.synchronize()
        return scores, encoded.input_ids.shape[1], (time.perf_counter() - started) * 1000

    def generate(self, prompt: str, max_new_tokens: int = 96) -> Generation:
        self.synchronize()
        started = time.perf_counter()
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_tokens = encoded.input_ids.shape[1]
        new_tokens = output[0, prompt_tokens:]
        self.synchronize()
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return Generation(text, prompt_tokens, len(new_tokens), (time.perf_counter() - started) * 1000)
