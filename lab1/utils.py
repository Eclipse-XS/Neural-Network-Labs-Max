"""
max/lab1/utils.py
Спільні утиліти для ЛР1: токенізація, словник, позиційне кодування, причинна маска.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Спеціальні токени
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

# Регулярний вираз для word-level токенізації (слова та окремі знаки пунктуації)
TOKEN_REGEX = re.compile(r"\w+|[^\w\s]")


def tokenize_text(text: str) -> List[str]:
    """Розбиває рядок на токени в нижньому регістрі за допомогою regex."""
    if not isinstance(text, str):
        return []
    return TOKEN_REGEX.findall(text.lower())


class Vocabulary:
    """Клас словника з підтримкою спеціальних токенів та серіалізації."""

    def __init__(
        self,
        token2id: Optional[Dict[str, int]] = None,
        id2token: Optional[Dict[int, str]] = None,
    ):
        if token2id is not None and id2token is not None:
            self.token2id = token2id
            self.id2token = {int(k): v for k, v in id2token.items()}
        else:
            self.token2id = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
            self.id2token = {idx: tok for idx, tok in enumerate(SPECIAL_TOKENS)}

    def __len__(self) -> int:
        return len(self.token2id)

    def __contains__(self, token: str) -> bool:
        return token in self.token2id

    @classmethod
    def build_from_texts(
        cls,
        texts: List[str],
        min_freq: int = 2,
        max_size: Optional[int] = 10000,
    ) -> "Vocabulary":
        """Будує словник за списком сирих текстів (тільки train split!)."""
        counter = Counter()
        for text in texts:
            tokens = tokenize_text(text)
            counter.update(tokens)

        vocab = cls()
        # Сортуємо токени за спаданням частоти, потім алфавітно для детермінізму
        sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

        for token, count in sorted_tokens:
            if count < min_freq:
                break
            if max_size is not None and len(vocab) >= max_size:
                break
            if token not in vocab.token2id:
                new_idx = len(vocab)
                vocab.token2id[token] = new_idx
                vocab.id2token[new_idx] = token

        return vocab

    def encode(
        self,
        tokens: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: Optional[int] = None,
        pad_to_max: bool = False,
    ) -> List[int]:
        """Кодує список токенів у список числових ID."""
        ids = []
        if add_bos:
            ids.append(BOS_IDX)

        for tok in tokens:
            ids.append(self.token2id.get(tok, UNK_IDX))

        if add_eos:
            ids.append(EOS_IDX)

        if max_len is not None:
            ids = ids[:max_len]
            if pad_to_max and len(ids) < max_len:
                ids = ids + [PAD_IDX] * (max_len - len(ids))

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Декодує список числових ID у текстовий рядок."""
        tokens = []
        for token_id in ids:
            tok = self.id2token.get(token_id, UNK_TOKEN)
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def to_dict(self) -> dict:
        return {
            "token2id": self.token2id,
            "id2token": {str(k): v for k, v in self.id2token.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        return cls(
            token2id=data["token2id"],
            id2token={int(k): v for k, v in data["id2token"].items()},
        )


class PositionalEncoding(nn.Module):
    """
    Синусоїдальне позиційне кодування:
    PE(pos, 2i) = sin(pos / (10000^(2i / d_model)))
    PE(pos, 2i+1) = cos(pos / (10000^(2i / d_model)))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Реєструємо як buffer (не є навчальним параметром)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor форми [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


def create_causal_mask(size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Створює квадратну булеву причинну маску для декодера.
    True позначає позиції, які мають бути замасковані (майбутні токени).
    """
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
    return mask


def set_seed(seed: int = 42) -> None:
    """Фіксує генератори випадкових чисел для повної відтворюваності."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_plot_style():
    """Встановлює єдиний академічний стиль для візуалізацій Matplotlib та Seaborn."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "semibold",
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })



def generate_text_autoregressive(
    prompt: str,
    model: nn.Module,
    vocab: Vocabulary,
    device: torch.device,
    max_new_tokens: int = 25,
    context_len: int = 64,
    do_sample: bool = False,
    temperature: float = 0.85,
    top_k: int = 25,
    repetition_penalty: float = 1.25,
) -> Tuple[str, str]:
    """
    Авторегресивна генерація продовження тексту з підтримкою greedy та temperature/top-k sampling.
    """
    import torch.nn.functional as F

    model.eval()
    tokens = tokenize_text(prompt)
    if not tokens:
        tokens = [BOS_TOKEN]
    input_ids = vocab.encode(tokens)
    generated_ids = list(input_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_context = generated_ids[-context_len:]
            x_tensor = torch.tensor([curr_context], dtype=torch.long, device=device)
            logits = model(x_tensor)[0, -1, :].clone()
            logits[PAD_IDX] = -float("inf")

            if not do_sample:
                next_token_id = int(torch.argmax(logits).item())
            else:
                if repetition_penalty != 1.0:
                    for past_id in set(generated_ids[-context_len:]):
                        if logits[past_id] > 0:
                            logits[past_id] /= repetition_penalty
                        else:
                            logits[past_id] *= repetition_penalty

                if temperature > 0:
                    logits = logits / temperature

                if top_k > 0:
                    top_k_val = min(top_k, logits.size(-1))
                    vals, _ = torch.topk(logits, top_k_val)
                    logits[logits < vals[-1]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())

            if next_token_id == EOS_IDX:
                break
            generated_ids.append(next_token_id)

    continuation_ids = generated_ids[len(input_ids):]
    continuation_text = vocab.decode(continuation_ids)
    full_text = vocab.decode(generated_ids)
    return continuation_text, full_text

