"""
max/lab1/tests/test_lab1.py
Компактні тести ключової логіки ЛР1: токенізація, словник, маски, позиційне кодування.
"""

import sys
from pathlib import Path

# Додаємо директорію max/lab1 до шляхів пошуку
LAB_ROOT = Path(__file__).resolve().parent.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

import pytest
import torch
import torch.nn as nn

from utils import (
    PAD_IDX,
    UNK_IDX,
    BOS_IDX,
    EOS_IDX,
    Vocabulary,
    tokenize_text,
    create_causal_mask,
    PositionalEncoding,
)


def test_tokenization():
    text = "Subject: Hello World! This is an Enron email: $100."
    tokens = tokenize_text(text)
    assert "subject" in tokens
    assert "hello" in tokens
    assert "world" in tokens
    assert "!" in tokens
    assert "$" in tokens
    assert "100" in tokens


def test_vocabulary_encoding_decoding():
    texts = [
        "hello world",
        "hello enron spam",
        "world news",
    ]
    vocab = Vocabulary.build_from_texts(texts, min_freq=1, max_size=50)
    assert len(vocab) > 4  # special tokens + words
    assert vocab.encode(["hello"], add_bos=True, add_eos=True) == [
        BOS_IDX,
        vocab.token2id["hello"],
        EOS_IDX,
    ]
    assert vocab.encode(["unknownword"]) == [UNK_IDX]

    # Test padding
    encoded = vocab.encode(["hello"], max_len=5, pad_to_max=True)
    assert len(encoded) == 5
    assert encoded[1:] == [PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX]

    # Test serialization
    vocab_dict = vocab.to_dict()
    restored = Vocabulary.from_dict(vocab_dict)
    assert len(restored) == len(vocab)
    assert restored.token2id == vocab.token2id


def test_positional_encoding_shape():
    d_model = 64
    seq_len = 16
    batch_size = 4
    pe = PositionalEncoding(d_model=d_model, max_len=100, dropout=0.0)
    x = torch.zeros(batch_size, seq_len, d_model)
    out = pe(x)
    assert out.shape == (batch_size, seq_len, d_model)
    # Output should not be all zeros because PE is added
    assert not torch.allclose(out, torch.zeros_like(out))


def test_causal_mask():
    seq_len = 4
    mask = create_causal_mask(seq_len)
    assert mask.shape == (seq_len, seq_len)
    # Positions (i, j) where j > i should be True (masked)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                assert mask[i, j].item() is True
            else:
                assert mask[i, j].item() is False


def test_transformer_decoder_no_future_leakage():
    """Перевірка, що декодер із causal mask не витікає інформацію з майбутніх токенів."""
    vocab_size = 50
    d_model = 32
    nhead = 2
    seq_len = 8

    embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
    decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=64, dropout=0.0, batch_first=True)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
    projection = nn.Linear(d_model, vocab_size)

    embedding.eval()
    decoder.eval()
    projection.eval()

    # Створюємо послідовність seq1
    seq1 = torch.tensor([[5, 10, 15, 20, 25, 30, 35, 40]])
    # Створюємо seq2 з ідентичним початком до позиції 4, але зміненими токенами на позиціях 5..7
    seq2 = torch.tensor([[5, 10, 15, 20, 25, 9, 8, 7]])

    with torch.no_grad():
        emb1 = embedding(seq1)
        emb2 = embedding(seq2)
        mem1 = torch.zeros(1, 1, d_model)
        mem2 = torch.zeros(1, 1, d_model)
        mask = create_causal_mask(seq_len)

        out1 = projection(decoder(emb1, mem1, tgt_mask=mask))
        out2 = projection(decoder(emb2, mem2, tgt_mask=mask))

    # Виходи на позиціях 0..4 мають бути ідентичними!
    assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-5)
    # Виходи на позиції 5 і далі мають відрізнятися
    assert not torch.allclose(out1[:, 5:, :], out2[:, 5:, :], atol=1e-4)


def test_encoder_forward_and_padding_mask():
    """Перевірка прямого проходу Transformer Encoder із padding mask."""
    vocab_size = 100
    d_model = 64
    seq_len = 16
    batch_size = 2

    emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
    pe = PositionalEncoding(d_model=d_model, max_len=seq_len, dropout=0.0)
    layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=128, dropout=0.0, batch_first=True)
    encoder = nn.TransformerEncoder(layer, num_layers=1)
    classifier = nn.Linear(d_model, 2)

    # Batch з різною кількістю PAD токенів
    x = torch.tensor([
        [10, 15, 20, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 15, 20, 25, 30, 35, 40, 45, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=torch.long)

    padding_mask = (x == PAD_IDX)
    h = pe(emb(x) * (d_model ** 0.5))
    encoded = encoder(h, src_key_padding_mask=padding_mask)

    # Masked pooling
    mask = (~padding_mask).unsqueeze(-1).float()
    pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    logits = classifier(pooled)

    assert logits.shape == (batch_size, 2)
    assert not torch.isnan(logits).any()


def test_checkpoint_loading_and_inference():
    """Перевірка коректності завантаження натренованих чекпоінтів та інференсу."""
    checkpoints_dir = LAB_ROOT / "outputs" / "checkpoints"
    enron_ckpt_path = checkpoints_dir / "best_enron_transformer.pt"
    wiki_ckpt_path = checkpoints_dir / "best_wikitext_decoder.pt"

    assert enron_ckpt_path.exists(), "Чекпоінт Enron повинен існувати!"
    assert wiki_ckpt_path.exists(), "Чекпоінт WikiText повинен існувати!"

    # 1. Enron Checkpoint
    enron_data = torch.load(enron_ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in enron_data
    assert "vocab" in enron_data
    assert "config" in enron_data
    assert enron_data["best_val_f1"] > 0.90

    # 2. WikiText Checkpoint
    wiki_data = torch.load(wiki_ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in wiki_data
    assert "vocab" in wiki_data
    assert "config" in wiki_data
    assert wiki_data["best_val_ppl"] < 250.0


def test_autoregressive_generation_runs():
    """Перевірка виконання авторегресивної генерації на відновленій моделі декодера."""
    wiki_ckpt_path = LAB_ROOT / "outputs" / "checkpoints" / "best_wikitext_decoder.pt"
    wiki_data = torch.load(wiki_ckpt_path, map_location="cpu", weights_only=False)
    vocab = Vocabulary.from_dict(wiki_data["vocab"])
    cfg = wiki_data["config"]

    emb = nn.Embedding(len(vocab), cfg["d_model"], padding_idx=PAD_IDX)
    pe = PositionalEncoding(cfg["d_model"], max_len=64, dropout=0.0)
    layer = nn.TransformerDecoderLayer(cfg["d_model"], cfg["nhead"], dim_feedforward=cfg["dim_ff"], dropout=0.0, batch_first=True)
    decoder = nn.TransformerDecoder(layer, num_layers=cfg["num_layers"])
    proj = nn.Linear(cfg["d_model"], len(vocab))

    state = wiki_data["model_state_dict"]
    emb.load_state_dict({"weight": state["embedding.weight"]})
    proj.load_state_dict({"weight": state["projection.weight"], "bias": state["projection.bias"]})

    emb.eval()
    decoder.eval()
    proj.eval()

    prompt_ids = vocab.encode(["the", "game", "was"])
    generated = list(prompt_ids)
    with torch.no_grad():
        for _ in range(5):
            inp = torch.tensor([generated[-64:]], dtype=torch.long)
            c_mask = create_causal_mask(inp.size(1))
            mem = torch.zeros(1, 1, cfg["d_model"])
            h = pe(emb(inp) * (cfg["d_model"] ** 0.5))
            logits = proj(decoder(h, mem, tgt_mask=c_mask))[0, -1, :]
            logits[PAD_IDX] = -float("inf")
            next_token = int(torch.argmax(logits).item())
            generated.append(next_token)

    assert len(generated) == len(prompt_ids) + 5
    generated_text = vocab.decode(generated)
    assert isinstance(generated_text, str)
    assert len(generated_text) > 0


def test_autoregressive_generation_with_sampling_runs():
    """Перевірка генерації з Temperature та Top-k sampling."""
    from utils import generate_text_autoregressive

    wiki_ckpt_path = LAB_ROOT / "outputs" / "checkpoints" / "best_wikitext_decoder.pt"
    wiki_data = torch.load(wiki_ckpt_path, map_location="cpu", weights_only=False)
    vocab = Vocabulary.from_dict(wiki_data["vocab"])
    cfg = wiki_data["config"]

    class DecoderLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(vocab), cfg["d_model"], padding_idx=PAD_IDX)
            self.pos_encoder = PositionalEncoding(cfg["d_model"], max_len=64, dropout=0.0)
            layer = nn.TransformerDecoderLayer(cfg["d_model"], cfg["nhead"], dim_feedforward=cfg["dim_ff"], dropout=0.0, batch_first=True)
            self.decoder = nn.TransformerDecoder(layer, num_layers=cfg["num_layers"])
            self.projection = nn.Linear(cfg["d_model"], len(vocab))

        def forward(self, x):
            b, s = x.shape
            c_mask = create_causal_mask(s, device=x.device)
            p_mask = (x == PAD_IDX)
            emb = self.pos_encoder(self.embedding(x) * (cfg["d_model"] ** 0.5))
            mem = emb.new_zeros(b, 1, cfg["d_model"])
            dec = self.decoder(emb, mem, tgt_mask=c_mask, tgt_key_padding_mask=p_mask)
            return self.projection(dec)

    model = DecoderLM()
    model.load_state_dict(wiki_data["model_state_dict"])
    model.eval()

    prompt = "the game was"
    # 1. Greedy
    cont_g, full_g = generate_text_autoregressive(prompt, model, vocab, torch.device("cpu"), max_new_tokens=6, do_sample=False)
    assert isinstance(cont_g, str)
    assert len(cont_g) > 0

    # 2. Sampling
    cont_s, full_s = generate_text_autoregressive(
        prompt, model, vocab, torch.device("cpu"), max_new_tokens=6, do_sample=True, temperature=0.85, top_k=25, repetition_penalty=1.2
    )
    assert isinstance(cont_s, str)
    assert len(cont_s) > 0

