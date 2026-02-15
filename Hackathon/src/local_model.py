"""
BERT model for phishing detection only.
Used by the Scam & Phishing section. RAG uses TF-IDF, not BERT.
Uses ElSlay/BERT-Phishing-Email-Model (BERT-base fine-tuned; 0=Legitimate, 1=Phishing).
"""

from typing import List, Optional, Tuple

import numpy as np

# Lazy-loaded model and tokenizer
_model = None
_tokenizer = None
_MODEL_ID = "ElSlay/BERT-Phishing-Email-Model"
_EMBED_MAX_LENGTH = 128
_PHISHING_MAX_LENGTH = 512
# Shorter length for faster inference; 256 tokens is enough for most messages/URLs
_PHISHING_MAX_LENGTH_FAST = 256


def _get_model():
    """Lazy load the BERT phishing model and tokenizer. Returns (model, tokenizer) or (None, None)."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    try:
        from transformers import BertForSequenceClassification, BertTokenizer

        _tokenizer = BertTokenizer.from_pretrained(_MODEL_ID)
        _model = BertForSequenceClassification.from_pretrained(_MODEL_ID)
        _model.eval()
        return _model, _tokenizer
    except Exception:
        _model = None
        _tokenizer = None
        return None, None


def is_available() -> bool:
    """Return True if the local model can be loaded."""
    model, tokenizer = _get_model()
    return model is not None and tokenizer is not None


def get_embeddings(texts: List[str], max_length: int = _EMBED_MAX_LENGTH) -> List[List[float]]:
    """
    Encode texts using the BERT encoder (mean-pooled last hidden state).
    Used by RAG for similarity search. Returns list of 768-dim vectors.
    """
    model, tokenizer = _get_model()
    if model is None or tokenizer is None:
        return []

    try:
        import torch

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        with torch.no_grad():
            # Use the BERT base encoder only (no classification head)
            outputs = model.bert(**{k: v.to(model.device) for k, v in inputs.items()})
            # last_hidden_state: (batch, seq_len, 768); mean over seq_len
            hidden = outputs.last_hidden_state
            attention = inputs.get("attention_mask")
            if attention is not None:
                attention = attention.unsqueeze(-1).to(hidden.device).float()
                masked = hidden * attention
                lengths = attention.sum(dim=1, keepdim=True).clamp(min=1e-9)
                pooled = (masked.sum(dim=1) / lengths).cpu().numpy()
            else:
                pooled = hidden.mean(dim=1).cpu().numpy()
        return [row.tolist() for row in pooled]
    except Exception:
        return []


def predict_phishing(text: str, max_length: int = _PHISHING_MAX_LENGTH) -> Tuple[str, float]:
    """
    Classify text as phishing or legitimate. Uses the full classification model.
    Returns (verdict, confidence) where verdict is "Supported" (phishing) or "Refuted" (legitimate).
    """
    results = predict_phishing_batch([text], max_length=max_length)
    return results[0] if results else ("Unknown", 0.0)


def predict_phishing_batch(
    texts: List[str],
    max_length: int = _PHISHING_MAX_LENGTH_FAST,
) -> List[Tuple[str, float]]:
    """
    Classify multiple texts in one forward pass. Much faster than calling predict_phishing in a loop.
    Returns list of (verdict, confidence) in same order as texts. Empty/invalid texts get ("Unknown", 0.0).
    """
    model, tokenizer = _get_model()
    if model is None or tokenizer is None:
        return [("Unknown", 0.0)] * max(1, len(texts))

    # Normalize and filter empty; we'll reinsert Unknown for empty later
    normalized = []
    empty_indices: set = set()
    for i, t in enumerate(texts):
        s = (t or "").strip()
        if not s:
            empty_indices.add(i)
        else:
            normalized.append(s)

    if not normalized:
        return [("Unknown", 0.0)] * max(1, len(texts))

    try:
        import torch

        inputs = tokenizer(
            normalized,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # Label 1 = Phishing, 0 = Legitimate (per model card)
        out_list: List[Tuple[str, float]] = []
        idx = 0
        for i in range(len(texts)):
            if i in empty_indices:
                out_list.append(("Unknown", 0.0))
            else:
                p = probs[idx]
                phishing_prob = float(p[1])
                legit_prob = float(p[0])
                if phishing_prob >= 0.5:
                    out_list.append(("Supported", phishing_prob))
                else:
                    out_list.append(("Refuted", legit_prob))
                idx += 1
        return out_list
    except Exception:
        return [("Unknown", 0.0)] * max(1, len(texts))


class EmbeddingWrapper:
    """Thin wrapper so RAG verifier can call .encode(texts) and get [0].tolist()-compatible output."""

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        vectors = get_embeddings(texts, **{k: v for k, v in kwargs.items() if k == "max_length"})
        if not vectors:
            return np.array([])
        return np.array(vectors, dtype=np.float32)


def get_embedding_model() -> Optional[EmbeddingWrapper]:
    """Return an object with .encode(texts) for RAG compatibility. Returns None if model unavailable."""
    if not is_available():
        return None
    return EmbeddingWrapper()
