"""
Single BERT model for phishing detection and RAG embeddings.
Replaces sentence-transformers: one model for both phishing classification and embedding.
Uses ElSlay/BERT-Phishing-Email-Model (BERT-base fine-tuned for phishing; 0=Legitimate, 1=Phishing).
"""

from typing import List, Optional, Tuple

import numpy as np

# Lazy-loaded model and tokenizer
_model = None
_tokenizer = None
_MODEL_ID = "ElSlay/BERT-Phishing-Email-Model"
_EMBED_MAX_LENGTH = 128
_PHISHING_MAX_LENGTH = 512


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
    model, tokenizer = _get_model()
    if model is None or tokenizer is None:
        return "Unknown", 0.0

    if not (text or "").strip():
        return "Unknown", 0.0

    try:
        import torch

        inputs = tokenizer(
            text.strip(),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        # Label 1 = Phishing, 0 = Legitimate (per model card)
        phishing_prob = float(probs[1])
        legit_prob = float(probs[0])
        if phishing_prob >= 0.5:
            return "Supported", phishing_prob  # Supported = scam/phishing
        return "Refuted", legit_prob  # Refuted = legitimate
    except Exception:
        return "Unknown", 0.0


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
