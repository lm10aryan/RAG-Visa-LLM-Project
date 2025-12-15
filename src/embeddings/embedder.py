from __future__ import annotations

from functools import lru_cache
import hashlib
import logging
import re
from typing import List, Sequence, Union

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class HashingSentenceTransformer:
    """Offline-friendly, deterministic embedding fallback.

    The hash encoder maps tokens into a fixed-dimension bag-of-words vector so
    semantic retrieval can still run in environments without access to the
    hosted model downloads. Vectors are L2-normalized to retain cosine
    similarity behaviour.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.is_fallback = True

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def encode(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,  # noqa: ARG002
        **_: object,
    ) -> np.ndarray:
        vectors = np.zeros((len(sentences), self.dimension), dtype=np.float32)

        for row, sentence in enumerate(sentences):
            for token in self._tokenize(sentence):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest, "big") % self.dimension
                vectors[row, bucket] += 1.0

        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = np.divide(
                vectors,
                norms,
                out=np.zeros_like(vectors),
                where=norms > 0,
            )

        if convert_to_numpy:
            return vectors
        return vectors.tolist()


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Lazily load and cache the sentence transformer model.

    In offline environments, fall back to a deterministic hashing encoder so
    semantic search and tests can still run without downloading checkpoints.
    """

    try:
        return SentenceTransformer(
            model_name,
            device="cpu",
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "Falling back to HashingSentenceTransformer due to load failure: %s",
            exc,
        )
        return HashingSentenceTransformer()


def _prepare_texts(data: Sequence[Union[str, dict]]) -> List[str]:
    texts: List[str] = []
    for item in data:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if text:
                texts.append(text)
    return texts


def generate_embeddings(
    data: Sequence[Union[str, dict]],
    model_name: str = DEFAULT_MODEL,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode chunk text into embedding vectors.
    """

    texts = _prepare_texts(data)
    if not texts:
        raise ValueError("No text available for embedding.")

    model = load_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )
    return embeddings


__all__ = ["generate_embeddings", "load_embedding_model", "DEFAULT_MODEL"]

