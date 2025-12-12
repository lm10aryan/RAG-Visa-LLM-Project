from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Sequence, Union

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Lazily load and cache the sentence transformer model.
    """

    return SentenceTransformer(model_name)


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

