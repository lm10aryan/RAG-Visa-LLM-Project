from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None  # type: ignore


GROQ_MODEL = "llama-3.3-70b-versatile" 	


@dataclass
class ModelResponse:
    text: str


class ModelManager:
    """
    Thin wrapper around the Groq completion API for evaluation passes.

    The class defers Groq client creation until a key is available. If the key
    is missing, generation falls back to a deterministic placeholder so the
    rest of the pipeline can still execute.
    """

    def __init__(self, model_name: str = GROQ_MODEL) -> None:
        self.model_name = model_name
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.client: Optional[Groq] = None
        if self.api_key and Groq is not None:
            self.client = Groq(api_key=self.api_key)

    def _prompt(self, question: str, context: str) -> str:
        intro = "You are an H-1B visa subject-matter expert.\n"
        instructions = (
            "Use the provided context to answer the user's question.\n"
            "If the context lacks the answer, respond with your best effort "
            "and clearly state any uncertainty.\n\n"
        )
        prompt = f"{intro}{instructions}Context:\n{context or 'N/A'}\n\nQuestion:\n{question}\nAnswer:"
        return prompt

    def generate_response(self, question: str, context: str = "") -> str:
        if self.client is None:
            return "Model unavailable (Groq API key not configured)."

        prompt = self._prompt(question, context)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful immigration assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=256,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            return f"Groq request failed: {exc}"

    def test_connection(self) -> bool:
        if self.client is None:
            return False
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Health check assistant."},
                    {"role": "user", "content": "Reply with OK if you can read this."},
                ],
                max_tokens=4,
            )
            return bool(completion.choices)
        except Exception:
            return False


__all__ = ["ModelManager", "ModelResponse"]
