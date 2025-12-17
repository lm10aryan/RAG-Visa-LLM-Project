from __future__ import annotations

from typing import Any, Dict

import requests


class ModelManager:
    """
    Lightweight wrapper around a locally hosted Ollama model, keeping the same
    interface that the rest of the pipeline expects from the original Groq client.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.2:1b") -> None:
        self.base_url = base_url
        self.model = model_name

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            "Based on the following context, answer the question concisely and accurately.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

    def _invoke(self, prompt: str) -> Dict[str, Any]:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=30,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "answer": f"Model error: {exc}",
                "confidence": 0.0,
                "model_used": self.model,
            }

        if response.status_code == 200:
            result = response.json()
            return {
                "answer": result.get("response", "").strip(),
                "confidence": 0.8,
                "model_used": self.model,
            }

        return {
            "answer": f"Error: {response.status_code}",
            "confidence": 0.0,
            "model_used": self.model,
        }

    def generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        prompt = self._build_prompt(query, context)
        return self._invoke(prompt)

    def generate_response(self, question: str, context: str = "") -> str:
        """
        Compatibility wrapper for existing callers that expect a simple string.
        """
        result = self.generate_answer(question, context)
        return result.get("answer", "")
