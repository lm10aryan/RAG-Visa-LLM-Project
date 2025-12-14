from __future__ import annotations

from src.reasoning.model_manager import ModelManager


def main() -> None:
    manager = ModelManager()
    print("Key set:", bool(manager.client))
    if not manager.client:
        print("Configure GROQ_API_KEY and try again.")
        return

    try:
        completion = manager.client.chat.completions.create(
            model=manager.model_name,
            messages=[{"role": "user", "content": "Hello from the health check"}],
            max_tokens=8,
        )
        print("Response:", completion.choices[0].message.content)
    except Exception as exc:  # noqa: BLE001
        print("API error:", exc)


if __name__ == "__main__":
    main()

