from __future__ import annotations
import os
from typing import Any, Dict, Optional
from openai import OpenAI

def make_client(timeout_sec: int = 60) -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Create .env or export the environment variable.")
    return OpenAI(api_key=api_key, timeout=timeout_sec)

def chat_json(
    client: OpenAI,
    *,
    model: str,
    system_message: str,
    user_message: str,
    seed: int,
    temperature: Optional[float],
    top_p: Optional[float],
    use_json_mode: bool,
) -> Any:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "seed": seed,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs)
