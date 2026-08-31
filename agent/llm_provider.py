"""Provider-agnostic LLM calls: OpenRouter or OpenAI."""
import os

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def resolve_provider(api_key=None, provider=None):
    """-> (provider, key), or (None, None) if nothing is configured."""
    if api_key and provider:
        return provider, api_key.strip()
    if api_key:
        return ("openrouter" if api_key.startswith("sk-or-") else "openai"), api_key.strip()

    for name, env in (("openrouter", "OPENROUTER_API_KEY"), ("openai", "OPENAI_API_KEY")):
        key = os.environ.get(env)
        if key and key.strip():
            return name, key.strip()
    return None, None


def _chat_completion(url, key, model, prompt, max_tokens, temperature, extra_headers=None):
    """Both providers speak the same OpenAI-compatible chat-completions shape."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    resp = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def call_llm(prompt, max_tokens=300, api_key=None, provider=None, model=None, temperature=0.2):
    """-> (text, meta). text is None on any failure; meta explains why."""
    resolved_provider, key = resolve_provider(api_key, provider)
    if not resolved_provider:
        return None, {"ok": False, "provider": None, "model": None,
                      "error": "No LLM provider configured (no API key supplied or found in environment)."}

    try:
        if resolved_provider == "openrouter":
            use_model = model or DEFAULT_OPENROUTER_MODEL
            text = _chat_completion(
                OPENROUTER_URL, key, use_model, prompt, max_tokens, temperature,
                extra_headers={"HTTP-Referer": "https://razorshield.local", "X-Title": "RazorShield"})
        elif resolved_provider == "openai":
            use_model = model or DEFAULT_OPENAI_MODEL
            text = _chat_completion(OPENAI_URL, key, use_model, prompt, max_tokens, temperature)
        else:
            return None, {"ok": False, "provider": resolved_provider, "model": None,
                          "error": f"Unknown provider '{resolved_provider}'. Use 'openrouter' or 'openai'."}
    except Exception as e:
        return None, {"ok": False, "provider": resolved_provider, "model": model, "error": str(e)}

    return text, {"ok": True, "provider": resolved_provider, "model": use_model, "error": None}
