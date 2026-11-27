"""llm_client.py -- reliable client for the chat-completion API (mock by default, real when LLM_MODE=live).
Run the demonstrations with:  python src/llm_client_demo.py
"""
import os
import time

import requests

import constants  # noqa: F401  (adds the repo root to sys.path so mock_llm can be imported)
import mock_llm

MODEL = "openai/gpt-4o-mini"
BASE_DELAY = 0.05
MAX_RETRIES = 3


class LLMError(RuntimeError):
    """Raised for a non-retryable response from the chat API (any status other than 200 or 429)."""


def _build_headers():
    """Builds the Authorization: Bearer ... and Content-Type headers.

    In the default (mock) mode any non-empty bearer token is accepted. When the environment variable
    LLM_MODE=live is set, the key is read from the OPENROUTER_API_KEY environment variable -- the key
    itself is never hard-coded in this file.
    """
    if os.environ.get("LLM_MODE") == "live":
        key = os.environ.get("OPENROUTER_API_KEY", "")
    else:
        key = "mock-key"
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _post(headers, payload):
    """Dispatches the HTTP POST: to the offline mock_llm.post() by default, or to the real
    chat-completion endpoint (mock_llm.URL) via requests.post() when LLM_MODE=live is set.
    """
    if os.environ.get("LLM_MODE") == "live":
        return requests.post(mock_llm.URL, headers=headers, json=payload, timeout=30)
    return mock_llm.post(mock_llm.URL, headers=headers, json=payload)


def call_llm(messages, temperature=0.0, max_tokens=300):
    """Calls the chat-completion API with the given list of {"role", "content"} messages.

    Builds the Authorization/Content-Type headers and the {"model", "messages", "temperature",
    "max_tokens"} payload, then posts it (mock by default, real endpoint when LLM_MODE=live).
    Returns the assistant's reply text on a 200 response. Retries only on 429 (rate limited), at most
    MAX_RETRIES=3 times, sleeping BASE_DELAY * 2**n seconds before retry number n (n = 0, 1, 2). Any
    other non-200 status (400, 401, 404, or 429 after the retries are exhausted) raises LLMError. If the
    reply was truncated (finish_reason == "length"), a note is printed before returning the partial text.
    """
    headers = _build_headers()
    payload = {"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

    attempt = 0
    while True:
        resp = _post(headers, payload)
        if resp.status_code == 200:
            choice = resp.json()["choices"][0]
            if choice.get("finish_reason") == "length":
                print("llm_client: note -- response was truncated (finish_reason='length')")
            return choice["message"]["content"]
        if resp.status_code == 429 and attempt < MAX_RETRIES:
            time.sleep(BASE_DELAY * 2 ** attempt)
            attempt += 1
            continue
        raise LLMError(f"Chat API call failed with status {resp.status_code}: {resp.text}")
