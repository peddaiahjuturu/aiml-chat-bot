"""llm_client_demo.py -- demonstrates llm_client.call_llm()'s error handling, truncation, statelessness,
and rate-limit retry behaviour (Part D, task A). Run:  python src/llm_client_demo.py
"""
import constants  # noqa: F401
import mock_llm
from llm_client import MODEL, call_llm


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def demo_401_404():
    section("(a) 401 (missing Authorization) and 404 (wrong URL)")
    mock_llm.reset()
    resp_401 = mock_llm.post(mock_llm.URL, headers={}, json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    resp_404 = mock_llm.post("https://wrong.example/api", headers={"Authorization": "Bearer x"},
                              json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    print(f"No Authorization header -> status {resp_401.status_code}")
    print(f"Wrong URL -> status {resp_404.status_code}")


def demo_length():
    section("(b) max_tokens=3 truncation")
    mock_llm.reset()
    reply = call_llm([{"role": "user", "content": "Tell me something interesting about television."}], max_tokens=3)
    print(f"Reply with max_tokens=3: {reply!r}")


def demo_no_memory():
    section("(c) the model has no memory across separate calls")
    mock_llm.reset()
    reply1 = call_llm([{"role": "user", "content": "What is my name?"}])
    print(f"Asked alone: {reply1!r}")

    mock_llm.reset()
    reply2 = call_llm([
        {"role": "user", "content": "My name is Asha."},
        {"role": "assistant", "content": "Nice to meet you, Asha!"},
        {"role": "user", "content": "What is my name?"},
    ])
    print(f"Asked with the earlier turn in the same messages list: {reply2!r}")


def demo_rate_limit_retry():
    section("(d) 30 simple calls recover from rate limiting")
    mock_llm.reset()
    for _ in range(30):
        call_llm([{"role": "user", "content": "hello"}])
    print(f"Total calls made to mock_llm.post: {mock_llm._state['calls']}")


if __name__ == "__main__":
    demo_401_404()
    demo_length()
    demo_no_memory()
    demo_rate_limit_retry()
