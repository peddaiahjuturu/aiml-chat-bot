"""ticket_classifier.py -- Part D task B: weak / zero-shot / few-shot ticket classification via
LangChain (ChatPromptTemplate + StrOutputParser) and the mock chat API.
Run:  python src/ticket_classifier.py
Reads data/raw/tickets.csv and data/raw/few_shot_examples.csv. Writes outputs/ticket_classification.csv.
"""
import os

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import constants  # noqa: F401
import mock_llm
from constants import FEW_SHOT_PATH, OUTPUT_DIR, TICKETS_RAW_PATH, TICKET_CLASSIFICATION_PATH
from llm_client import call_llm

LABELS = ["billing", "signal", "plan_change", "cancellation"]

WEAK_PROMPT = "TICKET CLASSIFIER. Classify the ticket."

STRONG_PROMPT = (
    "TICKET CLASSIFIER. You are an expert support-ticket classifier for a TV subscription service. "
    "Your task is to read the subscriber's ticket below and classify it into exactly one of these four "
    "labels: billing, signal, plan_change, cancellation. Respond with only the label, and nothing else -- "
    "do not create any new labels outside this list."
)


def _call_model(prompt_value):
    """Converts a ChatPromptTemplate PromptValue (via .to_messages()) to the API's message format and
    calls the model, returning the reply text."""
    role_map = {"system": "system", "human": "user", "ai": "assistant"}
    messages = [{"role": role_map.get(m.type, m.type), "content": m.content} for m in prompt_value.to_messages()]
    return call_llm(messages)


def build_chain(system_prompt):
    """Builds the prompt | model | StrOutputParser() chain for a given system prompt."""
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{ticket}")])
    return prompt | _call_model | StrOutputParser()


def build_few_shot_prompt():
    """Appends the 13 few-shot examples to the strong prompt, in the "Ticket: ...\\nLabel: ..." format
    that mock_llm's classifier reads."""
    examples = pd.read_csv(FEW_SHOT_PATH)
    lines = [STRONG_PROMPT, ""]
    for _, row in examples.iterrows():
        lines.append(f"Ticket: {row['text']}")
        lines.append(f"Label: {row['label']}")
        lines.append("")
    return "\n".join(lines)


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def run_prompt(name, system_prompt, tickets):
    section(f"{name} prompt")
    mock_llm.reset()
    chain = build_chain(system_prompt)
    outputs = [chain.invoke({"ticket": text}).strip() for text in tickets["text"]]
    valid = [o in LABELS for o in outputs]
    correct = [o == t for o, t in zip(outputs, tickets["true_category"])]
    recovered = mock_llm._state["calls"] - len(outputs)
    print(f"Valid labels: {sum(valid)} of {len(outputs)}")
    print(f"Accuracy: {sum(correct)} of {len(outputs)} ({sum(correct) / len(outputs) * 100:.2f}%)")
    print(f"Rate-limited calls recovered: {recovered}")
    return outputs


def main():
    tickets = pd.read_csv(TICKETS_RAW_PATH)
    tickets = tickets[tickets["text"].str.strip() != "..."].reset_index(drop=True)
    print(f"Readable tickets: {len(tickets)}")

    run_prompt("Weak", WEAK_PROMPT, tickets)
    zero_shot_outputs = run_prompt("Strong zero-shot", STRONG_PROMPT, tickets)
    few_shot_outputs = run_prompt("Few-shot", build_few_shot_prompt(), tickets)

    section("Few-shot mismatches vs. true_category")
    result = tickets.copy()
    result["zero_shot"] = zero_shot_outputs
    result["few_shot"] = few_shot_outputs
    wrong = result[result["few_shot"] != result["true_category"]]
    print(wrong[["ticket_id", "text", "true_category", "few_shot"]].to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result[["ticket_id", "true_category", "zero_shot", "few_shot"]].to_csv(TICKET_CLASSIFICATION_PATH, index=False)
    print(f"\nWrote {TICKET_CLASSIFICATION_PATH} ({len(result)} rows)")


if __name__ == "__main__":
    main()
