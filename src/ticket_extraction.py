"""ticket_extraction.py -- Part D task C: validated structured extraction (TicketRecord) with
retry-on-validation-failure and graceful manual_review fallback.
Run:  python src/ticket_extraction.py
Reads data/raw/tickets.csv. Writes outputs/ticket_extractions.jsonl.
"""
import json
import os
from typing import Literal, Optional

import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

import constants  # noqa: F401
import mock_llm
from constants import OUTPUT_DIR, TICKETS_RAW_PATH, TICKET_EXTRACTIONS_PATH
from llm_client import call_llm

MAX_RETRIES = 2
EXTRACTION_SYSTEM_TEMPLATE = (
    "TICKET EXTRACTOR. Extract the fields below from the subscriber's ticket that follows.\n\n{format_instructions}"
)


class TicketRecord(BaseModel):
    subscriber_id: Optional[str] = None
    category: Literal["billing", "signal", "plan_change", "cancellation", "other"]
    urgency: int = Field(ge=1, le=5)
    wants_refund: bool


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def demo_pydantic_validation():
    """Shows, with three hand-made inputs, how TicketRecord coerces/validates its fields."""
    section("Task C: Pydantic validation demonstrations")

    rec = TicketRecord(subscriber_id="TP00001", category="billing", urgency="3", wants_refund=False)
    print(f"urgency='3' (string) -> coerced to {rec.urgency!r} ({type(rec.urgency).__name__})")

    try:
        TicketRecord(subscriber_id="TP00001", category="billing", urgency=9, wants_refund=False)
    except ValidationError as exc:
        err = exc.errors()[0]
        print(f"urgency=9 -> ValidationError: field={err['loc']}, type={err['type']}")

    try:
        TicketRecord(subscriber_id="TP00001", category="refund", urgency=3, wants_refund=False)
    except ValidationError as exc:
        err = exc.errors()[0]
        print(f"category='refund' -> ValidationError: field={err['loc']}, type={err['type']}")


def _call_model(messages):
    """Calls the model with a plain {"role", "content"} message list and returns the reply text."""
    return call_llm(messages)


def build_extraction_prompt(parser):
    """Builds the TICKET EXTRACTOR prompt, with parser.get_format_instructions() inserted via .partial()."""
    return ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_SYSTEM_TEMPLATE),
        ("human", "{ticket}"),
    ]).partial(format_instructions=parser.get_format_instructions())


def extract_ticket(prompt, parser, ticket_text):
    """Runs the extraction chain for one ticket. On a parser failure, retries up to MAX_RETRIES times by
    appending the bad reply as an assistant message and a "VALIDATION ERROR: ..." user message so the
    model can correct itself. Never raises: returns (status, attempts, record_or_none, first_failure_kind)
    where status is "valid" or "manual_review", and first_failure_kind (only set if attempt 1 failed) is
    "invalid_json" or "schema_violation"."""
    role_map = {"system": "system", "human": "user", "ai": "assistant"}
    prompt_value = prompt.invoke({"ticket": ticket_text})
    messages = [{"role": role_map[m.type], "content": m.content} for m in prompt_value.to_messages()]

    first_failure_kind = None
    for attempt in range(1, MAX_RETRIES + 2):
        reply = _call_model(messages)
        try:
            record = parser.parse(reply)
            return "valid", attempt, record, first_failure_kind
        except Exception as exc:
            if attempt == 1:
                try:
                    json.loads(reply)
                    first_failure_kind = "schema_violation"
                except Exception:
                    first_failure_kind = "invalid_json"
            if attempt == MAX_RETRIES + 1:
                return "manual_review", attempt, None, first_failure_kind
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"VALIDATION ERROR: {exc}"})


def main():
    demo_pydantic_validation()

    section("Task C: extraction run over all 150 tickets")
    tickets = pd.read_csv(TICKETS_RAW_PATH)
    parser = PydanticOutputParser(pydantic_object=TicketRecord)
    prompt = build_extraction_prompt(parser)

    mock_llm.reset()
    results = []
    for _, row in tickets.iterrows():
        status, attempts, record, failure_kind = extract_ticket(prompt, parser, row["text"])
        results.append({
            "ticket_id": row["ticket_id"], "status": status, "attempts": attempts,
            "record": record, "failure_kind": failure_kind,
        })

    total_logical_calls = sum(r["attempts"] for r in results)
    api_calls = mock_llm._state["calls"]
    valid_first = sum(1 for r in results if r["status"] == "valid" and r["attempts"] == 1)
    valid_retry = sum(1 for r in results if r["status"] == "valid" and r["attempts"] > 1)
    manual_review = sum(1 for r in results if r["status"] == "manual_review")
    invalid_json = sum(1 for r in results if r["failure_kind"] == "invalid_json")
    schema_violation = sum(1 for r in results if r["failure_kind"] == "schema_violation")

    print(f"Valid on first attempt: {valid_first}")
    print(f"Valid after a retry: {valid_retry}")
    print(f"manual_review: {manual_review}")
    print(f"First-attempt failures: {invalid_json + schema_violation} "
          f"(not valid JSON: {invalid_json}, valid JSON but breaks the schema: {schema_violation})")
    print(f"Rate-limited responses recovered: {api_calls - total_logical_calls}")

    valid_records = [r["record"] for r in results if r["record"] is not None]
    n_refund = sum(1 for rec in valid_records if rec.wants_refund)
    n_null_sid = sum(1 for rec in valid_records if rec.subscriber_id is None)
    print(f"Among {len(valid_records)} valid records: wants_refund=True for {n_refund}, "
          f"null subscriber_id for {n_null_sid}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TICKET_EXTRACTIONS_PATH, "w") as f:
        for r in results:
            line = {
                "ticket_id": r["ticket_id"],
                "status": r["status"],
                "attempts": r["attempts"],
                "record": json.loads(r["record"].model_dump_json()) if r["record"] is not None else None,
            }
            f.write(json.dumps(line) + "\n")
    print(f"\nWrote {TICKET_EXTRACTIONS_PATH} ({len(results)} rows)")


if __name__ == "__main__":
    main()
