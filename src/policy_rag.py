"""policy_rag.py -- Part D task D: chunking, Chroma vector store, retrieval, evaluation and the grounded
policy-assistant answer() function.
Run:  python src/policy_rag.py
Reads data/policies/*.txt and data/raw/policy_eval.csv. Writes outputs/rag_eval.csv.
"""
import os
import re

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

import constants  # noqa: F401
import mock_llm
from constants import OUTPUT_DIR, POLICIES_DIR, POLICY_EVAL_PATH, RAG_EVAL_PATH
from llm_client import call_llm

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # served via Chroma's built-in ONNX runtime (no PyTorch needed)

GROUNDED_SYSTEM = (
    "POLICY ASSISTANT. Answer the question using only the context below. If the context does not contain "
    "the answer, say you do not have enough information to answer."
)
UNGROUNDED_SYSTEM = "POLICY ASSISTANT. Answer the question."
FALLBACK_ANSWER = "I don't have enough information to answer that."


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def chunk_text(text, size=400, overlap=60):
    """Splits text into overlapping chunks. Chunks start at 0, size-overlap, 2*(size-overlap), ...; each
    chunk is text[start:start+size]. A start position is only used while start + overlap < len(text), so
    no chunk consists solely of text already covered by the previous chunk."""
    step = size - overlap
    chunks = []
    start = 0
    while start + overlap < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks


def load_policy_texts():
    texts = {}
    for fname in sorted(os.listdir(POLICIES_DIR)):
        with open(os.path.join(POLICIES_DIR, fname), encoding="utf-8") as f:
            texts[fname] = f.read()
    return texts


def report_chunk_counts(policy_texts, size=400, overlap=60):
    section(f"Task D(a): chunk counts (size={size}, overlap={overlap})")
    total = 0
    for fname, text in policy_texts.items():
        n = len(chunk_text(text, size, overlap))
        total += n
        print(f"  {fname}: {n} chunks")
    print(f"Total: {total} chunks")


def build_collection(policy_texts, name, size=400, overlap=60):
    """Builds an in-memory Chroma collection (cosine distance) of the policy chunks, embedded with
    Chroma's built-in ONNX MiniLM embedding function (no PyTorch dependency)."""
    client = chromadb.Client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(
        name=name,
        embedding_function=embedding_functions.ONNXMiniLM_L6_V2(),
        metadata={"hnsw:space": "cosine"},
    )
    ids, documents, metadatas = [], [], []
    for fname, text in policy_texts.items():
        for i, chunk in enumerate(chunk_text(text, size, overlap)):
            ids.append(f"{fname}-{i}")
            documents.append(chunk)
            metadatas.append({"source": fname, "chunk": i})
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


def retrieve(collection, query, k=3):
    """Returns the top-k chunks for query as a list of {"source", "similarity", "text"} dicts
    (similarity = 1 - cosine distance)."""
    result = collection.query(query_texts=[query], n_results=k)
    out = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        out.append({"source": meta["source"], "similarity": 1 - dist, "text": doc})
    return out


def keyword_baseline(question, policy_texts):
    """A lower-cased word-overlap baseline: picks the policy file sharing the most non-stopword words
    with the question, breaking ties alphabetically."""
    q_words = set(re.findall(r"[a-z]+", question.lower())) - mock_llm.STOP
    best_file, best_count = None, -1
    for fname in sorted(policy_texts):
        f_words = set(re.findall(r"[a-z]+", policy_texts[fname].lower())) - mock_llm.STOP
        count = len(q_words & f_words)
        if count > best_count:
            best_file, best_count = fname, count
    return best_file


def build_context_block(chunks):
    return "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)


def grounded_answer(collection, question, threshold):
    """Returns (answer, sources). Below the similarity threshold, returns the fallback sentence without
    calling the model; otherwise sends a grounded system+user message using only the retrieved context."""
    chunks = retrieve(collection, question, k=3)
    top1_similarity = chunks[0]["similarity"] if chunks else -1
    if top1_similarity < threshold:
        return FALLBACK_ANSWER, []
    user_message = f"CONTEXT:\n{build_context_block(chunks)}\n\nQUESTION:\n{question}"
    reply = call_llm([{"role": "system", "content": GROUNDED_SYSTEM}, {"role": "user", "content": user_message}])
    return reply, [c["source"] for c in chunks]


def ungrounded_answer(question):
    reply = call_llm([{"role": "system", "content": UNGROUNDED_SYSTEM}, {"role": "user", "content": question}])
    return reply


def evaluate_retrieval(collection, eval_df, policy_texts):
    section("Task D(c): retrieval evaluation")
    answerable = eval_df[eval_df["expected_file"].notna() & (eval_df["expected_file"] != "")]
    unanswerable = eval_df[~eval_df.index.isin(answerable.index)]

    top1_hits, top3_hits, top1_sims = 0, 0, []
    baseline_hits = 0
    for _, row in answerable.iterrows():
        chunks = retrieve(collection, row["question"], k=3)
        sources = [c["source"] for c in chunks]
        top1_sims.append(chunks[0]["similarity"])
        if row["expected_file"] == sources[0]:
            top1_hits += 1
        if row["expected_file"] in sources:
            top3_hits += 1
        if keyword_baseline(row["question"], policy_texts) == row["expected_file"]:
            baseline_hits += 1

    unanswerable_sims = [retrieve(collection, row["question"], k=3)[0]["similarity"] for _, row in unanswerable.iterrows()]

    print(f"Top-3 file hits: {top3_hits} of {len(answerable)}")
    print(f"Top-1 file hits: {top1_hits} of {len(answerable)}")
    print(f"Answerable top-1 similarity range: {min(top1_sims):.3f} to {max(top1_sims):.3f} "
          f"(mean {sum(top1_sims) / len(top1_sims):.3f})")
    print(f"Unanswerable top-1 similarities: {[round(s, 3) for s in unanswerable_sims]}")
    print(f"Keyword baseline top-1 accuracy: {baseline_hits} of {len(answerable)}")
    return top1_sims, unanswerable_sims


def main():
    policy_texts = load_policy_texts()
    report_chunk_counts(policy_texts, size=400, overlap=60)

    collection = build_collection(policy_texts, "policies_400_60", size=400, overlap=60)
    eval_df = pd.read_csv(POLICY_EVAL_PATH)

    answerable_sims, unanswerable_sims = evaluate_retrieval(collection, eval_df, policy_texts)
    threshold = (min(answerable_sims) + max(unanswerable_sims)) / 2
    section("Task D(d): threshold calibration")
    print(f"Chosen rejection threshold: {threshold:.3f} "
          f"(midpoint of lowest answerable {min(answerable_sims):.3f} and highest unanswerable {max(unanswerable_sims):.3f})")
    print(f"Margin between the two ranges: {min(answerable_sims) - max(unanswerable_sims):.3f}")

    section("Task D(d): grounded vs. ungrounded answers over all 16 questions")
    mock_llm.reset()
    rows = []
    for _, row in eval_df.iterrows():
        chunks = retrieve(collection, row["question"], k=3)
        top1 = chunks[0]
        hit_at_3 = bool(row["expected_file"]) and row["expected_file"] in [c["source"] for c in chunks]
        g_answer, _sources = grounded_answer(collection, row["question"], threshold)
        u_answer = ungrounded_answer(row["question"])
        rows.append({
            "question": row["question"], "expected_file": row["expected_file"],
            "top1_source": top1["source"], "top1_similarity": round(top1["similarity"], 4),
            "hit_at_3": hit_at_3, "grounded_answer": g_answer, "ungrounded_answer": u_answer,
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(RAG_EVAL_PATH, index=False)
    print(f"Wrote {RAG_EVAL_PATH} ({len(rows)} rows)")

    section("Task D(e): chunk-size comparison (size=200, overlap=30)")
    report_chunk_counts(policy_texts, size=200, overlap=30)
    small_collection = build_collection(policy_texts, "policies_200_30", size=200, overlap=30)
    answerable = eval_df[eval_df["expected_file"].notna() & (eval_df["expected_file"] != "")]
    fact_in_top3, grounded_correct = 0, 0
    for _, row in answerable.iterrows():
        chunks = retrieve(small_collection, row["question"], k=3)
        if any(str(row["expected_fact"]).lower() in c["text"].lower() for c in chunks):
            fact_in_top3 += 1
        g_answer, _ = grounded_answer(small_collection, row["question"], threshold)
        if str(row["expected_fact"]).lower() in g_answer.lower():
            grounded_correct += 1
    print(f"Expected fact in top-3 chunks: {fact_in_top3} of {len(answerable)}")
    print(f"Grounded answers containing the expected fact: {grounded_correct} of {len(answerable)}")


if __name__ == "__main__":
    main()
