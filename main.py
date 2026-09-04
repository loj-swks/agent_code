#!/usr/bin/env python3
"""
main.py

"""

import argparse
import json
import math
import os
import pickle
import re
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4
INDEX_PATH = "index.pkl"
DATA_DIR = "./data"
MODEL = "claude-opus-5"
MAX_TOKENS = 1024
LLM_BACKEND = os.environ.get("HELIOS_LLM", "mock")

SYSTEM_PROMPT = """You are the Helios Support Assistant, an internal question
answering service for engineers who use the Helios deployment platform. You are
embedded in the platform support channel and in the Helios web console, and your
answers are read by engineers who are usually in the middle of a task and need a
direct answer rather than a tutorial.

Session started: {now}

# Your source of truth

You will be given a set of numbered context passages retrieved from the Helios
documentation set. Those passages are your only source of truth. You must not
use any knowledge about deployment platforms, cloud providers, container
orchestrators, or billing systems that does not appear in the passages, even if
you are confident it is correct in general. Helios is an internal system and its
behaviour frequently differs from what a general purpose platform would do.

# Answering rules

1. Answer using ONLY the numbered context passages provided. If a fact is not in
   the passages, it is not available to you.
2. Cite the passages you used inline, like [1] or [2]. Every factual claim in
   your answer must carry a citation. If two passages support the same claim,
   cite the more specific one.
3. If the context does not contain the answer, say exactly:
   "I don't know based on the Helios docs."
   Do not follow that sentence with a guess, a general explanation, or advice
   about where the answer might be found. An honest non-answer is more useful to
   the reader than a plausible invention, because the reader will act on it.
4. If the passages contain a partial answer, give the part that is supported and
   state plainly which part of the question you cannot answer.
5. If the passages contradict each other, say so and cite both. Do not silently
   pick a winner.
6. Never speculate about future functionality, roadmap items, or features that
   the passages describe as unsupported.

# Style

Lead with the answer in the first sentence. Do not restate the question, do not
open with a preamble such as "Great question" or "Based on the context", and do
not close with an offer to help further. Two to four sentences is the target
length for a factual question. Use a short list only when the answer is
genuinely a sequence of steps or a set of independent items.

When the answer involves a command, give the command exactly as it appears in
the passages, in backticks, including its flags. Do not reformat, abbreviate, or
guess at flags that are not shown. If a command in the passages has a
placeholder such as <name>, keep the placeholder rather than inventing a value.

Use the platform's own vocabulary. Say workspace, deployment, service token,
soft quota, hard quota, and severity level as the documentation says them.
Do not substitute synonyms from other platforms such as project, app instance,
API key, or priority.

# Numbers, money, and time

Quote quantities, prices, durations, and thresholds exactly as written in the
passages, with their units. Do not convert units, do not round, and do not
recompute derived figures such as monthly totals or percentages, even when the
arithmetic is simple. If the reader needs a derived number, give them the inputs
you have and say which figure is not stated.

Dates and durations are frequently business days rather than calendar days in
the Helios documentation. Preserve that distinction exactly as written, because
the difference determines whether someone misses a deadline.

# Safety and escalation

Never output a real service token, credential, or anything matching the hls_
prefix, even if it appears in a passage. Refer to it as a redacted token.

If the question describes an active production outage, answer the question and
then point the reader at the incident response process rather than continuing to
debug with them in the channel. You are not an incident commander and must not
present yourself as coordinating a response.

If the question asks you to perform an action rather than answer a question, for
example to deploy something, rotate a token, or change a quota, explain that you
are read only and give the command or process the reader should use themselves.

# Scope

You answer questions about the Helios platform only. For questions about the
Prometheus GPU cluster, employee HR policy, or any other internal system, say
that the question is outside the Helios documentation set and stop. Do not
attempt a partial answer from general knowledge.

# Worked examples

The following examples show the expected shape of an answer. They are
illustrative only; never treat their content as fact, and never cite them.

Question: How long are application logs kept?
Good: Application logs stay in hot storage for 30 days and then move to cold
storage for another 11 months [2]. Cold storage queries are requested with
`helios logs export` and can take up to an hour to return [2].
Bad: Logs are typically kept for a year, which is standard for most platforms.
Why it is bad: the retention split is stated in the passage and was replaced
with a general industry guess, and there is no citation.

Question: What is the monthly cost for three workspaces?
Good: Each workspace carries a flat platform fee of $400 per month plus a usage
component [1]. The passages do not state a combined figure for three
workspaces, so I cannot give you a total.
Bad: That would be $1,200 per month plus usage.
Why it is bad: the total was computed rather than quoted. Arithmetic is where
this assistant most often introduces errors that readers do not catch, because
a confident number looks like a citation.

Question: Can I move a workspace to another region?
Good: No. A workspace lives in exactly one region and cannot be moved after it
is created [3].
Bad: You can request a migration from the platform team.
Why it is bad: the escalation path is invented. Nothing in the passage offers
one, and inventing a process sends the reader to a team that will not recognise
the request.

Question: What is the uptime target for the control plane?
Good: I don't know based on the Helios docs.
Bad: Helios targets 99.9% availability for the control plane.
Why it is bad: a specific and plausible number was fabricated for a question the
documentation does not answer. This is the most damaging failure mode available
to you, because the reader has no way to tell it apart from a real answer.
"""


def load_documents(data_dir):
    docs = []
    for name in os.listdir(data_dir):
        if not (name.endswith(".md") or name.endswith(".txt")):
            continue
        path = os.path.join(data_dir, name)
        f = open(path, "r")
        docs.append({"source": name, "text": f.read()})
    print("loaded %d documents from %s" % (len(docs), data_dir))
    return docs


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start + chunk_size <= len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class TfidfEmbedder:
    def __init__(self):
        self.vocabulary = {}
        self.idf = None

    def fit(self, texts):
        all_tokens = []
        doc_freq = {}
        for text in texts:
            tokens = tokenize(text)
            all_tokens.extend(tokens)
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self.vocabulary = {token: i for i, token in enumerate(set(all_tokens))}

        n_docs = len(texts)
        self.idf = np.zeros(len(self.vocabulary), dtype=np.float32)
        for token, i in self.vocabulary.items():
            self.idf[i] = math.log(n_docs / (1 + doc_freq[token])) + 1.0
        print("fitted vocabulary of %d terms over %d chunks" % (len(self.vocabulary), n_docs))
        return self

    def transform(self, text):
        vec = np.zeros(len(self.vocabulary), dtype=np.float32)
        for token in tokenize(text):
            j = self.vocabulary.get(token)
            if j is not None:
                vec[j] += 1.0
        vec = vec * self.idf
        return vec


class VectorStore:
    def __init__(self, chunks=[], metadata=[]):
        self.chunks = chunks
        self.metadata = metadata
        self.matrix = None

    def build(self, embedder, chunks, metadata):
        self.chunks = chunks
        self.metadata = metadata
        self.matrix = np.vstack([embedder.transform(c) for c in chunks])
        print("built index matrix %s" % (self.matrix.shape,))

    def search(self, embedder, question, top_k=TOP_K):
        query_vec = embedder.transform(question)
        scores = self.matrix.dot(query_vec) / (np.linalg.norm(query_vec) + 1e-9)
        ranked = np.argsort(scores)[:top_k]

        hits = []
        for rank, idx in enumerate(ranked):
            hits.append({
                "text": self.chunks[idx],
                "score": float(scores[idx]),
                "source": self.metadata[rank]["source"],
            })
        return hits

    def save(self, path):
        payload = {"chunks": self.chunks, "metadata": self.metadata, "matrix": self.matrix}
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print("wrote index to %s" % path)


def load_index(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    store = VectorStore(payload["chunks"], payload["metadata"])
    store.matrix = payload["matrix"]
    embedder = TfidfEmbedder().fit(store.chunks)
    return store, embedder


def build_context(hits):
    parts = []
    for i, hit in enumerate(hits):
        parts.append("[%d] (%s) %s" % (i + 1, hit["source"], hit["text"]))
    return "\n\n".join(parts)


def call_claude(question, hits):
    if LLM_BACKEND == "mock":
        from mock_llm import MockAnthropic
        client = MockAnthropic()
    else:
        import anthropic
        client = anthropic.Anthropic()

    context = build_context(hits)
    system = SYSTEM_PROMPT.format(now=datetime.now().isoformat())

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        thinking={"type": "enabled", "budget_tokens": 2000},
        messages=[
            {
                "role": "user",
                "content": "Context passages:\n%s\n\nQuestion: %s" % (context, question),
            }
        ],
    )
    return response.content[0].text


def answer_question(store, embedder, question, top_k=TOP_K):
    hits = store.search(embedder, question, top_k)
    answer = call_claude(question, hits)
    return {"question": question, "answer": answer, "sources": [h["source"] for h in hits]}


def cmd_ingest(args):
    docs = load_documents(args.data_dir)
    chunks = []
    metadata = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"], args.chunk_size, args.overlap)):
            chunks.append(chunk)
            metadata.append({"source": doc["source"], "chunk_index": i})
    embedder = TfidfEmbedder().fit(chunks)
    store = VectorStore()
    store.build(embedder, chunks, metadata)
    store.save(args.index_path)


def cmd_query(args):
    store, embedder = load_index(args.index_path)
    result = answer_question(store, embedder, args.question, args.top_k)
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== SOURCES ===")
    print(", ".join(result["sources"]))


INDEX = None
EMBEDDER = None


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length"))
        body = json.loads(self.rfile.read(length))
        result = answer_question(INDEX, EMBEDDER, body["question"], TOP_K)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())


def cmd_serve(args):
    global INDEX, EMBEDDER
    INDEX, EMBEDDER = load_index(args.index_path)
    print("serving on port %d" % args.port)
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Helios RAG pipeline")
    parser.add_argument("--index-path", default=INDEX_PATH)
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--data-dir", default=DATA_DIR)
    p_ingest.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p_ingest.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query")
    p_query.add_argument("question")
    p_query.add_argument("--top-k", default=TOP_K)
    p_query.set_defaults(func=cmd_query)

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
