# AI Engineer Exercise — Helios RAG Pipeline

## Context

`rag_pipeline.py` is a working-ish prototype of a retrieval-augmented generation
pipeline over a small internal documentation set (`data/`). It was written
quickly by someone who has since left. It ingests markdown, builds a TF-IDF
index, retrieves passages for a question, and asks Claude to answer using only
those passages.

It runs. The answers are bad. It is also not something we could deploy.

## Your task

Two things, in whichever order you think is right:

### 1. Make it correct

The script contains **multiple intentional bugs**. Some crash. Some silently
degrade retrieval quality or cost money without failing. We are not telling you
how many there are or where they live.

For each bug you find, we want to hear:

- the symptom you observed,
- the root cause,
- the fix,
- how you would prevent it from coming back.

Finding a bug and explaining it clearly is worth more than silently patching it.

### 2. Make it deployable

Refactor the single script into a proper Python package that you would be
comfortable putting into production. We are deliberately not prescribing the
layout — the design decisions are part of what we are evaluating. Things we will
look for include:

- a sensible module boundary between ingestion, chunking, embedding, retrieval,
  generation, and the service layer
- configuration that is not module-level constants
- an installable package with declared dependencies and an entry point
- tests that would have caught the bugs you fixed
- logging instead of `print`
- error handling around I/O and the model API
- an interface that could be swapped (e.g. a different embedder or vector store)
  without rewriting the pipeline

You do not need to finish every item. Prioritize, and be ready to explain what
you deprioritized and why.

## Setup

No API key and no network access are required.

```bash
pip install -r requirements.txt

python rag_pipeline.py ingest --data-dir ./data
python rag_pipeline.py query "How do I rotate a Helios service token?"
python rag_pipeline.py serve --port 8080
```

### The model is mocked

`mock_llm.py` is an offline stand-in for the Anthropic Messages API, and the
pipeline uses it by default. Treat it as the real API: it applies the same
request validation, returns the same response shape (`.content` is a list of
typed blocks, plus `.usage` and `.stop_reason`), reports prompt-cache accounting
under the real rules, and raises the same exception types with the same status
codes. **A bug you find against the mock is a bug against the real API.**

The one thing it does not do is generate text. Answers are produced by
deterministic extractive matching against the context passages it is given, and
it will never state a fact that is not in those passages. That makes answer
quality a direct readout of retrieval quality: retrieve the wrong passages and
you get a wrong answer or `I don't know based on the Helios docs.`

Two knobs are available if you want to exercise failure handling:

```bash
HELIOS_MOCK_FAIL_RATE=0.5 python rag_pipeline.py query "..."   # inject 500s
HELIOS_MOCK_LATENCY_MS=800 python rag_pipeline.py query "..."  # inject latency
```

You do not need to modify `mock_llm.py`, and we are not evaluating it — but do
read it, because it documents the API contract your code has to satisfy. If you
have an API key of your own and would rather use it, set `HELIOS_LLM=anthropic`
and `pip install anthropic`.

## Ground rules

- Use whatever tools you normally use, including AI assistants. Tell us what you
  used and where you verified its output. Using a coding agent is fine; not
  understanding the code it produced is not.
- Ask questions. Assumptions you make silently will be treated as assumptions.
- Prefer a smaller, correct, well-explained result over a large unfinished one.
- Commit as you go so we can see how you worked.

## Time

90 minutes of hands-on work, then ~20 minutes walking us through it.
