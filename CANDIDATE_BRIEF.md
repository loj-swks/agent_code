## Setup

No API key and no network access are required.

```bash
pip install -r requirements.txt

python main.py ingest --data-dir ./data
python main.py query "How do I rotate a Helios service token?"
python main.py serve --port 8080
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
HELIOS_MOCK_FAIL_RATE=0.5 python main.py query "..."   # inject 500s
HELIOS_MOCK_LATENCY_MS=800 python main.py query "..."  # inject latency
```

You do not need to modify `mock_llm.py`, and we are not evaluating it — but do
read it, because it documents the API contract your code has to satisfy. If you
have an API key of your own and would rather use it, set `HELIOS_LLM=anthropic`
and `pip install anthropic`.

