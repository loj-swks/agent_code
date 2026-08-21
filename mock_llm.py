"""
mock_llm.py

Offline stand-in for the Anthropic Messages API, so this exercise can be run
without network access or an API key.

It is a deliberately faithful imitation of the real thing:

  * same request validation (rejects parameters the current models reject)
  * same response shape (`.content` is a list of typed blocks, `.usage`, `.stop_reason`)
  * same prompt-cache accounting rules (prefix match, minimum cacheable prefix,
    5 minute ephemeral TTL) with state persisted across processes
  * same exception hierarchy and status codes

Answers are produced by deterministic extractive matching against the context
passages in the prompt — there is no model here. That means answer quality is a
direct function of retrieval quality: if the pipeline retrieves the wrong
passages, the answer is wrong or absent, exactly as it would be with a real
model. It will never invent a fact that is not in the context.

Swap it for the real SDK by setting HELIOS_LLM=anthropic.

Optional knobs, for exercising error handling:
    HELIOS_MOCK_FAIL_RATE=0.3     fraction of calls that raise a 529
    HELIOS_MOCK_LATENCY_MS=250    artificial per-call delay
"""

import hashlib
import json
import math
import os
import random
import re
import time

MIN_CACHEABLE_PREFIX_TOKENS = 1024
CACHE_TTL_SECONDS = 300
CACHE_STATE_PATH = os.environ.get("HELIOS_MOCK_CACHE_PATH", ".mock_llm_cache.json")

KNOWN_MODELS = {
    "claude-fable-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
}

# Models on which the fixed thinking-token budget was removed.
MODELS_WITHOUT_BUDGET_TOKENS = {
    "claude-fable-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
}

# Models on which sampling parameters and assistant prefill were removed.
MODELS_WITHOUT_SAMPLING = MODELS_WITHOUT_BUDGET_TOKENS | {
    "claude-opus-4-6",
    "claude-sonnet-4-6",
}

REFUSAL_TEXT = "I don't know based on the Helios docs."

STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "get", "happens", "how", "i", "if", "in", "is", "it", "long",
    "many", "me", "much", "my", "of", "on", "or", "s", "should", "that", "the",
    "their", "them", "there", "they", "this", "to", "until", "up", "was", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your",
}


class APIError(Exception):
    """Base class, mirroring anthropic.APIError."""

    status_code = None

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class APIStatusError(APIError):
    pass


class BadRequestError(APIStatusError):
    status_code = 400


class AuthenticationError(APIStatusError):
    status_code = 401


class NotFoundError(APIStatusError):
    status_code = 404


class RateLimitError(APIStatusError):
    status_code = 429


class InternalServerError(APIStatusError):
    status_code = 500


class APIConnectionError(APIError):
    pass


class ThinkingBlock:
    type = "thinking"

    def __init__(self, thinking=""):
        self.thinking = thinking

    def __repr__(self):
        return "ThinkingBlock(thinking=%r)" % (self.thinking[:40],)


class TextBlock:
    type = "text"

    def __init__(self, text=""):
        self.text = text

    def __repr__(self):
        return "TextBlock(text=%r)" % (self.text[:40],)


class Usage:
    def __init__(self, input_tokens=0, output_tokens=0,
                 cache_creation_input_tokens=0, cache_read_input_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens

    def to_dict(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }

    def __repr__(self):
        return "Usage(%r)" % (self.to_dict(),)


class Message:
    def __init__(self, id, model, content, stop_reason, usage):
        self.id = id
        self.model = model
        self.role = "assistant"
        self.type = "message"
        self.content = content
        self.stop_reason = stop_reason
        self.stop_details = None
        self.usage = usage

    def to_dict(self):
        return {
            "id": self.id,
            "model": self.model,
            "role": self.role,
            "content": [
                {"type": b.type, "text": b.text} if b.type == "text"
                else {"type": b.type, "thinking": b.thinking}
                for b in self.content
            ],
            "stop_reason": self.stop_reason,
            "usage": self.usage.to_dict(),
        }


def estimate_tokens(text):
    """Crude but stable token estimate; the real API would use its tokenizer."""
    return max(1, len(text) // 4)


def _normalize_system(system):
    """The API accepts a string or a list of blocks. Return (text, cached_text)."""
    if system is None:
        return "", ""
    if isinstance(system, str):
        return system, ""

    full = []
    cached = []
    breakpoint_seen = False
    for block in system:
        text = block.get("text", "") if isinstance(block, dict) else str(block)
        full.append(text)
        if not breakpoint_seen:
            cached.append(text)
        if isinstance(block, dict) and block.get("cache_control"):
            breakpoint_seen = True
    if not breakpoint_seen:
        return "\n".join(full), ""
    return "\n".join(full), "\n".join(cached)


def _load_cache_state():
    try:
        with open(CACHE_STATE_PATH, "r") as f:
            state = json.load(f)
    except (IOError, ValueError):
        return {}
    now = time.time()
    return {k: v for k, v in state.items() if now - v < CACHE_TTL_SECONDS}


def _save_cache_state(state):
    try:
        with open(CACHE_STATE_PATH, "w") as f:
            json.dump(state, f)
    except IOError:
        pass


def _account_for_cache(cached_prefix):
    """Return (cache_creation_tokens, cache_read_tokens) for this prefix.

    Mirrors the real rules: a prefix shorter than the minimum never caches, and
    any byte change to the prefix is a miss.
    """
    if not cached_prefix:
        return 0, 0

    prefix_tokens = estimate_tokens(cached_prefix)
    if prefix_tokens < MIN_CACHEABLE_PREFIX_TOKENS:
        return 0, 0

    key = hashlib.sha256(cached_prefix.encode("utf-8")).hexdigest()
    state = _load_cache_state()
    if key in state:
        state[key] = time.time()
        _save_cache_state(state)
        return 0, prefix_tokens

    state[key] = time.time()
    _save_cache_state(state)
    return prefix_tokens, 0


def _flatten_content(content):
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "tool_result":
                parts.append(str(block.get("content", "")))
        else:
            parts.append(str(block))
    return "\n".join(parts)


def _content_tokens(word_set):
    """Weight by token length as a cheap proxy for term rarity."""
    return sum(len(w) for w in word_set)


def _stem(word):
    """Very light singularization so 'logs'/'log' and 'gpus'/'gpu' match."""
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es") and not word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in STOPWORDS]


def _split_question(prompt):
    """Separate the question from the context passages, robustly.

    Looks for an explicit 'Question:' label first; otherwise falls back to the
    last line that looks like a question. Deliberately tolerant of prompt
    format changes, so refactoring the prompt does not break the mock.
    """
    match = None
    for match in re.finditer(r"(?im)^\s*(?:question|query|user question)\s*:\s*(.+)$", prompt):
        pass
    if match:
        question = match.group(1).strip()
        context = prompt[:match.start()]
        return question, context

    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].endswith("?"):
            return lines[i], "\n".join(lines[:i])
    if lines:
        return lines[-1], "\n".join(lines[:-1])
    return "", ""


ANSWER_THRESHOLD = 0.55


def _split_passages(context):
    """Split context into (citation, text) pairs on [n] markers."""
    parts = re.split(r"\[(\d+)\]", context)
    if len(parts) == 1:
        return [(None, context)]

    passages = []
    for i in range(1, len(parts) - 1, 2):
        text = re.sub(r"^\s*\([^)]*\)\s*", "", parts[i + 1])
        passages.append((parts[i], text))
    return passages


def _sentences(text):
    """Split into sentences, undoing hard line wrapping first.

    Source documents are wrapped at ~78 columns, so a single newline is a wrap,
    not a boundary. Treating it as a boundary shreds every sentence.
    """
    text = re.sub(r"\n{2,}", "\v", text)
    text = text.replace("\n", " ")
    out = []
    for paragraph in text.split("\v"):
        # Strip emphasis before splitting: '**...?**' must end a sentence.
        paragraph = paragraph.replace("**", "").replace("__", "")
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        paragraph = re.sub(r"^#+\s*", "", paragraph)
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            sentence = sentence.strip()
            if len(sentence) >= 20:
                out.append(sentence)
    return out


def _candidates(context):
    """Build scoring candidates, each aware of what follows it.

    `followup` matters because the FAQ documents are shaped as a bolded question
    line followed by the answer; the question line is the best keyword match but
    the answer is the next sentence.
    """
    records = []
    for citation, text in _split_passages(context):
        sentences = _sentences(text)
        for i, sentence in enumerate(sentences):
            records.append({
                "sentence": sentence,
                "citation": citation,
                "is_question": sentence.rstrip().endswith("?"),
                "complete": bool(re.search(r"[.!?]$", sentence.rstrip())),
                "followup": [s for s in sentences[i + 1:i + 3] if not s.endswith("?")],
            })
    return records


def _synthesize_answer(prompt):
    """Extractive, deterministic answer generation. Never invents facts."""
    question, context = _split_question(prompt)
    q_words = set(_words(question))
    if not q_words:
        return REFUSAL_TEXT

    records = _candidates(context)
    if not records:
        return REFUSAL_TEXT

    # IDF over the retrieved passages, so ubiquitous terms like "helios" cannot
    # carry a match on their own, and terms absent from the context entirely
    # (the signal that a question is unanswerable) carry the most weight.
    doc_freq = {}
    for record in records:
        record["words"] = set(_words(record["sentence"]))
        for word in record["words"]:
            doc_freq[word] = doc_freq.get(word, 0) + 1
    n_records = len(records)

    def weight(word):
        return math.log((n_records + 1.0) / (doc_freq.get(word, 0) + 0.5))

    q_mass = sum(weight(w) for w in q_words)
    if q_mass <= 0:
        return REFUSAL_TEXT

    scored = []
    for record in records:
        overlap = q_words & record["words"]
        if not overlap:
            continue
        score = sum(weight(w) for w in overlap) / q_mass
        # Prefer focused sentences over sprawling ones.
        score *= 1.0 / (1.0 + 0.003 * max(0, len(record["sentence"]) - 130))
        # Prefer sentences that survived chunking intact.
        if not record["complete"]:
            score *= 0.8
        scored.append((score, record))

    if not scored:
        return REFUSAL_TEXT

    scored.sort(key=lambda t: (-t[0], t[1]["sentence"]))
    best_score, best = scored[0]
    if best_score < ANSWER_THRESHOLD:
        return REFUSAL_TEXT

    # A question line is a locator, not an answer — answer with what follows it.
    if best["is_question"]:
        if not best["followup"]:
            return REFUSAL_TEXT
        body = " ".join(best["followup"])
        return "%s [%s]" % (body.rstrip("."), best["citation"]) if best["citation"] else body

    chosen = [best]
    for score, record in scored[1:]:
        if len(chosen) >= 2:
            break
        if score < best_score * 0.75 or record["is_question"]:
            continue
        # Overlapping chunks yield near-duplicate sentences, and chunk
        # boundaries yield fragments whose words are a subset of the full
        # sentence. Drop both.
        duplicate = False
        for picked in chosen:
            shared = len(record["words"] & picked["words"])
            union = len(record["words"] | picked["words"])
            smaller = min(len(record["words"]), len(picked["words"]))
            if union and shared / float(union) > 0.5:
                duplicate = True
            if smaller and shared / float(smaller) > 0.8:
                duplicate = True
            if duplicate:
                break
        if not duplicate:
            chosen.append(record)

    parts = []
    for record in chosen:
        text = record["sentence"].rstrip(".")
        if record["citation"]:
            parts.append("%s [%s]" % (text, record["citation"]))
        else:
            parts.append(text)
    return ". ".join(parts) + "."


def _maybe_inject_failure():
    fail_rate = float(os.environ.get("HELIOS_MOCK_FAIL_RATE", "0") or 0)
    if fail_rate > 0 and random.random() < fail_rate:
        raise InternalServerError("Overloaded (injected by mock_llm)")

    latency_ms = float(os.environ.get("HELIOS_MOCK_LATENCY_MS", "0") or 0)
    if latency_ms > 0:
        time.sleep(latency_ms / 1000.0)


class _Messages:
    def create(self, model=None, max_tokens=None, messages=None, system=None,
               thinking=None, output_config=None, cache_control=None, **kwargs):
        if model not in KNOWN_MODELS:
            raise NotFoundError(
                "model: %s. Not found. Check the model name and your access." % model
            )

        if max_tokens is None:
            raise BadRequestError("max_tokens: Field required")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise BadRequestError("max_tokens: Input should be a valid integer greater than 0")

        if not messages:
            raise BadRequestError("messages: Field required")
        if messages[0].get("role") != "user":
            raise BadRequestError("messages: first message must use the \"user\" role")
        if messages[-1].get("role") == "assistant":
            raise BadRequestError(
                "messages: final assistant content blocks (prefill) are not supported on "
                "model %s. Use output_config.format or system instructions instead." % model
            )

        for param in ("temperature", "top_p", "top_k"):
            if kwargs.get(param) is not None and model in MODELS_WITHOUT_SAMPLING:
                raise BadRequestError(
                    "%s: Extra inputs are not permitted. Sampling parameters were removed "
                    "on model %s." % (param, model)
                )

        effort = (output_config or {}).get("effort", "high")
        if effort not in ("low", "medium", "high", "xhigh", "max"):
            raise BadRequestError("output_config.effort: unexpected value %r" % effort)

        thinking_enabled = True
        thinking_display = "omitted"
        if thinking is not None:
            t_type = thinking.get("type")
            if t_type not in ("adaptive", "enabled", "disabled"):
                raise BadRequestError("thinking.type: unexpected value %r" % t_type)
            if "budget_tokens" in thinking and model in MODELS_WITHOUT_BUDGET_TOKENS:
                raise BadRequestError(
                    "thinking.budget_tokens: Extra inputs are not permitted. The fixed "
                    "thinking budget was removed on model %s; use "
                    "thinking={\"type\": \"adaptive\"} and output_config.effort instead." % model
                )
            if t_type == "enabled" and "budget_tokens" not in thinking:
                raise BadRequestError("thinking.budget_tokens: Field required when type is \"enabled\"")
            if t_type == "disabled":
                thinking_enabled = False
                if effort in ("xhigh", "max"):
                    raise BadRequestError(
                        "thinking: cannot be disabled at effort %r on model %s" % (effort, model)
                    )
            thinking_display = thinking.get("display", "omitted")

        _maybe_inject_failure()

        system_text, cached_prefix = _normalize_system(system)
        if cache_control and not cached_prefix:
            cached_prefix = system_text

        prompt = _flatten_content(messages[-1].get("content"))
        answer = _synthesize_answer(prompt)

        cache_creation, cache_read = _account_for_cache(cached_prefix)
        uncached_input = estimate_tokens(system_text) + estimate_tokens(prompt)
        uncached_input -= (cache_creation + cache_read)

        content = []
        thinking_tokens = 0
        if thinking_enabled:
            thinking_tokens = 180
            summary = ""
            if thinking_display == "summarized":
                summary = (
                    "Scanning the numbered context passages for sentences that answer the "
                    "question, then citing the passages actually used."
                )
            content.append(ThinkingBlock(thinking=summary))

        stop_reason = "end_turn"
        answer_tokens = estimate_tokens(answer)
        if thinking_tokens + answer_tokens > max_tokens:
            budget_left = max(0, max_tokens - thinking_tokens)
            answer = answer[: budget_left * 4]
            answer_tokens = estimate_tokens(answer) if answer else 0
            stop_reason = "max_tokens"

        content.append(TextBlock(text=answer))

        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        return Message(
            id="msg_mock_%s" % digest,
            model=model,
            content=content,
            stop_reason=stop_reason,
            usage=Usage(
                input_tokens=max(0, uncached_input),
                output_tokens=thinking_tokens + answer_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
            ),
        )

    def count_tokens(self, model=None, messages=None, system=None, **kwargs):
        text = _normalize_system(system)[0]
        for message in messages or []:
            text += _flatten_content(message.get("content"))

        class _Count:
            input_tokens = estimate_tokens(text)

        return _Count()


class MockAnthropic:
    """Duck-typed stand-in for anthropic.Anthropic()."""

    def __init__(self, api_key=None, **kwargs):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "mock")
        self.messages = _Messages()

    def with_options(self, **kwargs):
        return self
