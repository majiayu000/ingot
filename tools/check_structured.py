"""Structured-output check against a running `ingot serve`.

Sends 20 varied prompts with `response_format` (json_object and json_schema,
greedy and sampled) plus two Anthropic `output_config.format` requests, and
checks that every answer parses and validates against its schema.

    uv run --with jsonschema tools/check_structured.py http://127.0.0.1:8080
"""

import json
import sys
import time
import urllib.request

import jsonschema


def obj(props, required=None):
    return {
        "type": "object",
        "properties": props,
        "required": list(props) if required is None else required,
        "additionalProperties": False,
    }


S = {"type": "string"}
I = {"type": "integer"}
N = {"type": "number"}
B = {"type": "boolean"}

CASES = [
    ("Give me a JSON object describing a cat with name and age.", None, 0),
    ("List three fruits as JSON with a key 'fruits'.", None, 0.7),
    ("Return JSON with the capital of France and its population.", None, 0),
    (
        "Extract the person: 'Alice Smith is 34 and lives in Berlin.'",
        obj({"name": S, "age": I, "city": S}),
        0,
    ),
    (
        "Classify the sentiment of: 'This product broke after one day.'",
        obj({"sentiment": {"enum": ["positive", "neutral", "negative"]}, "confidence": N}),
        0,
    ),
    (
        "Give two recipe ideas with ingredients.",
        obj(
            {
                "recipes": {
                    "type": "array",
                    "items": obj({"title": S, "ingredients": {"type": "array", "items": S}}),
                    "minItems": 2,
                    "maxItems": 2,
                }
            }
        ),
        0.8,
    ),
    (
        "Is 17 a prime number? Answer with reasoning.",
        obj({"answer": B, "reasoning": S}),
        0,
    ),
    (
        "Describe a book you like.",
        obj({"title": S, "author": S, "year": I, "genres": {"type": "array", "items": S}}),
        0.6,
    ),
    (
        "Convert 3 miles to kilometers.",
        obj({"miles": N, "kilometers": N}),
        0,
    ),
    (
        "Create a user profile; middle name is unknown.",
        obj({"first": S, "middle": {"type": ["string", "null"]}, "last": S}),
        0,
    ),
    (
        "Make a todo list for moving house with priorities.",
        obj(
            {
                "items": {
                    "type": "array",
                    "items": obj({"task": S, "priority": {"enum": ["low", "medium", "high"]}, "done": B}),
                }
            }
        ),
        0.9,
    ),
    (
        "Translate 'good morning' into Chinese, Japanese and French.",
        obj({"zh": S, "ja": S, "fr": S}),
        0,
    ),
    (
        "Write a haiku about rain.",
        obj({"lines": {"type": "array", "items": S, "minItems": 3, "maxItems": 3}}),
        1.0,
    ),
    (
        "Describe an org chart: a CEO with two reports, one of whom has a report.",
        {
            "$defs": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": S,
                        "reports": {"type": "array", "items": {"$ref": "#/$defs/person"}},
                    },
                    "required": ["name", "reports"],
                    "additionalProperties": False,
                }
            },
            "$ref": "#/$defs/person",
        },
        0,
    ),
    (
        "Give word counts for the sentence 'the cat and the hat'.",
        {"type": "object", "additionalProperties": I},
        0,
    ),
    (
        "Pick a random HTTP status code and explain it.",
        obj({"code": I, "meaning": S, "retryable": B}),
        1.2,
    ),
    (
        "What is the weather like on Mars? Optional fields allowed.",
        obj({"summary": S, "temperature_c": N, "dust_storm": B}, required=["summary"]),
        0.5,
    ),
    (
        "Return the first five square numbers.",
        {"type": "array", "items": I, "minItems": 5, "maxItems": 5},
        0,
    ),
    (
        'Write a JSON string escaping test: a line with "quotes", a backslash \\ and a newline.',
        obj({"text": S}),
        0,
    ),
    (
        "Rate three programming languages from 1 to 10.",
        obj(
            {
                "ratings": {
                    "type": "array",
                    "items": obj({"language": S, "score": {"anyOf": [I, {"type": "null"}]}}),
                }
            }
        ),
        0.7,
    ),
]


def post(base, path, body):
    req = urllib.request.Request(
        base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    base = sys.argv[1].rstrip("/")
    failures = 0
    rates = []
    for i, (prompt, schema, temperature) in enumerate(CASES):
        if schema is None:
            fmt = {"type": "json_object"}
        else:
            fmt = {"type": "json_schema", "json_schema": {"name": f"case{i}", "strict": True, "schema": schema}}
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "response_format": fmt,
            "temperature": temperature,
            "seed": i,
            "max_tokens": 600,
        }
        start = time.time()
        r = post(base, "/v1/chat/completions", body)
        elapsed = time.time() - start
        choice = r["choices"][0]
        text = choice["message"]["content"]
        tokens = r["usage"]["completion_tokens"]
        rates.append(tokens / elapsed)
        try:
            value = json.loads(text)
            jsonschema.validate(value, schema or {"type": "object"})
            ok = choice["finish_reason"] == "stop"
            status = "ok" if ok else f"finish={choice['finish_reason']}"
        except Exception as e:  # report and count every failure
            ok, status = False, f"INVALID: {e}"
        failures += not ok
        print(f"[{i:2}] t={temperature} {tokens:3} tok {tokens / elapsed:5.1f} tok/s {status}: {text[:110]!r}")
    for i, schema in enumerate([obj({"city": S, "country": S}), obj({"steps": {"type": "array", "items": S}})]):
        body = {
            "max_tokens": 400,
            "messages": [{"role": "user", "content": ["Name a city and its country.", "How do I boil an egg?"][i]}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        r = post(base, "/v1/messages", body)
        text = r["content"][0]["text"]
        try:
            jsonschema.validate(json.loads(text), schema)
            status = "ok" if r["stop_reason"] == "end_turn" else r["stop_reason"]
        except Exception as e:
            status = f"INVALID: {e}"
        failures += status != "ok"
        print(f"[anthropic {i}] {status}: {text[:110]!r}")
    print(f"{len(CASES) + 2 - failures}/{len(CASES) + 2} valid; mean {sum(rates) / len(rates):.1f} tok/s (incl. prefill)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
