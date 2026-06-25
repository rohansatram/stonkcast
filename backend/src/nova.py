"""
Thin Amazon Nova (Bedrock) client. Reads credentials from the environment / a
backend/.env file:

    AWS_BEARER_TOKEN_BEDROCK=<your Bedrock API key>
    AWS_REGION=eu-central-1

Exposes converse(): send messages, get back (text, usage) where usage carries
token counts so the agent can report tokens and estimated cost (the brief
requires both).
"""

import os
from functools import lru_cache
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_REGION = os.getenv("AWS_REGION", "eu-central-1")
# eu-central-1 serves Nova via the EU cross-region inference profiles.
DEFAULT_MODEL = os.getenv("NOVA_MODEL", "eu.amazon.nova-lite-v1:0")

# USD per token (Bedrock Nova list prices, input / output).
PRICING = {
    "nova-micro": (0.035e-6, 0.14e-6),
    "nova-lite": (0.06e-6, 0.24e-6),
    "nova-pro": (0.80e-6, 3.20e-6),
    "nova-premier": (2.50e-6, 12.50e-6),
}


@lru_cache(maxsize=4)
def _client(region: str):
    return boto3.client("bedrock-runtime", region_name=region)


def _pricing_key(model_id: str) -> str:
    """Resolve a model id to a PRICING family, or fail loud. A silent default would
    show a wrong cost on the counter the judges compare across entries."""
    key = next((name for name in PRICING if name in model_id), None)
    if key is None:
        raise ValueError(f"No pricing entry matches model id {model_id!r}; known families: {list(PRICING)}")
    return key


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    input_cost_per_token, output_cost_per_token = PRICING[_pricing_key(model_id)]
    return round(input_tokens * input_cost_per_token + output_tokens * output_cost_per_token, 6)


# Fail at import time, not mid-demo, if NOVA_MODEL is misconfigured.
_pricing_key(DEFAULT_MODEL)


def converse(user_text: str, system: str | None = None, model_id: str = DEFAULT_MODEL,
             region: str = DEFAULT_REGION, max_tokens: int = 1200, temperature: float = 0.05) -> tuple[str, dict]:
    """Single-turn Converse call. Returns (text, usage) where usage =
    {input_tokens, output_tokens, total_tokens, cost_usd, model}."""
    request = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": user_text}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        request["system"] = [{"text": system}]

    response = _client(region).converse(**request)
    reply_text = response["output"]["message"]["content"][0]["text"]
    raw_usage = response["usage"]
    usage = {
        "input_tokens": raw_usage["inputTokens"],
        "output_tokens": raw_usage["outputTokens"],
        "total_tokens": raw_usage["totalTokens"],
        "cost_usd": estimate_cost(model_id, raw_usage["inputTokens"], raw_usage["outputTokens"]),
        "model": model_id,
    }
    return reply_text, usage


if __name__ == "__main__":
    print(f"Region: {DEFAULT_REGION} | Model: {DEFAULT_MODEL}")
    if not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        print("ERROR: AWS_BEARER_TOKEN_BEDROCK not set. Put it in backend/.env")
        raise SystemExit(1)
    text, usage = converse("Reply with exactly: NOVA OK")
    print("Response:", text.strip())
    print("Usage:", usage)
