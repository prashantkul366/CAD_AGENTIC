"""LLM client and model selection for CADSmith.

Supports two backends, chosen with the LLM_BACKEND environment variable:

  bedrock   (default) Claude on Amazon Bedrock. Credentials come from
            AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
            and AWS_REGION, or from the standard AWS credential chain.
  anthropic The first-party Claude API, using ANTHROPIC_API_KEY
            (this is what the original paper used).

Model defaults (paper: arXiv:2603.26512):
  Coder / Planner / Refiners : Claude Sonnet 4.5 (claude-sonnet-4-5-20250929), as in the paper
  Validator Judge            : Claude Opus 4.5   (claude-opus-4-5-20251101)

The paper's Judge was Claude Opus 4 (claude-opus-4-20250514). It has reached
end of life on Bedrock, as has its successor Opus 4.1, so the default Judge
is Opus 4.5: the closest Opus still served, and like Opus 4 it does not
think unless asked. This is the one deliberate deviation from the paper.

Override with CODER_MODEL / JUDGE_MODEL. On Bedrock, a bare Anthropic model
ID such as "claude-sonnet-4-5-20250929" is converted to the matching
cross-region inference profile ID, e.g.
"us.anthropic.claude-sonnet-4-5-20250929-v1:0". A value that already
contains "anthropic." (a full Bedrock model ID, inference profile ID or ARN)
is passed through unchanged.
"""

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

PAPER_CODER_MODEL = "claude-sonnet-4-5-20250929"
PAPER_JUDGE_MODEL = "claude-opus-4-20250514"  # retired on Bedrock
DEFAULT_JUDGE_MODEL = "claude-opus-4-5-20251101"


def get_backend() -> str:
    backend = os.getenv("LLM_BACKEND", "bedrock").strip().lower()
    if backend not in ("bedrock", "anthropic"):
        raise ValueError(f"LLM_BACKEND must be 'bedrock' or 'anthropic', got {backend!r}")
    return backend


def _aws_region() -> str:
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"


def _bedrock_geo_prefix(region: str) -> str:
    """Cross-region inference profile prefix for an AWS region."""
    override = os.getenv("BEDROCK_INFERENCE_PREFIX")
    if override is not None:
        return override.strip().rstrip(".")
    if region.startswith("us-gov-"):
        return "us-gov"
    if region.startswith("eu-"):
        return "eu"
    if region.startswith("ap-"):
        return "apac"
    return "us"


def resolve_model(model: str) -> str:
    """Map a model name to the ID the active backend expects."""
    if get_backend() != "bedrock" or "anthropic." in model:
        return model
    prefix = _bedrock_geo_prefix(_aws_region())
    bedrock_id = f"anthropic.{model}-v1:0"
    return f"{prefix}.{bedrock_id}" if prefix else bedrock_id


def coder_model() -> str:
    return resolve_model(os.getenv("CODER_MODEL", PAPER_CODER_MODEL))


def judge_model() -> str:
    return resolve_model(os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL))


def response_text(response) -> str:
    """Text of the first text block in a Messages API response.

    Models that think by default put a thinking block before the text, so
    response.content[0] is not always text.
    """
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Model refused the request (model={response.model})")
    for block in response.content:
        if block.type == "text":
            return block.text
    raise RuntimeError(f"No text block in response (stop_reason={response.stop_reason})")


_client = None


def get_client():
    """Return a shared Messages API client for the configured backend."""
    global _client
    if _client is not None:
        return _client
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "8"))
    if get_backend() == "bedrock":
        _client = anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID") or None,
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
            aws_session_token=os.getenv("AWS_SESSION_TOKEN") or None,
            aws_profile=os.getenv("AWS_PROFILE") or None,
            aws_region=_aws_region(),
            max_retries=max_retries,
        )
    else:
        _client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_retries=max_retries,
        )
    return _client


def describe() -> dict:
    """Backend and resolved model IDs, for recording in experiment configs."""
    info = {
        "backend": get_backend(),
        "coder_model": coder_model(),
        "judge_model": judge_model(),
    }
    if info["backend"] == "bedrock":
        info["aws_region"] = _aws_region()
    return info
