"""Translate Lobster thinking presets into each provider's request shape.

Lobster presets (``light``/``standard``/``extended``/``deep``) are stored as a
budget (``{"thinking": {"type": "enabled", "budget_tokens": N}}``). Model
families differ in what they accept:

- Claude 3.x/4.x: ``enabled`` + ``budget_tokens`` (unchanged).
- Claude 5 and later (and Fable): adaptive thinking plus an effort level;
  ``budget_tokens`` is rejected.
- OpenAI direct: ``reasoning={"effort": ...}``.
- Unrecognised model IDs on Claude routes (custom ARNs, application
  profiles) keep the legacy ``enabled`` + ``budget_tokens`` shape.
- GPT on Bedrock and other providers have no verified thinking field, so the
  preset is dropped with a warning instead of sending a field the API rejects.
"""

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CLAUDE_MAJOR = re.compile(r"claude-(?:sonnet|haiku|opus|fable)-(\d+)")
_BEDROCK_ROUTES = {"bedrock", "omics-os"}
_warned: set = set()


def effort_for_budget(budget_tokens: Optional[int]) -> str:
    """Map a preset budget to an effort level (light/standard/extended/deep)."""
    budget = budget_tokens or 0
    if budget <= 1000:
        return "low"
    if budget <= 2000:
        return "medium"
    if budget <= 5000:
        return "high"
    return "max"


def uses_adaptive_thinking(model_id: str) -> bool:
    """Return True for Claude models that need adaptive thinking (5 and later)."""
    model = model_id.lower()
    if "fable" in model:
        return True
    match = _CLAUDE_MAJOR.search(model)
    if match:
        return int(match.group(1)) >= 5
    return False


def translate_thinking(
    provider: str, model_id: str, fields: Dict[str, Any]
) -> Dict[str, Any]:
    """Return provider kwargs with any ``thinking`` preset translated.

    ``fields`` are the extra kwargs the factory passes to the provider. Only a
    Lobster-style ``{"type": "enabled", "budget_tokens": N}`` preset is
    translated; anything else is passed through unchanged.
    """
    thinking = fields.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
        return fields

    out = {k: v for k, v in fields.items() if k != "thinking"}
    effort = effort_for_budget(thinking.get("budget_tokens"))
    model = model_id or ""

    claude_route = provider in ("anthropic", *_BEDROCK_ROUTES)
    # GPT on Bedrock: reasoning_effort is rejected live; no thinking field is known.
    known_non_claude = ".openai." in model.lower() or model.lower().startswith(
        "openai."
    )
    if claude_route and not known_non_claude:
        # Unknown IDs (custom ARNs, application profiles) keep the legacy shape.
        if uses_adaptive_thinking(model):
            request = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
            }
        else:
            request = {"thinking": thinking}
        if provider == "anthropic":
            out.update(request)
        else:
            # Converse takes model-specific fields only inside this mapping.
            extra = dict(out.get("additional_model_request_fields") or {})
            extra.update(request)
            out["additional_model_request_fields"] = extra
        return out

    if provider == "openai":
        out["reasoning"] = {"effort": "high" if effort == "max" else effort}
        return out

    if (provider, model) not in _warned:  # once per route, not once per agent
        _warned.add((provider, model))
        logger.warning(
            "Thinking preset ignored for provider '%s' model '%s': no supported "
            "thinking setting is known for this route.",
            provider,
            model,
        )
    return out
