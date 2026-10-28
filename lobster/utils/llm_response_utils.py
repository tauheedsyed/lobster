"""Provider-agnostic public text extraction from LLM message content.

Reasoning, signatures, tool calls, images and unknown blocks are not answer text.
The original message and its signed history are never rewritten.
"""

from typing import Any

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def _text_parts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content] if content.strip() else []
    if not isinstance(content, list):
        return []
    parts = []
    for block in content:
        if isinstance(block, str):
            if block.strip():
                parts.append(block)
        elif isinstance(block, dict):
            # LangChain text blocks, plus native Converse {"text": ...} blocks.
            if block.get("type") in (None, "text"):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
    return parts


def extract_text_from_llm_response(response: Any, *, allow_empty: bool = False) -> str:
    """Extract only public text from string/list or LangChain/dict messages.

    Normalized content_blocks are preferred, with a safe raw-content fallback.
    Reasoning-only/unknown-block messages raise ValueError unless allow_empty is
    requested; their Python representation is never fabricated as an answer.
    """
    try:
        blocks = getattr(response, "content_blocks", None)
        parts = _text_parts(blocks)
    except Exception as exc:
        logger.debug(
            "content_blocks extraction failed: %s; using raw content",
            type(exc).__name__,
        )
        parts = []
    if not parts:
        if isinstance(response, dict):
            content = response.get("content", "")
        elif hasattr(response, "content"):
            content = response.content
        else:
            content = response
        parts = _text_parts(content)
        # Preserve the prior simple-object fallback, but never stringify structured
        # blocks, absent content, or an entire message object's representation.
        if not parts and isinstance(content, (int, float, bool)):
            parts = [str(content)]
    text = "\n\n".join(parts).strip()
    if text or allow_empty:
        return text
    raise ValueError("Could not extract text from LLM response")
