"""Fake-model checks for signed/list message public-text boundaries."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from lobster.utils.llm_response_utils import extract_text_from_llm_response


@pytest.mark.parametrize(
    "message, expected",
    [
        ("  plain response  ", "plain response"),
        (AIMessage(content="plain response"), "plain response"),
        (
            {"content": [{"type": "text", "text": '{"datasets": []}'}]},
            '{"datasets": []}',
        ),
        (
            SimpleNamespace(content=["first", {"type": "text", "text": "second"}]),
            "first\n\nsecond",
        ),
        (
            SimpleNamespace(
                content=[
                    {"reasoningContent": {"reasoningText": {"text": "private"}}},
                    {"text": "answer"},
                ]
            ),
            "answer",
        ),
        (
            AIMessage(
                content=[
                    {
                        "type": "reasoning_content",
                        "reasoning_content": {
                            "text": "private",
                            "signature": "signature",
                        },
                    },
                    {"type": "text", "text": "answer"},
                ]
            ),
            "answer",
        ),
    ],
)
def test_public_text_only(message, expected):
    result = extract_text_from_llm_response(message)
    assert result == expected
    assert isinstance(result, str)


@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "reasoning", "reasoning": "private"}],
        [
            {
                "type": "reasoning_content",
                "reasoning_content": {"text": "private", "signature": "signature"},
            }
        ],
        [{"type": "thinking", "text": "private"}],
        [{"type": "tool_use", "name": "tool", "input": {}}],
        [{"type": "image", "source": {}}],
        [{"type": "text", "text": {"unexpected": "object"}}],
    ],
)
def test_no_text_never_stringifies_reasoning_or_payload(content):
    message = SimpleNamespace(content=content)
    assert extract_text_from_llm_response(message, allow_empty=True) == ""
    with pytest.raises(ValueError, match="Could not extract text"):
        extract_text_from_llm_response(message)


def test_normalized_blocks_preferred_and_raw_fallback_is_safe():
    message = SimpleNamespace(
        content_blocks=[
            {"type": "reasoning", "reasoning": "private"},
            {"type": "text", "text": "normalized"},
        ],
        content=[{"type": "text", "text": "raw"}],
    )
    assert extract_text_from_llm_response(message) == "normalized"

    class FailedNormalization:
        content = [{"type": "text", "text": "raw"}]

        @property
        def content_blocks(self):
            raise ValueError("normalization unavailable")

    assert extract_text_from_llm_response(FailedNormalization()) == "raw"


def test_actual_langchain_aws_parser_and_fake_model_handoff_preserve_history():
    # Pure conversion of a synthetic Converse payload; no SDK request/model call.
    from langchain_aws.chat_models.bedrock_converse import _parse_response

    from lobster.agents.graph import _invoke_and_store

    payload = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "text": "private reasoning",
                                "signature": "test-signature",
                            }
                        }
                    },
                    {"text": '{"datasets": ["GSE_SYNTHETIC"]}'},
                ],
            }
        },
        "usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3},
        "stopReason": "end_turn",
        "metrics": {"latencyMs": 1},
    }
    message = _parse_response(deepcopy(payload))
    assert isinstance(message.content, list)
    before = deepcopy(message.content)
    model = FakeMessagesListChatModel(responses=[message])

    class FakeAgent:
        async def ainvoke(self, inputs, config):
            return {"messages": [await model.ainvoke(inputs["messages"])]}

    response = asyncio.run(_invoke_and_store(FakeAgent(), "test_agent", "task", None))
    assert response == '{"datasets": ["GSE_SYNTHETIC"]}'
    assert "private reasoning" not in response and "test-signature" not in response
    assert message.content == before  # Signed blocks remain in graph history.


def test_reasoning_only_handoff_does_not_publish_private_content():
    from lobster.agents.graph import _invoke_and_store

    message = AIMessage(content=[{"type": "reasoning", "reasoning": "private"}])
    agent = SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": [message]}))
    assert (
        asyncio.run(_invoke_and_store(agent, "test_agent", "task", None))
        == "Agent test_agent returned no text response."
    )


def test_local_client_raw_fallback_does_not_stringify_reasoning():
    from lobster.core.client import AgentClient

    client = AgentClient.__new__(AgentClient)
    client.enable_reasoning = False
    assert (
        client._extract_from_raw_content(
            [{"type": "reasoning_content", "reasoning_content": {"text": "private"}}]
        )
        == ""
    )
    assert (
        client._extract_from_raw_content([{"type": "text", "text": "answer"}])
        == "answer"
    )


def test_minimum_langchain_aws_profiles_drop_unsupported_temperature():
    from langchain_aws.chat_models.bedrock_converse import _get_default_model_profile

    for model in (
        "global.anthropic.claude-sonnet-5-5",
        "global.anthropic.claude-haiku-5-5",
    ):
        assert _get_default_model_profile(model)["temperature"] is False
