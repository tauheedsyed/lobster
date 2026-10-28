"""Opt-in Bedrock prompt caching without changing controlled inference clients."""

from langchain_aws import ChatBedrockConverse


class CachingChatBedrockConverse(ChatBedrockConverse):
    """Request default cache points; retain explicit caller cache control."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.setdefault("cache_control", {"type": "ephemeral"})
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.setdefault("cache_control", {"type": "ephemeral"})
        yield from super()._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
