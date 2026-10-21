"""Shared ChatBedrockConverse construction for direct-AWS and gateway paths."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def build_bedrock_converse(
    *,
    model_id: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    client: Any = None,
    bedrock_client: Any = None,
    region_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    config: Any = None,
    additional_model_request_fields: Optional[dict] = None,
    **kwargs: Any,
) -> Any:
    """Build a ChatBedrockConverse instance.

    Two modes:
    - Gateway path: pass `client` (runtime) + `bedrock_client` (control stub)
    - Direct AWS path: pass AWS credentials + region

    ``max_tokens`` is an optional output-token limit. When it is ``None`` (the
    default) no limit is sent and Bedrock applies the model's own maximum; an
    explicit value is forwarded unchanged.

    Returns:
        ChatBedrockConverse instance with full tool calling support
    """
    from langchain_aws import ChatBedrockConverse

    params: dict[str, Any] = {"model_id": model_id}

    # Send a temperature only when set; newer models accept only their default.
    if temperature is not None:
        params["temperature"] = temperature

    # Send an output-token limit only when the caller asks for one. Without it
    # Bedrock applies the model's own maximum, whereas a built-in default would
    # silently cap every call (and sit below extended-thinking budgets).
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    # Gateway path: inject pre-built clients
    if client is not None:
        params["client"] = client
    if bedrock_client is not None:
        params["bedrock_client"] = bedrock_client

    # Direct AWS path: pass credentials
    if region_name:
        params["region_name"] = region_name
    if aws_access_key_id:
        params["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key:
        params["aws_secret_access_key"] = aws_secret_access_key

    # Extended thinking and other model-specific fields
    if additional_model_request_fields:
        params["additional_model_request_fields"] = additional_model_request_fields

    # Boto config for timeouts
    if config:
        params["config"] = config

    # Pass through remaining kwargs
    params.update(kwargs)

    logger.debug(
        f"Building ChatBedrockConverse: model={model_id}, gateway={client is not None}"
    )

    return ChatBedrockConverse(**params)
