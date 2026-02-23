"""ByteDance Doubao model integrations for ReAct agent."""

import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from ..utils import normalize_region


def create_doubao_model(
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    region: Optional[str] = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Create a ByteDance Doubao model using the Volcengine API.

    Args:
        model_name: The model name / endpoint ID (e.g., 'doubao-pro-32k', 'doubao-lite-4k')
        api_key: Volcengine API key (defaults to env var DOUBAO_API_KEY)
        base_url: Custom base URL for API (optional)
        region: Region setting ('prc'/'cn' for China, 'international'/'en' for global)
                Defaults to env var REGION
        **kwargs: Additional model parameters

    Returns:
        Configured ChatOpenAI instance pointing to Volcengine endpoint
    """
    # Get API key from env if not provided
    if api_key is None:
        api_key = os.getenv("DOUBAO_API_KEY")

    # Get region from env if not provided
    if region is None:
        region = os.getenv("REGION")

    # Set base URL based on region if not explicitly provided
    if base_url is None:
        if region:
            normalized_region = normalize_region(region)
            if normalized_region == "prc":
                base_url = "https://ark.cn-beijing.volces.com/api/v3"
            elif normalized_region == "international":
                base_url = "https://ark.ap-southeast.volces.com/api/v3"
        else:
            # Default to China endpoint (Doubao is primarily a China service)
            base_url = "https://ark.cn-beijing.volces.com/api/v3"

    # Create ChatOpenAI configuration (Volcengine is OpenAI-compatible)
    config = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
        **kwargs,
    }

    return ChatOpenAI(**config)
