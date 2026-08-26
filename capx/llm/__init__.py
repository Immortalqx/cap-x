"""LLM client module for querying language models.

This module provides utilities for querying various LLM providers
(OpenAI, Claude, open-source models, OpenRouter, MiniMax) with support
for streaming, ensemble queries, and backward-compatible aliases.
"""

from capx.llm.client import (
    CLAUDE_MODELS,
    ENSEMBLE_CONFIGS,
    GPT_MODELS,
    MINIMAX_CODE_ROLES,
    MINIMAX_DEFAULT_BASE_URL,
    MINIMAX_MODELS,
    MINIMAX_VDM_ROLES,
    OPENROUTER_MODELS,
    OPENROUTER_SERVER_URL,
    OSS_MODELS,
    VLM_MODELS,
    ModelQueryArgs,
    _completions_to_responses_convert_prompt,
    collapse_text_image_inputs,
    is_minimax_model,
    is_openrouter_model,
    query_model,
    query_model_ensemble,
    query_model_streaming,
    query_single_model_ensemble,
    strip_execute_wrappers,
)

__all__ = [
    "CLAUDE_MODELS",
    "ENSEMBLE_CONFIGS",
    "GPT_MODELS",
    "MINIMAX_CODE_ROLES",
    "MINIMAX_DEFAULT_BASE_URL",
    "MINIMAX_MODELS",
    "MINIMAX_VDM_ROLES",
    "OPENROUTER_MODELS",
    "OPENROUTER_SERVER_URL",
    "OSS_MODELS",
    "VLM_MODELS",
    "ModelQueryArgs",
    "_completions_to_responses_convert_prompt",
    "collapse_text_image_inputs",
    "is_minimax_model",
    "is_openrouter_model",
    "query_model",
    "query_model_ensemble",
    "query_model_streaming",
    "query_single_model_ensemble",
    "strip_execute_wrappers",
]
