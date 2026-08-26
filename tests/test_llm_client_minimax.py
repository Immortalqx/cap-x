"""Tests for MiniMax M3 client integration.

These tests cover the role-aware payload, response parsing, audit
metadata, and YAML merge behaviour added for CaP-X + MiniMax-M3.

Run with:
    cd /home/lqx/Project/CodeAsPolicy/repos/cap-x
    conda activate cap-x
    pytest tests/test_llm_client_minimax.py -v
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capx.llm import (  # noqa: E402
    MINIMAX_CODE_ROLES,
    MINIMAX_DEFAULT_BASE_URL,
    MINIMAX_MODELS,
    MINIMAX_VDM_ROLES,
    is_minimax_model,
    query_model,
    strip_execute_wrappers,
)
from capx.llm.client import (  # noqa: E402
    MINIMAX_DEFAULT_BASE_URL as _DEFAULT_URL_FROM_CLIENT,
    ModelQueryArgs,
    _resolve_minimax_api_key,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_minimax_models_constant():
    assert MINIMAX_MODELS == ["MiniMax-M3"]


def test_minimax_default_base_url():
    assert MINIMAX_DEFAULT_BASE_URL == "https://api.minimax.io/v1"
    assert _DEFAULT_URL_FROM_CLIENT == MINIMAX_DEFAULT_BASE_URL


def test_role_sets_partitioned():
    """VDM and code roles must be disjoint so thinking mode is unambiguous."""
    assert MINIMAX_VDM_ROLES.isdisjoint(MINIMAX_CODE_ROLES)
    assert "initial_scene_vdm" in MINIMAX_VDM_ROLES
    assert "post_action_vdm" in MINIMAX_VDM_ROLES
    assert "video_differencing" in MINIMAX_VDM_ROLES
    assert "initial_code" in MINIMAX_CODE_ROLES
    assert "finish_regenerate" in MINIMAX_CODE_ROLES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_is_minimax_model():
    assert is_minimax_model("MiniMax-M3") is True
    assert is_minimax_model("google/gemini-3.1-pro-preview") is False
    assert is_minimax_model("openrouter/MiniMax-M3") is False
    assert is_minimax_model("") is False


def test_strip_execute_wrappers():
    assert strip_execute_wrappers("hello") == "hello"
    assert strip_execute_wrappers("<execute>print('x')</execute>") == "print('x')"
    assert strip_execute_wrappers("a<execute>b</execute>c") == "abc"
    assert strip_execute_wrappers(None) is None


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

def test_resolve_minimax_key_from_args():
    args = ModelQueryArgs(model="MiniMax-M3", server_url="x", api_key="sk-test-123")
    assert _resolve_minimax_api_key(args) == "sk-test-123"


def test_resolve_minimax_key_from_env(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-env-456")
    monkeypatch.delattr(ModelQueryArgs, "api_key", raising=False)
    args = ModelQueryArgs(model="MiniMax-M3", server_url="x")
    assert _resolve_minimax_api_key(args) == "sk-env-456"


def test_resolve_minimax_key_missing(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    args = ModelQueryArgs(model="MiniMax-M3", server_url="x")
    with pytest.raises(RuntimeError, match="MINIMAX_API_KEY"):
        _resolve_minimax_api_key(args)


# ---------------------------------------------------------------------------
# Payload assembly (mocked HTTP)
# ---------------------------------------------------------------------------

def _mock_response(payload: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


def test_minimax_code_role_payload(monkeypatch):
    """Code role uses thinking=adaptive + reasoning_split=True + max_completion_tokens."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    captured = {}
    def fake_post(url, headers, data, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json.loads(data)
        return _mock_response({
            "choices": [{"message": {"role": "assistant", "content": "print('hi')"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    with patch("capx.llm.client.requests.post", side_effect=fake_post):
        args = ModelQueryArgs(model="MiniMax-M3", server_url="https://api.minimax.io/v1/chat/completions",
                              max_tokens=512, temperature=0.5)
        requests_log: list[dict] = []
        out = query_model(
            args,
            [{"role": "user", "content": "hi"}],
            request_role="initial_code",
            request_metadata=requests_log,
        )

    assert captured["url"] == "https://api.minimax.io/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    payload = captured["payload"]
    assert payload["model"] == "MiniMax-M3"
    assert payload["temperature"] == 0.5
    assert payload["max_completion_tokens"] == 512
    assert payload["thinking"]["type"] == "adaptive"
    assert payload.get("reasoning_split") is True
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert out["content"] == "print('hi')"
    # audit metadata
    assert len(requests_log) == 1
    meta = requests_log[0]
    assert meta["role"] == "initial_code"
    assert meta["model"] == "MiniMax-M3"
    assert meta["thinking_mode"] == "adaptive"
    assert meta["reasoning_split"] is True
    assert meta["status_code"] == 200
    assert meta["prompt_tokens"] == 10
    assert meta["completion_tokens"] == 5
    # api key never stored
    assert "sk-test" not in json.dumps(meta)


def test_minimax_vdm_role_payload(monkeypatch):
    """VDM role uses thinking=disabled and DOES NOT set reasoning_split."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    captured = {}
    def fake_post(url, headers, data, timeout):
        captured["payload"] = json.loads(data)
        return _mock_response({
            "choices": [{"message": {"role": "assistant", "content": "describe scene"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        })

    with patch("capx.llm.client.requests.post", side_effect=fake_post):
        args = ModelQueryArgs(model="MiniMax-M3", server_url="https://api.minimax.io/v1/chat/completions",
                              max_tokens=256)
        out = query_model(
            args,
            [{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ]}],
            request_role="initial_scene_vdm",
            request_metadata=[],
        )

    payload = captured["payload"]
    assert payload["thinking"]["type"] == "disabled"
    assert "reasoning_split" not in payload
    # image_url preserved as base64
    assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert out["content"] == "describe scene"


def test_strip_execute_wrappers_in_code_role(monkeypatch):
    """Code role output with <execute>...</execute> wrappers should be cleaned."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    def fake_post(*args, **kwargs):
        return _mock_response({
            "choices": [{"message": {"role": "assistant", "content": "<execute>print('x')</execute>"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    with patch("capx.llm.client.requests.post", side_effect=fake_post):
        args = ModelQueryArgs(model="MiniMax-M3", server_url="https://api.minimax.io/v1/chat/completions")
        out = query_model(
            args, [{"role": "user", "content": "x"}],
            request_role="initial_code", request_metadata=[],
        )
    assert out["content"] == "print('x')"
    # VDM does NOT strip; this is implicitly tested in test_minimax_vdm_role_payload
    # (content there has no wrappers to strip anyway).


def test_reasoning_content_captured(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    def fake_post(*args, **kwargs):
        return _mock_response({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "OK",
                    "reasoning_content": "thinking...",
                    "reasoning_details": [{"step": 1}],
                }
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        })

    with patch("capx.llm.client.requests.post", side_effect=fake_post):
        args = ModelQueryArgs(model="MiniMax-M3", server_url="https://api.minimax.io/v1/chat/completions")
        out = query_model(
            args, [{"role": "user", "content": "x"}],
            request_role="initial_code", request_metadata=[],
        )
    assert out["content"] == "OK"
    assert out["reasoning_content"] == "thinking..."
    assert out["reasoning_details"] == [{"step": 1}]


def test_cached_tokens_recorded(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    def fake_post(*args, **kwargs):
        return _mock_response({
            "choices": [{"message": {"role": "assistant", "content": "y"}}],
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 10,
                "total_tokens": 210,
                "prompt_tokens_details": {"cached_tokens": 150},
            },
        })

    with patch("capx.llm.client.requests.post", side_effect=fake_post):
        args = ModelQueryArgs(model="MiniMax-M3", server_url="https://api.minimax.io/v1/chat/completions")
        requests_log = []
        query_model(
            args, [{"role": "user", "content": "x"}],
            request_role="initial_code", request_metadata=requests_log,
        )
    assert requests_log[0]["cached_tokens"] == 150


def test_429_not_retried(monkeypatch):
    """429 (rate limit) must NOT trigger the heavy retry loop."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    call_count = {"n": 0}

    def fake_post(*args, **kwargs):
        call_count["n"] += 1
        return _mock_response({}, status_code=429)

    with patch("capx.llm.client.requests.post", side_effect=fake_post), \
         patch("capx.llm.client.time.sleep") as _sleep:
        args = ModelQueryArgs(model="MiniMax-M3", server_url="https://api.minimax.io/v1/chat/completions")
        with pytest.raises(Exception):
            query_model(
                args, [{"role": "user", "content": "x"}],
                request_role="initial_code", request_metadata=[],
            )
    # 429 must not retry on MiniMax path; only one HTTP attempt expected
    assert call_count["n"] == 1
    # No sleeps on 429 either
    assert _sleep.call_count == 0


def test_503_retries_with_bounded_attempts(monkeypatch):
    """500/502/503/504 should retry up to 3 attempts total on MiniMax."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    call_count = {"n": 0}

    def fake_post(*args, **kwargs):
        call_count["n"] += 1
        return _mock_response({}, status_code=503)

    with patch("capx.llm.client.requests.post", side_effect=fake_post), \
         patch("capx.llm.client.time.sleep"):
        args = ModelQueryArgs(model="MiniMax-M3", server_url="https://api.minimax.io/v1/chat/completions")
        with pytest.raises(Exception):
            query_model(
                args, [{"role": "user", "content": "x"}],
                request_role="initial_code", request_metadata=[],
            )
    # MiniMax: bounded retry — max 3 total attempts (initial + 2 retries).
    assert call_count["n"] <= 3


# ---------------------------------------------------------------------------
# LaunchArgs + YAML merge
# ---------------------------------------------------------------------------

def test_launch_args_has_max_trial_attempts():
    from capx.envs.launch import LaunchArgs
    import dataclasses
    fields = {f.name for f in dataclasses.fields(LaunchArgs)}
    assert "max_trial_attempts" in fields


def test_yaml_merge_overrides_model_when_cli_default():
    """When the user does NOT pass --model, YAML should override the CLI default."""
    from capx.utils.launch_utils import _load_config
    from capx.envs.launch import LaunchArgs

    cfg_path = REPO_ROOT / "env_configs" / "cube_lifting" / "franka_robosuite_cube_lifting_minimax_m3.yaml"
    args = LaunchArgs(config_path=str(cfg_path))
    # args.model is the LaunchArgs default = "google/gemini-3.1-pro-preview"
    assert args.model == "google/gemini-3.1-pro-preview"
    _, merged, _ = _load_config(args)
    # After merge, args.model should be MiniMax-M3 (from YAML)
    assert args.model == "MiniMax-M3"
    assert args.server_url == "https://api.minimax.io/v1/chat/completions"
    assert merged["max_trial_attempts"] == 1


def test_yaml_merge_does_not_override_when_cli_explicit():
    """If the user passes --model explicitly, CLI value wins."""
    from capx.utils.launch_utils import _load_config
    from capx.envs.launch import LaunchArgs

    cfg_path = REPO_ROOT / "env_configs" / "cube_lifting" / "franka_robosuite_cube_lifting_minimax_m3.yaml"
    args = LaunchArgs(config_path=str(cfg_path), model="some-other-model")
    _, _, _ = _load_config(args)
    assert args.model == "some-other-model"


def test_yaml_merge_max_trial_attempts_validation():
    """max_trial_attempts < 1 must raise ValueError."""
    from capx.utils.launch_utils import _load_config
    from capx.envs.launch import LaunchArgs

    cfg_path = REPO_ROOT / "env_configs" / "cube_lifting" / "franka_robosuite_cube_lifting_minimax_m3.yaml"
    args = LaunchArgs(config_path=str(cfg_path), max_trial_attempts=0)
    with pytest.raises(ValueError, match="max-trial-attempts"):
        _load_config(args)


def test_extract_code_strips_execute_wrapper():
    from capx.utils.launch_utils import _extract_code
    content = "<execute>\n```python\nprint('x')\n```\n</execute>"
    blocks = _extract_code(content)
    assert blocks == ["print('x')"]
