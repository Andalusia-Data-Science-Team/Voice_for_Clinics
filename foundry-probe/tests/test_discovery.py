"""
tests/test_discovery.py – Fake-client tests for discovery, profiles, error mapping.
No network, no cost. Python 3.9 compatible.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Helpers ───────────────────────────────────────────────────────────────────
def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_ok_response(content: str = "ok") -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.prompt_tokens = 5
    usage.completion_tokens = 3
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ── Discovery ─────────────────────────────────────────────────────────────────
class TestDiscovery:
    def test_ok_deployment_discovered(self, tmp_path, monkeypatch):
        """A deployment that responds OK shows up in results['ok']."""
        from demo import deployments as D

        D.invalidate_cache()

        async def fake_create(**kwargs):
            return _make_ok_response()

        fake_client = MagicMock()
        fake_client.chat.completions.create = AsyncMock(side_effect=fake_create)
        fake_client.models.list = AsyncMock(return_value=MagicMock(data=[]))

        monkeypatch.chdir(tmp_path)
        os.makedirs("results", exist_ok=True)

        result = run(D.discover(fake_client, "fakekey", extra_names=["my-dep"]))
        ok_names = [d["name"] for d in result["ok"]]
        assert "my-dep" in ok_names

    def test_not_found_classified(self, tmp_path, monkeypatch):
        import openai
        from demo import deployments as D

        D.invalidate_cache()

        async def fake_create(**kwargs):
            raise openai.NotFoundError(
                message="not found", response=MagicMock(status_code=404), body={}
            )

        fake_client = MagicMock()
        fake_client.chat.completions.create = AsyncMock(side_effect=fake_create)
        fake_client.models.list = AsyncMock(return_value=MagicMock(data=[]))

        monkeypatch.chdir(tmp_path)
        os.makedirs("results", exist_ok=True)

        result = run(D.discover(fake_client, "fakekey", extra_names=["ghost-dep"]))
        failed_names = {d["name"]: d["status"] for d in result["failed"]}
        assert "ghost-dep" in failed_names
        assert failed_names["ghost-dep"] == "not_found"


# ── Role resolution ───────────────────────────────────────────────────────────
class TestRoleResolution:
    def test_prefer_luna_for_structuring(self, monkeypatch):
        from demo.deployments import resolve_roles

        monkeypatch.delenv("STRUCTURING_DEPLOYMENT", raising=False)
        ok = [
            {"name": "dep-terra", "supports": {}},
            {"name": "gpt-luna-1", "supports": {}},
            {"name": "dep-mini",  "supports": {}},
        ]
        roles = resolve_roles(ok)
        assert roles["STRUCTURING"] == "gpt-luna-1"

    def test_env_override_honoured_when_working(self, monkeypatch):
        from demo.deployments import resolve_roles

        monkeypatch.setenv("STRUCTURING_DEPLOYMENT", "dep-terra")
        ok = [{"name": "dep-terra", "supports": {}}, {"name": "gpt-luna-1", "supports": {}}]
        roles = resolve_roles(ok)
        assert roles["STRUCTURING"] == "dep-terra"

    def test_env_override_not_working_falls_back(self, monkeypatch):
        from demo.deployments import resolve_roles

        monkeypatch.setenv("STRUCTURING_DEPLOYMENT", "missing-dep")
        ok = [{"name": "gpt-luna-1", "supports": {}}]
        roles = resolve_roles(ok)
        # Should NOT silently use missing-dep
        assert roles["STRUCTURING"] != "missing-dep"


# ── Model profiles ────────────────────────────────────────────────────────────
class TestModelProfiles:
    def test_max_completion_tokens_used_when_supported(self):
        from demo.model_profiles import build_kwargs, _PROFILES

        _PROFILES["test-dep"] = {"max_completion_tokens": True, "max_tokens": False,
                                  "temperature": True, "streaming": False,
                                  "json_object": False, "json_schema_strict": False}
        kw = build_kwargs("test-dep", max_tokens=100, temperature=0.5)
        assert "max_completion_tokens" in kw
        assert "max_tokens" not in kw

    def test_max_tokens_fallback(self):
        from demo.model_profiles import build_kwargs, _PROFILES

        _PROFILES["test-dep2"] = {"max_completion_tokens": False, "max_tokens": True,
                                   "temperature": True, "streaming": False,
                                   "json_object": False, "json_schema_strict": False}
        kw = build_kwargs("test-dep2", max_tokens=100)
        assert "max_tokens" in kw
        assert "max_completion_tokens" not in kw

    def test_json_schema_downgrade_to_object(self):
        from demo.model_profiles import build_kwargs, _PROFILES

        _PROFILES["test-dep3"] = {"max_completion_tokens": True, "max_tokens": True,
                                   "temperature": True, "streaming": False,
                                   "json_object": True, "json_schema_strict": False}
        rf = {"type": "json_schema", "json_schema": {"name": "t", "strict": True, "schema": {}}}
        kw = build_kwargs("test-dep3", max_tokens=50, response_format=rf)
        assert kw["response_format"]["type"] == "json_object"


# ── Security: key never in error output ───────────────────────────────────────
class TestSecurityNoKeyLeak:
    def test_provider_error_no_key(self):
        """map_provider_error must not include any API key."""
        import openai
        import os
        from demo.errors import map_provider_error

        FOUNDRY_API_KEY = os.getenv("FOUNDRY_API_KEY", "testkey123")
        exc = openai.NotFoundError(
            message=f"not found key={FOUNDRY_API_KEY}",
            response=MagicMock(status_code=404),
            body={},
        )
        result = map_provider_error(exc, "test_step", "dep-x")
        body_str = json.dumps(result["body"])
        assert FOUNDRY_API_KEY not in body_str, "API key leaked into error response!"
        assert result["status_code"] == 502

    def test_redact_strips_key(self, monkeypatch):
        import os
        from demo.errors import redact

        fake_key = "FakeKey-1234567890abcdefXYZ"
        monkeypatch.setenv("FOUNDRY_API_KEY", fake_key)
        text = f"Error: key={fake_key} was rejected"
        redacted = redact(text)
        assert fake_key not in redacted
        assert "***" in redacted

    def test_no_key_in_not_found_message(self):
        """Verify NotFoundError mapped to generic message only."""
        import openai
        from demo.errors import map_provider_error

        exc = openai.NotFoundError(
            message="DeploymentNotFound",
            response=MagicMock(status_code=404),
            body={},
        )
        result = map_provider_error(exc, "structuring", "dep-luna")
        assert result["body"]["message"] == "Deployment not found on the Azure resource."
        assert result["status_code"] == 502
