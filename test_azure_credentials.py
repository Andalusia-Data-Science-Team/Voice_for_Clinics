"""
# ==============================================================================
#   MedVoice -- Azure Credentials Diagnostic Script
#
#  Tests the 3 Azure values in .env:
#    1. API Key            (AZURE_API_KEY / AI_FOUNDRY_KEY)
#    2. Project Endpoint   (AZURE_PROJECT_ENDPOINT)
#    3. Azure OpenAI Endpoint (AZURE_OPENAI_ENDPOINT / AI_FOUNDRY_ENDPOINT)
#
#  Run from project root:
#    python test_azure_credentials.py
#  Or with a custom .env path:
#    python test_azure_credentials.py --env src/.env
# ==============================================================================
"""

from __future__ import annotations

# Force UTF-8 output on Windows so box-drawing / emoji chars don't crash cp1252
import sys as _sys, io as _io
if hasattr(_sys.stdout, "reconfigure"):
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
else:
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# -- ANSI colours (Windows 10+ supports them) ----------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def ok(msg):  print(f"  {GREEN}[OK]{RESET}  {msg}")
def err(msg): print(f"  {RED}[FAIL]{RESET}  {RED}{msg}{RESET}")
def warn(msg):print(f"  {YELLOW}[WARN]{RESET}  {YELLOW}{msg}{RESET}")
def info(msg):print(f"  {CYAN}[INFO]{RESET}  {DIM}{msg}{RESET}")
def hdr(msg): print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}  {msg}{RESET}\n{'-'*60}")


# -- .env loader (no external deps) --------------------------------------------

def load_env(path: Path) -> dict[str, str]:
    """Parse a .env file and return key->value pairs (no subprocess, no dotenv)."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        result[key] = val
    return result


# -- HTTP helpers (stdlib only) -------------------------------------------------

def http_get(url: str, headers: dict, timeout: int = 15) -> tuple[int, dict | None, str]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            return resp.status, json.loads(body) if body else {}, ""
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return e.code, None, body
    except Exception as exc:
        return 0, None, str(exc)


def http_post(url: str, headers: dict, payload: dict, timeout: int = 30) -> tuple[int, dict | None, str]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            return resp.status, json.loads(body) if body else {}, ""
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            detail = json.loads(body)
        except Exception:
            detail = None
        return e.code, detail, body
    except Exception as exc:
        return 0, None, str(exc)


# ==============================================================================
# TEST 1 -- API Key format check
# ==============================================================================

def test_api_key_format(api_key: str) -> bool:
    hdr("TEST 1 -- API Key (format + presence)")
    if not api_key:
        err("API key is empty -- set AZURE_API_KEY or AI_FOUNDRY_KEY in .env")
        return False
    if len(api_key) < 30:
        warn(f"API key looks short ({len(api_key)} chars) -- double-check it")
    else:
        ok(f"API key present  ({len(api_key)} chars, starts: {api_key[:6]}...{api_key[-4:]})")
    return True


# ==============================================================================
# TEST 2 -- Azure OpenAI endpoint: /models list
# ==============================================================================

def test_openai_models(endpoint: str, api_key: str) -> list[str]:
    hdr("TEST 2 -- Azure OpenAI Endpoint -> GET /models")

    if not endpoint:
        err("AZURE_OPENAI_ENDPOINT is empty")
        return []

    base = endpoint.rstrip("/")
    if "/openai/v1" in base:
        url = f"{base}/models"
    else:
        url = f"{base}/openai/models?api-version=2024-06-01"

    info(f"GET {url}")
    t0 = time.perf_counter()
    status, body, raw_err = http_get(url, headers={"api-key": api_key})
    elapsed = (time.perf_counter() - t0) * 1000

    if status == 200 and body:
        models = [m.get("id", "?") for m in body.get("data", [])]
        ok(f"Connected  ({elapsed:.0f} ms)  --  {len(models)} model(s) available")
        for m in sorted(models):
            info(f"  -> {m}")
        return models
    elif status == 401:
        err(f"HTTP 401 Unauthorized -- API key is wrong or expired")
        info(f"Raw: {raw_err[:300]}")
    elif status == 404:
        err(f"HTTP 404 -- endpoint URL may be incorrect: {url}")
        info(f"Raw: {raw_err[:300]}")
    elif status == 0:
        err(f"Connection failed: {raw_err}")
    else:
        err(f"HTTP {status} -- {raw_err[:300]}")
        if body:
            info(json.dumps(body, indent=2)[:400])
    return []


# ==============================================================================
# TEST 3 -- Azure OpenAI: real chat completion
# ==============================================================================

def test_chat_completion(endpoint: str, api_key: str, model: str) -> bool:
    hdr(f"TEST 3 -- Azure OpenAI -> Chat Completion  (model: {model})")

    if not endpoint:
        err("Endpoint empty -- skipping")
        return False

    base = endpoint.rstrip("/")
    if "/openai/v1" in base:
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/openai/deployments/{model}/chat/completions?api-version=2024-06-01"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 5,
        "temperature": 0,
    }

    info(f"POST {url}")
    t0 = time.perf_counter()
    status, body, raw_err = http_post(url, headers={"api-key": api_key}, payload=payload)
    elapsed = (time.perf_counter() - t0) * 1000

    if status == 200 and body:
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "?")
        usage   = body.get("usage", {})
        ok(f"Completion success  ({elapsed:.0f} ms)")
        ok(f"Model reply: {content.strip()!r}")
        info(f"Tokens used -- prompt: {usage.get('prompt_tokens','?')}  "
             f"completion: {usage.get('completion_tokens','?')}  "
             f"total: {usage.get('total_tokens','?')}")
        return True
    elif status == 404:
        warn(f"HTTP 404 -- deployment '{model}' not found on this endpoint")
        info("Try a different AI_FOUNDRY_DEPLOYMENT value in .env")
        info(f"Raw: {raw_err[:300]}")
    elif status == 401:
        err("HTTP 401 -- API key rejected")
    elif status == 429:
        warn("HTTP 429 -- Rate limited (key works, quota exceeded)")
        return True  # Key is valid, just rate-limited
    elif status == 0:
        err(f"Connection failed: {raw_err}")
    else:
        err(f"HTTP {status}")
        if body:
            info(json.dumps(body, indent=2)[:500])
        else:
            info(raw_err[:300])
    return False


# ==============================================================================
# TEST 4 -- Azure AI Foundry Project Endpoint
# ==============================================================================

def test_foundry_project(project_endpoint: str, api_key: str) -> bool:
    hdr("TEST 4 -- Azure AI Foundry Project Endpoint")

    if not project_endpoint:
        err("AZURE_PROJECT_ENDPOINT is empty")
        return False

    base = project_endpoint.rstrip("/")

    # Try multiple API versions -- Foundry is picky about which version the
    # instance supports. We probe all known versions and report the first one that works.
    api_versions = ["2024-07-01-preview", "2024-05-01-preview", "2025-05-01-preview", "2025-01-01"]
    probes = []
    for ver in api_versions:
        probes.append(("GET", f"{base}/deployments?api-version={ver}", f"Deployments ({ver})"))
    probes.append(("GET", f"{base}/connections?api-version=2024-05-01-preview", "Connections list"))


    any_ok = False
    for method, url, label in probes:
        info(f"{method} {url}")
        t0 = time.perf_counter()
        status, body, raw_err = http_get(url, headers={"api-key": api_key})
        elapsed = (time.perf_counter() - t0) * 1000

        if status == 200 and body:
            items = body.get("value", body.get("data", []))
            ok(f"{label}: {len(items)} item(s)  ({elapsed:.0f} ms)")
            for item in items[:10]:
                name = item.get("name") or item.get("id") or str(item)[:60]
                info(f"  -> {name}")
            any_ok = True
        elif status == 401:
            err(f"{label}: HTTP 401 -- key rejected on project endpoint")
        elif status == 404:
            warn(f"{label}: HTTP 404 -- resource not found (may require different API version)")
            info(f"  Raw: {raw_err[:200]}")
        elif status == 403:
            warn(f"{label}: HTTP 403 -- authenticated but no permission on this resource")
        elif status == 0:
            err(f"{label}: Connection failed -- {raw_err}")
        else:
            warn(f"{label}: HTTP {status}")
            info(f"  Raw: {raw_err[:200]}")

    return any_ok


# ==============================================================================
# TEST 5 -- Embedding support (bonus: useful for future RAG)
# ==============================================================================

def test_embeddings(endpoint: str, api_key: str, models: list[str]) -> None:
    embed_models = [m for m in models if "embed" in m.lower() or "ada" in m.lower()]
    if not embed_models:
        info("No embedding models found -- skipping embedding test")
        return

    hdr(f"TEST 5 -- Embeddings  (model: {embed_models[0]})")
    base = endpoint.rstrip("/")
    url  = f"{base}/embeddings"
    payload = {"model": embed_models[0], "input": "test"}

    t0 = time.perf_counter()
    status, body, raw_err = http_post(url, headers={"api-key": api_key}, payload=payload)
    elapsed = (time.perf_counter() - t0) * 1000

    if status == 200 and body:
        dims = len(body.get("data", [{}])[0].get("embedding", []))
        ok(f"Embeddings working  ({elapsed:.0f} ms)  --  vector dimensions: {dims}")
    else:
        warn(f"HTTP {status} -- embeddings not available or not tested")


# ==============================================================================
# SUMMARY
# ==============================================================================

def print_summary(results: dict[str, bool]) -> None:
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  DIAGNOSTIC SUMMARY{RESET}")
    print(f"{'='*60}")
    all_pass = True
    for label, passed in results.items():
        if passed:
            print(f"  {GREEN}[OK]  PASS{RESET}  {label}")
        else:
            print(f"  {RED}[FAIL]  FAIL{RESET}  {RED}{label}{RESET}")
            all_pass = False
    print(f"{'='*60}")
    if all_pass:
        print(f"\n  {GREEN}{BOLD}All checks passed!{RESET}")
        print(f"  {DIM}Your Azure credentials are valid and all endpoints are reachable.{RESET}")
        print(f"\n  {CYAN}Next step:{RESET} Set LLM_PROVIDER=azure in .env to route LLM calls")
        print(f"  through Azure AI Foundry instead of OpenRouter.\n")
    else:
        print(f"\n  {YELLOW}{BOLD}Some checks failed -- see details above.{RESET}\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    # Enable ANSI on Windows
    if sys.platform == "win32":
        os.system("color")
        import ctypes
        try:
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="MedVoice Azure credential diagnostic")
    parser.add_argument("--env", default="src/.env",
                        help="Path to .env file (default: src/.env)")
    parser.add_argument("--model", default=None,
                        help="Override AI_FOUNDRY_DEPLOYMENT for completion test")
    args = parser.parse_args()

    env_path = Path(args.env)
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  MedVoice - Azure Credentials Diagnostic{RESET}")
    print(f"{'='*60}")

    # -- Load .env --------------------------------------------------------------
    if env_path.exists():
        env = load_env(env_path)
        ok(f"Loaded .env from: {env_path.resolve()}")
    else:
        warn(f".env not found at '{env_path}' -- reading from OS environment")
        env = dict(os.environ)

    api_key          = env.get("AI_FOUNDRY_KEY") or env.get("AZURE_API_KEY", "")
    openai_endpoint  = env.get("AI_FOUNDRY_ENDPOINT") or env.get("AZURE_OPENAI_ENDPOINT", "")
    project_endpoint = env.get("AZURE_PROJECT_ENDPOINT", "")
    deployment       = args.model or env.get("AI_FOUNDRY_DEPLOYMENT", "gpt-4o-mini")

    print(f"\n  {DIM}API Key        : {api_key[:6]}...{api_key[-4:] if api_key else 'MISSING'}{RESET}")
    print(f"  {DIM}OpenAI Endpoint: {openai_endpoint or 'MISSING'}{RESET}")
    print(f"  {DIM}Project Endpoint: {project_endpoint or 'MISSING'}{RESET}")
    print(f"  {DIM}Deployment     : {deployment}{RESET}")

    results: dict[str, bool] = {}

    # -- Run tests --------------------------------------------------------------
    results["API Key present"]                 = test_api_key_format(api_key)
    models = test_openai_models(openai_endpoint, api_key)
    results["Azure OpenAI endpoint reachable"] = len(models) > 0

    if models or deployment:
        test_model = deployment if deployment in models else (models[0] if models else deployment)
        results[f"Chat completion ({test_model})"] = test_chat_completion(
            openai_endpoint, api_key, test_model
        )
    else:
        results["Chat completion"] = False

    results["AI Foundry project endpoint"] = test_foundry_project(project_endpoint, api_key)

    if models:
        test_embeddings(openai_endpoint, api_key, models)

    print_summary(results)


if __name__ == "__main__":
    main()
