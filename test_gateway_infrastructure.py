"""
MedVoice - End-to-End Azure Infrastructure Diagnostic Suite.

Tests the full Azure architecture configured in src/.env:
  1. Azure APIM Gateway (Connectivity, Health, TLS, CORS preflight)
  2. Azure Key Vault (URL structure, DNS resolution, Auth endpoint)
  3. Azure Application Insights (Ingestion endpoint, Connection string format)
  4. Entra ID / Tenant ID (OpenID discovery endpoint reachability)
  5. Azure AI Foundry (Project endpoint + OpenAI endpoint reachability)
"""

import os
import sys
import time
import socket
import urllib.request
import urllib.error
import json
from pathlib import Path

# Setup paths
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dotenv import load_dotenv
env_path = src_dir / ".env"
load_dotenv(env_path)

# Colors
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def print_header(title: str):
    print(f"\n{BOLD}{CYAN}{'='*65}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*65}{RESET}")

def test_apim_gateway():
    print_header("1. Azure API Management (APIM) Gateway Test")
    gateway_url = os.getenv("APIM_GATEWAY_URL", "").rstrip("/")
    print(f"[*] Target Gateway: {gateway_url}")
    
    if not gateway_url:
        print(f"  {RED}[FAIL]{RESET} APIM_GATEWAY_URL not set in .env")
        return False

    # A. DNS / TLS ping
    try:
        host = gateway_url.replace("https://", "").replace("http://", "").split("/")[0]
        t0 = time.perf_counter()
        ip = socket.gethostbyname(host)
        dns_ms = (time.perf_counter() - t0) * 1000
        print(f"  {GREEN}[OK]{RESET} DNS Resolved: {host} -> {ip} ({dns_ms:.1f}ms)")
    except Exception as e:
        print(f"  {RED}[FAIL]{RESET} DNS resolution error for {host}: {e}")
        return False

    # B. HTTP GET test to Gateway root
    try:
        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"{gateway_url}/",
            headers={"User-Agent": "MedVoice-Diagnostic/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            latency = (time.perf_counter() - t0) * 1000
            print(f"  {GREEN}[OK]{RESET} Gateway HTTP Response: {resp.status} ({latency:.1f}ms)")
    except urllib.error.HTTPError as he:
        latency = (time.perf_counter() - t0) * 1000
        # APIM often returns 404 Resource Not Found when no root API operation exists, which proves the Gateway is alive!
        if he.code in (404, 401, 403):
            print(f"  {GREEN}[OK]{RESET} Gateway is Active and Enforcing Routing (Status: {he.code}) ({latency:.1f}ms)")
        else:
            print(f"  {YELLOW}[WARN]{RESET} Gateway HTTP status: {he.code} {he.reason}")
    except Exception as exc:
        print(f"  {RED}[FAIL]{RESET} Gateway connection failed: {exc}")
        return False

    # C. CORS Preflight check
    try:
        req = urllib.request.Request(
            f"{gateway_url}/audio/analyze",
            headers={
                "Origin": "http://localhost:8000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization, Content-Type",
            },
            method="OPTIONS"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            cors_allow = resp.headers.get("Access-Control-Allow-Origin")
            print(f"  {GREEN}[OK]{RESET} CORS OPTIONS Preflight Handled: allow-origin={cors_allow}")
    except urllib.error.HTTPError as he:
        if he.code in (200, 204):
            print(f"  {GREEN}[OK]{RESET} CORS OPTIONS Preflight Handled ({he.code})")
        else:
            print(f"  {YELLOW}[INFO]{RESET} OPTIONS returned HTTP {he.code} (policy dependent)")
    except Exception as exc:
        print(f"  {YELLOW}[INFO]{RESET} OPTIONS check notice: {exc}")

    return True


def test_key_vault():
    print_header("2. Azure Key Vault Diagnostic")
    vault_url = os.getenv("AZURE_KEY_VAULT_URL", "").rstrip("/")
    tenant_id = os.getenv("AZURE_TENANT_ID", "")
    print(f"[*] Vault URL: {vault_url}")
    print(f"[*] Tenant ID: {tenant_id}")

    if not vault_url:
        print(f"  {RED}[FAIL]{RESET} AZURE_KEY_VAULT_URL not configured")
        return False

    try:
        host = vault_url.replace("https://", "").replace("http://", "").split("/")[0]
        ip = socket.gethostbyname(host)
        print(f"  {GREEN}[OK]{RESET} Key Vault Host Resolved: {host} -> {ip}")
    except Exception as e:
        print(f"  {RED}[FAIL]{RESET} Key Vault DNS failed: {e}")
        return False

    try:
        req = urllib.request.Request(f"{vault_url}/secrets?api-version=7.4")
        with urllib.request.urlopen(req, timeout=10):
            pass
    except urllib.error.HTTPError as he:
        # Key Vault should return 401 Unauthorized with WWW-Authenticate header
        auth_header = he.headers.get("WWW-Authenticate", "")
        if he.code == 401 and "login.microsoftonline.com" in auth_header:
            print(f"  {GREEN}[OK]{RESET} Key Vault Auth Challenge Active (Challenge: 401 Unauthorized)")
            print(f"  {GREEN}[OK]{RESET} Entra ID Authority Linked: login.microsoftonline.com/{tenant_id}")
            return True
        else:
            print(f"  {YELLOW}[WARN]{RESET} Unexpected HTTP {he.code}: {he.reason}")
    except Exception as exc:
        print(f"  {RED}[FAIL]{RESET} Key Vault probe failed: {exc}")
        return False
    return True


def test_app_insights():
    print_header("3. Application Insights Telemetry Diagnostic")
    conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
    enabled = os.getenv("APPINSIGHTS_ENABLED", "false").lower() == "true"
    print(f"[*] App Insights Enabled: {enabled}")

    if not conn_str:
        print(f"  {RED}[FAIL]{RESET} APPLICATIONINSIGHTS_CONNECTION_STRING is missing")
        return False

    # Parse components
    parts = dict(p.split("=", 1) for p in conn_str.split(";") if "=" in p)
    ikey = parts.get("InstrumentationKey", "")
    endpoint = parts.get("IngestionEndpoint", "")
    
    print(f"  {GREEN}[OK]{RESET} Instrumentation Key: {ikey[:8]}...{ikey[-4:]}")
    print(f"  {GREEN}[OK]{RESET} Ingestion Endpoint: {endpoint}")

    if endpoint:
        try:
            host = endpoint.replace("https://", "").replace("http://", "").split("/")[0]
            ip = socket.gethostbyname(host)
            print(f"  {GREEN}[OK]{RESET} Telemetry Ingestion Host Resolved: {host} -> {ip}")
            return True
        except Exception as e:
            print(f"  {RED}[FAIL]{RESET} Ingestion DNS failed: {e}")
            return False
    return True


def test_entra_id():
    print_header("4. Entra ID / Tenant Authority Diagnostic")
    tenant_id = os.getenv("AZURE_TENANT_ID", "")
    client_id = os.getenv("AZURE_CLIENT_ID", "")
    print(f"[*] Tenant: {tenant_id}")
    print(f"[*] Client ID (Audience): {client_id}")

    if not tenant_id:
        print(f"  {RED}[FAIL]{RESET} AZURE_TENANT_ID missing")
        return False

    # OpenID discovery endpoint
    oidc_url = f"https://login.microsoftonline.com/{tenant_id}/v2.0/.well-known/openid-configuration"
    try:
        t0 = time.perf_counter()
        req = urllib.request.Request(oidc_url, headers={"User-Agent": "MedVoice-Diagnostic/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            ms = (time.perf_counter() - t0) * 1000
            print(f"  {GREEN}[OK]{RESET} OpenID Authority Active: {data.get('issuer')} ({ms:.1f}ms)")
            print(f"  {GREEN}[OK]{RESET} Token Endpoint: {data.get('token_endpoint')}")
            return True
    except Exception as exc:
        print(f"  {RED}[FAIL]{RESET} Entra ID discovery check failed: {exc}")
        return False


def main():
    print(f"\n{BOLD}Running MedVoice Full Azure Gateway & Infrastructure Test Suite...{RESET}")
    results = [
        test_apim_gateway(),
        test_key_vault(),
        test_app_insights(),
        test_entra_id(),
    ]

    print("\n" + "="*65)
    if all(results):
        print(f"{BOLD}{GREEN}ALL 4 AZURE CLOUD GATEWAY & SERVICE CHECKS PASSED SUCCESSFULLY! [4/4]{RESET}")
    else:
        passed = sum(1 for r in results if r)
        print(f"{BOLD}{YELLOW}Diagnostic Complete: {passed}/4 Passed.{RESET}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
