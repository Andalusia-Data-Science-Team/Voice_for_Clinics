"""
Phase 1: Foundation & Event Infrastructure Tests
Tests:
1.1 SSE Client Hardening:
    - Auto-reconnect with exponential backoff (1s, 2s, 4s; max 3 retries)
    - Connection timeout (30s)
    - Heartbeat detection
    - Last-Event-ID tracking across simulated network drops
1.2 Event State Store:
    - Centralized state object: { structured, fields, reasoning, cautions, speakers, meta }
    - Immutable update pattern (Object.freeze)
    - High performance update benchmark (< 2ms per event)
    - All 5+ event types correctly processed and transitioned
1.3 Error Boundary Wrapper:
    - Section-level error boundary catches exceptions
    - Localized fallback rendering ('⚠ Section loading…')
    - Other sections remain intact and unaffected (Zero full-page crashes)
"""

import subprocess
import tempfile
from pathlib import Path
import pytest


def test_phase1_components_in_html():
    """Verify that all Phase 1 core classes and styles are integrated into index.html."""
    html_path = Path(__file__).resolve().parent.parent / "src" / "static" / "index.html"
    assert html_path.is_file(), f"index.html not found at {html_path}"
    content = html_path.read_text(encoding="utf-8")

    # 1.1 SSE Client Hardening
    assert "class ResilientSSEClient" in content, "Missing ResilientSSEClient"
    assert "retryDelays" in content
    assert "connectionTimeoutMs" in content
    assert "heartbeatIntervalMs" in content
    assert "Last-Event-ID" in content

    # 1.2 Event State Store
    assert "class EventStateStore" in content, "Missing EventStateStore"
    assert "Object.freeze" in content
    assert "transitionLog" in content
    assert "window.appStateStore" in content

    # 1.3 Error Boundary Wrapper
    assert "class SectionErrorBoundary" in content, "Missing SectionErrorBoundary"
    assert "section-error-boundary" in content
    assert "sectionErrorBoundaryRetry" in content


def test_phase1_node_unit_tests():
    """Execute rigorous Node.js unit tests against the extracted Phase 1 implementation."""
    html_path = Path(__file__).resolve().parent.parent / "src" / "static" / "index.html"
    assert html_path.is_file(), f"index.html not found at {html_path}"

    test_js_path = Path(__file__).resolve().parent / "run_phase1_node_tests.js"
    assert test_js_path.is_file(), f"run_phase1_node_tests.js not found at {test_js_path}"

    res = subprocess.run(["node", str(test_js_path), str(html_path)], capture_output=True, text=True)

    if res.returncode != 0:
        pytest.fail(f"Phase 1 Node unit tests failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    else:
        print(f"\n{res.stdout}")


if __name__ == "__main__":
    test_phase1_components_in_html()
    test_phase1_node_unit_tests()
    print("All Phase 1 checks passed successfully!")
