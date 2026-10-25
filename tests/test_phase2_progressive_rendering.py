"""
Phase 2: Progressive Rendering Enhancement Tests
Tests:
2.1 Skeleton Screens:
    - Shimmer placeholders for all 3 departments: ER, OPD, Dental
    - Immediate render and transition < 100ms
2.2 Staggered Reveal Animation:
    - CSS @keyframes staggeredFadeIn (60fps, 50ms stagger per child)
    - Zero Cumulative Layout Shift (CLS)
2.3 Streaming Text for Reasoning:
    - If reasoning text is long (>500 chars), renders smoothly with typewriter effect
    - Short reasoning renders immediately
2.4 Progressive Rendering Benchmarks:
    - Measures TTFS (Time-to-First-Section) < 2s and TTC (Time-to-Complete) < 8s on 10 sample encounters
"""

import subprocess
from pathlib import Path
import pytest


def test_phase2_css_and_components_in_html():
    """Verify that all Phase 2 CSS classes and utility components exist in index.html."""
    html_path = Path(__file__).resolve().parent.parent / "src" / "static" / "index.html"
    assert html_path.is_file(), f"index.html not found at {html_path}"
    content = html_path.read_text(encoding="utf-8")

    # 2.1 Skeleton CSS
    assert ".skeleton-container" in content, "Missing .skeleton-container CSS"
    assert ".skeleton-shimmer" in content, "Missing .skeleton-shimmer CSS"
    assert "skeletonShimmer" in content, "Missing skeletonShimmer @keyframes"
    assert "class SkeletonScreenRenderer" in content, "Missing SkeletonScreenRenderer class"

    # 2.2 Staggered Reveal Animation CSS
    assert "@keyframes staggeredFadeIn" in content, "Missing @keyframes staggeredFadeIn"
    assert ".stagger-reveal" in content, "Missing .stagger-reveal CSS"
    assert ".stagger-delay-1" in content, "Missing .stagger-delay-1 CSS"
    assert ".stagger-delay-2" in content, "Missing .stagger-delay-2 CSS"
    assert ".stagger-delay-3" in content, "Missing .stagger-delay-3 CSS"

    # 2.3 Streaming Text for Reasoning
    assert "class TypewriterStreamer" in content, "Missing TypewriterStreamer class"
    assert ".typewriter-cursor" in content, "Missing .typewriter-cursor CSS"
    assert "blinkCursor" in content, "Missing blinkCursor @keyframes"

    # 2.4 Progressive Rendering Benchmarks
    assert "class ProgressivePerfMonitor" in content, "Missing ProgressivePerfMonitor class"
    assert "window.progressivePerfMonitor" in content, "Missing global progressivePerfMonitor"


def test_phase2_node_execution():
    """Execute Phase 2 Node unit tests covering skeletons, typewriter streaming, and 10 encounter benchmarks."""
    html_path = Path(__file__).resolve().parent.parent / "src" / "static" / "index.html"
    assert html_path.is_file(), f"index.html not found at {html_path}"

    test_js_path = Path(__file__).resolve().parent / "run_phase2_node_tests.js"
    assert test_js_path.is_file(), f"run_phase2_node_tests.js not found at {test_js_path}"

    res = subprocess.run(["node", str(test_js_path), str(html_path)], capture_output=True, text=True)

    if res.returncode != 0:
        pytest.fail(f"Phase 2 Node unit tests failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    else:
        print(f"\n{res.stdout}")


if __name__ == "__main__":
    test_phase2_css_and_components_in_html()
    test_phase2_node_execution()
    print("All Phase 2 progressive rendering tests passed successfully!")
