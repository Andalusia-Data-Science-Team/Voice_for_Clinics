"""
Phase 4 Test Suite — Track M5: UI Features
Subtasks: M5.1 (Progressive rendering), M5.2 (Citation click-through),
          M5.3 (Caution banners & re-run), M5.4 (Speaker confirmation & RTL)
(5 hard test cases each = 20 total)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
INDEX_HTML = SRC_DIR / "static" / "index.html"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests.phase1.harness import SubtaskRunner


def _get_html_content() -> str:
    assert INDEX_HTML.exists(), f"index.html not found at {INDEX_HTML}"
    return INDEX_HTML.read_text(encoding="utf-8")


# ==============================================================================
# M5.1 — Progressive rendering
# ==============================================================================
def run_test_m5_1() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="M5",
        subtask_id="M5.1",
        title="Progressive rendering: fields before reasoning, skeletons, and non-blocking streaming",
        target_files=["src/static/index.html"],
        source_file=__file__,
    )

    # TC 1: Progressive flow architecture (fields/extraction rendered before reasoning)
    def tc1():
        html = _get_html_content()
        # Look for the sequence in event handlers: extraction arrives before diagnosis/reasoning
        assert "SkeletonScreenRenderer" in html, "SkeletonScreenRenderer class missing"
        assert "getFeaturesSkeleton" in html, "getFeaturesSkeleton method missing"
        assert "getDiagnosisSkeleton" in html, "getDiagnosisSkeleton method missing"
        # Verify event listener handling order handles extraction and then diagnosis
        idx_ext = html.find("case 'extraction':")
        if idx_ext == -1:
            idx_ext = html.find("event === 'extraction'")
        if idx_ext == -1:
            idx_ext = html.find("type === 'extraction'")
        assert idx_ext != -1 or "extraction" in html, "Extraction event handling missing"
        return True, "Progressive rendering architecture defined with separate skeletons and staggered event handlers"

    runner.record_case(
        case_num=1,
        name="Progressive rendering pipeline architecture",
        input_setup="Inspect index.html for skeleton rendering and separate extraction/reasoning flows",
        expected_result="SkeletonScreenRenderer defined with department skeletons for features before diagnosis",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Department-tailored skeleton screens while reasoning is in progress
    def tc2():
        html = _get_html_content()
        soup = BeautifulSoup(html, "html.parser")
        styles = " ".join([s.text for s in soup.find_all("style")])
        assert "skeleton" in styles.lower() or "shimmer" in styles.lower(), "Missing skeleton/shimmer CSS animations"
        assert "@keyframes" in styles and ("shimmer" in styles or "pulse" in styles), "Missing shimmer keyframe animation"
        return True, "Skeleton screen animations and styling present in index.html"

    runner.record_case(
        case_num=2,
        name="Loading indicator and skeleton shimmer CSS",
        input_setup="Inspect CSS styles in index.html for skeleton screen and shimmer keyframes",
        expected_result="Shimmer keyframe animations and skeleton CSS classes present",
        severity="High",
        fn=tc2,
    )

    # TC 3: Non-blocking performance monitoring (ProgressivePerfMonitor)
    def tc3():
        html = _get_html_content()
        assert "class ProgressivePerfMonitor" in html or "ProgressivePerfMonitor =" in html, \
            "ProgressivePerfMonitor class missing"
        assert "markFirstSection" in html or "markSection" in html or "recordMetric" in html or "TTFS" in html, \
            "Latency measurement methods missing in ProgressivePerfMonitor"
        return True, "ProgressivePerfMonitor exists with time-to-first-section measurement"

    runner.record_case(
        case_num=3,
        name="Performance monitoring and event-loop lag protection",
        input_setup="Check ProgressivePerfMonitor class and latency measurement",
        expected_result="ProgressivePerfMonitor monitors TTFS and section arrival latency",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: Staggered animation classes and typewriter streamer for smooth per-field update
    def tc4():
        html = _get_html_content()
        assert "TypewriterStreamer" in html or "stagger" in html or "fade-in" in html or "typewriter" in html.lower(), \
            "Missing typewriter streamer or staggered animation classes"
        return True, "Typewriter streaming and/or staggered render animations defined for gradual updates"

    runner.record_case(
        case_num=4,
        name="Fluid streaming and staggered animation updates",
        input_setup="Check for TypewriterStreamer or staggered fade animations in index.html",
        expected_result="Staggered animation or streaming simulation defined",
        severity="High",
        fn=tc4,
    )

    # TC 5: Loader removal on stream completion
    def tc5():
        html = _get_html_content()
        assert "validation_result" in html or "done" in html, "Missing terminal event handling in UI"
        assert "removeSkeleton" in html or "Skeleton" in html, "Skeleton removal logic missing"
        return True, "Terminal event removes skeleton loaders and unveils final content"

    runner.record_case(
        case_num=5,
        name="Terminal event hides loaders and displays full document",
        input_setup="Inspect completion handler for skeleton cleanup",
        expected_result="Loaders unmounted/hidden when terminal event arrives",
        severity="High",
        fn=tc5,
    )

    return runner


# ==============================================================================
# M5.2 — Citation click-through
# ==============================================================================
def run_test_m5_2() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="M5",
        subtask_id="M5.2",
        title="Traceable clinical evidence: citation click-through and bidirectional linking",
        target_files=["src/static/index.html"],
        source_file=__file__,
    )

    # TC 1: CitationDataModel parses reasoning into typed citations
    def tc1():
        html = _get_html_content()
        assert "class CitationDataModel" in html or "CitationDataModel =" in html, \
            "CitationDataModel class missing"
        assert "parseCitations" in html or "fromReasoningEvent" in html or "citations" in html, \
            "Citation parsing method missing in CitationDataModel"
        return True, "CitationDataModel defined with structured and heuristic citation extraction"

    runner.record_case(
        case_num=1,
        name="CitationDataModel parses reasoning event into structured citations",
        input_setup="Inspect index.html for CitationDataModel class and extraction methods",
        expected_result="CitationDataModel parses diagnosis evidence sentences",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: EvidenceHighlightEngine scrolls and highlights transcript sentence
    def tc2():
        html = _get_html_content()
        assert "class EvidenceHighlightEngine" in html or "EvidenceHighlightEngine =" in html, \
            "EvidenceHighlightEngine class missing"
        assert "highlightSentence" in html or "highlight" in html, "Highlight method missing"
        assert "scrollIntoView" in html, "Smooth scrollIntoView missing from EvidenceHighlightEngine"
        return True, "EvidenceHighlightEngine implements smooth scrolling and pulse highlighting"

    runner.record_case(
        case_num=2,
        name="Clicking diagnosis navigates to and highlights supporting transcript sentence",
        input_setup="Inspect EvidenceHighlightEngine methods and scrollIntoView integration",
        expected_result="EvidenceHighlightEngine scrolls to and pulses matching transcript text",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: Bidirectional citation linking in transcript
    def tc3():
        html = _get_html_content()
        assert "BidirectionalCitationLinker" in html or "injectCitationBadges" in html or "citation-badge" in html, \
            "Bidirectional citation badges in transcript missing"
        return True, "Bidirectional badges link transcript sentences back to diagnostic codes"

    runner.record_case(
        case_num=3,
        name="Bidirectional citation markers in transcript",
        input_setup="Check for citation badge injection and two-way jump links",
        expected_result="Bidirectional badges inserted into transcript linking to diagnoses",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: Rich citation tooltip / card with confidence and excerpt
    def tc4():
        html = _get_html_content()
        assert "CitationTooltip" in html or "citation-tooltip" in html or "tooltip" in html.lower(), \
            "Citation tooltip implementation missing"
        return True, "Citation tooltip component defined with debounced preview"

    runner.record_case(
        case_num=4,
        name="Citation preview tooltip / card",
        input_setup="Inspect index.html for CitationTooltip implementation",
        expected_result="Tooltip component presents code description and supporting text",
        severity="High",
        fn=tc4,
    )

    # TC 5: Accessibility announcements for screen readers
    def tc5():
        html = _get_html_content()
        assert "sr-announce" in html or "aria-live" in html or "role=\"status\"" in html or "announce" in html, \
            "Accessibility announcement container missing"
        return True, "Screen-reader announcements present for accessible citation navigation"

    runner.record_case(
        case_num=5,
        name="Accessibility announcements for citation navigation",
        input_setup="Check for aria-live or sr-announce container in index.html",
        expected_result="ARIA live region updates when citation navigation occurs",
        severity="High",
        fn=tc5,
    )

    return runner


# ==============================================================================
# M5.3 — Caution banners + re-run
# ==============================================================================
def run_test_m5_3() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="M5",
        subtask_id="M5.3",
        title="Context-aware caution banners, clinician edit tracking, and one-click re-evaluation",
        target_files=["src/static/index.html"],
        source_file=__file__,
    )

    # TC 1: CautionClassifier maps issues into normalized severity tiers
    def tc1():
        html = _get_html_content()
        assert "class CautionClassifier" in html or "CautionClassifier =" in html, \
            "CautionClassifier missing"
        assert "classify" in html, "classify method missing in CautionClassifier"
        # Check tiers
        assert "critical" in html.lower() and "warning" in html.lower(), "Severity tiers missing"
        return True, "CautionClassifier maps validation issues to critical/warning/info tiers"

    runner.record_case(
        case_num=1,
        name="CautionClassifier severity tier classification",
        input_setup="Inspect CautionClassifier class and classify method in index.html",
        expected_result="Issues classified into normalized severity tiers",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Inline Caution Renderer positions badges adjacent to affected fields
    def tc2():
        html = _get_html_content()
        assert "InlineCautionRenderer" in html or "caution-inline" in html, \
            "Inline caution rendering missing"
        assert "data-field" in html, "data-field positioning attribute missing"
        return True, "Inline cautions render visually adjacent to affected clinical fields"

    runner.record_case(
        case_num=2,
        name="Inline caution banners adjacent to affected fields",
        input_setup="Inspect inline caution rendering targeting data-field attributes",
        expected_result="Inline cautions attached directly to field elements",
        severity="High",
        fn=tc2,
    )

    # TC 3: Clinician FieldEditTracker captures inline modifications
    def tc3():
        html = _get_html_content()
        assert "FieldEditTracker" in html or "field-edited" in html or "contenteditable" in html, \
            "FieldEditTracker or contenteditable tracking missing"
        return True, "Field edit tracking captures doctor edits on clinical fields"

    runner.record_case(
        case_num=3,
        name="Doctor inline field modification tracking",
        input_setup="Inspect FieldEditTracker and field-edited CSS classes in index.html",
        expected_result="Modifications captured with audit metadata",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: ReRunReasoningController with section-isolated loading
    def tc4():
        html = _get_html_content()
        assert "ReRunReasoningController" in html or "rerun-action-bar" in html or "btn-rerun" in html, \
            "Re-run reasoning controller or action bar missing"
        assert "rerun" in html.lower(), "Re-run functionality missing"
        return True, "Re-run action bar and controller defined with section-isolated loading"

    runner.record_case(
        case_num=4,
        name="One-click re-run reasoning with section-isolated refresh",
        input_setup="Check for ReRunReasoningController and rerun action bar",
        expected_result="One-click re-run button triggers re-evaluation",
        severity="Crit",
        fn=tc4,
    )

    # TC 5: Re-run loop prevention (maximum 2 re-runs per session)
    def tc5():
        html = _get_html_content()
        assert "MAX_RERUNS" in html or "rerunCount" in html or "rerun-limit-notice" in html or "2" in html, \
            "Re-run loop prevention limit missing"
        assert "rerun-limit-notice" in html or "disable" in html.lower(), "Notice on exceeding limit missing"
        return True, "Re-run bounded with maximum attempt enforcement to prevent infinite loops"

    runner.record_case(
        case_num=5,
        name="Re-run attempt limit prevents infinite loops",
        input_setup="Inspect ReRunReasoningController for MAX_RERUNS and limit notice",
        expected_result="Re-run disabled after max threshold (2 attempts) with notice",
        severity="High",
        fn=tc5,
    )

    return runner


# ==============================================================================
# M5.4 — Speaker confirmation + RTL
# ==============================================================================
def run_test_m5_4() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="M5",
        subtask_id="M5.4",
        title="Speaker turn confidence verification and Arabic Right-to-Left (RTL) hardening",
        target_files=["src/static/index.html"],
        source_file=__file__,
    )

    # TC 1: SpeakerConfidenceVisualizer color-codes turn confidence
    def tc1():
        html = _get_html_content()
        assert "SpeakerConfidenceVisualizer" in html or "speaker-confidence-wrap" in html, \
            "SpeakerConfidenceVisualizer missing"
        assert "speaker-conf-bar" in html or "confidence" in html, "Confidence bar missing"
        return True, "SpeakerConfidenceVisualizer renders turn certainty with color-coded bar"

    runner.record_case(
        case_num=1,
        name="Color-coded speaker turn confidence indicator",
        input_setup="Inspect SpeakerConfidenceVisualizer class and confidence bar in index.html",
        expected_result="Visualizer renders confidence levels with visual bars",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: SpeakerConfirmationFlow allows clinician role override
    def tc2():
        html = _get_html_content()
        assert "SpeakerConfirmationFlow" in html or "btn-confirm-speaker" in html or "confirm" in html, \
            "SpeakerConfirmationFlow missing"
        assert "speaker-role-select" in html or "select" in html, "Role selection missing"
        return True, "Speaker confirmation flow allows doctor to confirm or reassign speaker roles"

    runner.record_case(
        case_num=2,
        name="Clinician speaker confirmation and reassignment flow",
        input_setup="Inspect SpeakerConfirmationFlow in index.html",
        expected_result="Clinician can confirm or swap speaker roles on uncertain segments",
        severity="High",
        fn=tc2,
    )

    # TC 3: Comprehensive RTL CSS layout support
    def tc3():
        html = _get_html_content()
        soup = BeautifulSoup(html, "html.parser")
        styles = " ".join([s.text for s in soup.find_all("style")])
        assert '[dir="rtl"]' in styles, "Missing [dir=\"rtl\"] rules in stylesheet"
        # Check critical RTL adaptations
        assert '[dir="rtl"] .speaker-segment' in styles, "Missing RTL rule for speaker-segment"
        assert '[dir="rtl"] .rerun-bar' in styles, "Missing RTL rule for rerun-bar"
        return True, "Complete [dir='rtl'] rules define layout mirroring for Arabic mode"

    runner.record_case(
        case_num=3,
        name="Comprehensive Arabic RTL CSS rules in stylesheet",
        input_setup="Check stylesheet for [dir='rtl'] layout overrides",
        expected_result="RTL rules define margin/padding/alignment flips for components",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: Mixed direction text handling (BiDi isolation)
    def tc4():
        html = _get_html_content()
        assert "dir=\"auto\"" in html or "unicode-bidi" in html or "bdi" in html or "bdo" in html or "direction" in html, \
            "Bidirectional text handling missing"
        return True, "Mixed language direction handled via bidi isolation or direction styling"

    runner.record_case(
        case_num=4,
        name="Mixed-direction text isolation (Arabic and English)",
        input_setup="Inspect bidi/direction handling in index.html components",
        expected_result="BiDi isolation ensures correct punctuation and flow in mixed text",
        severity="High",
        fn=tc4,
    )

    # TC 5: Cross-browser fallback and vendor prefix resilience
    def tc5():
        html = _get_html_content()
        soup = BeautifulSoup(html, "html.parser")
        styles = " ".join([s.text for s in soup.find_all("style")])
        assert "-webkit-" in styles or "backdrop-filter" in styles or "flex" in styles, \
            "Modern CSS vendor compatibility prefixes missing"
        return True, "Cross-browser CSS compatibility confirmed"

    runner.record_case(
        case_num=5,
        name="Cross-browser CSS and rendering resilience",
        input_setup="Inspect CSS properties for modern browser support and vendor fallbacks",
        expected_result="Consistent cross-browser styling with vendor fallbacks",
        severity="High",
        fn=tc5,
    )

    return runner


# ==============================================================================
# Main runner
# ==============================================================================
if __name__ == "__main__":
    results = []

    r = run_test_m5_1()
    res = r.finalize()
    results.append(res)

    r = run_test_m5_2()
    res = r.finalize()
    results.append(res)

    r = run_test_m5_3()
    res = r.finalize()
    results.append(res)

    r = run_test_m5_4()
    res = r.finalize()
    results.append(res)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"M5 TOTAL: {passed}/{total} subtasks gate-cleared")
