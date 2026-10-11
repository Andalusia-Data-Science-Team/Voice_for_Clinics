"""Phase 5.3 Frontend Static and Contract Validation."""

from pathlib import Path


def test_frontend_icd_components():
    html_path = Path(__file__).resolve().parent.parent / "src" / "static" / "index.html"
    assert html_path.is_file(), f"index.html not found at {html_path}"

    content = html_path.read_text(encoding="utf-8")

    # 1. Verify autocomplete dropdown & spinner exist
    assert "icd-dropdown" in content, "Missing .icd-dropdown in index.html"
    assert "icd-search-spinner" in content, "Missing .icd-search-spinner in index.html"
    assert "icd-search-input" in content, "Missing .icd-search-input in index.html"

    # 2. Verify unmatched warning card exists
    assert "icd-unmatched-card" in content, "Missing .icd-unmatched-card in index.html"
    assert "btn-replace-icd" in content, "Missing .btn-replace-icd in index.html"
    assert "focusIcdSearch" in content, "Missing focusIcdSearch function in index.html"

    # 3. Verify free-text edit is disabled on pills
    assert 'contenteditable="false"' in content, "Pills must be non-editable (contenteditable='false')"

    # 4. Verify search endpoint is wired
    assert "/audio/icd10/search" in content, "Autocomplete must call /audio/icd10/search"

    # 5. Verify keyboard navigation handling
    assert "ArrowDown" in content, "Keyboard navigation missing ArrowDown"
    assert "ArrowUp" in content, "Keyboard navigation missing ArrowUp"
    assert "Escape" in content, "Keyboard navigation missing Escape"


if __name__ == "__main__":
    test_frontend_icd_components()
    print("All Phase 5.3 frontend static and contract assertions passed successfully!")
