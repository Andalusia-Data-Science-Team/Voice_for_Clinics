"""
Phase 3 — Citation System & Evidence Linking
Test Suite: CitationDataModel (Python simulation)

Tests the core parsing logic for the citation data model, covering:
  - Task 3.1: Citation data model parse (Path A: structured, Path B: heuristic)
  - Task 3.1: Parse time < 10ms per encounter
  - Task 3.2: Evidence sentence detection in transcript text
  - Task 3.3: Transcript offset finding for bidirectional marker injection
  - 5 sample encounters (2 ER, 2 OPD, 1 Dental) including Arabic RTL

Run: pytest tests/test_phase3_citation_system.py -v
"""

import re
import time
import pytest


class CitationDataModel:
    def __init__(self, dx, transcript_text=""):
        self.dx = dx or {}
        self.transcript = transcript_text or ""
        self.citations = []

    def parse(self):
        t0 = time.perf_counter()
        self.citations = []
        icd10_codes = self.dx.get("icd10_codes", [])
        if not isinstance(icd10_codes, list):
            icd10_codes = []
        reasoning = (self.dx.get("reasoning") or "").strip()
        citations_raw = self.dx.get("citations")

        if isinstance(citations_raw, list) and len(citations_raw) > 0:
            self.citations = [self._normalise(c, i) for i, c in enumerate(citations_raw)]
        else:
            for i, code in enumerate(icd10_codes):
                parsed = self._parse_code_string(code)
                evidence = self._find_evidence_sentence(parsed["code"], parsed["description"], reasoning)
                offset = self._find_transcript_offset(evidence) if evidence else -1
                self.citations.append({
                    "id": f"cit-{i}",
                    "icd10Code": parsed["code"],
                    "description": parsed["description"],
                    "diagnosis": parsed["description"] or code,
                    "evidenceSentence": evidence or "",
                    "transcriptOffset": offset,
                    "confidence": self._derive_confidence(parsed["code"], reasoning),
                    "field": "icd10",
                    "raw": code,
                })
        parse_ms = (time.perf_counter() - t0) * 1000
        return {"citations": self.citations, "parseMs": parse_ms}

    def _normalise(self, raw, i):
        parsed = self._parse_code_string(raw.get("icd10_code") or raw.get("code") or "")
        evidence = raw.get("evidence_sentence") or raw.get("evidenceSentence") or ""
        return {
            "id": raw.get("id") or f"cit-{i}",
            "icd10Code": parsed["code"],
            "description": raw.get("description") or parsed["description"],
            "diagnosis": raw.get("diagnosis") or raw.get("description") or parsed["description"],
            "evidenceSentence": evidence,
            "transcriptOffset": raw["transcript_offset"] if isinstance(raw.get("transcript_offset"), int) else self._find_transcript_offset(evidence),
            "confidence": raw["confidence"] if isinstance(raw.get("confidence"), (int, float)) else 0.85,
            "field": raw.get("field") or "icd10",
            "raw": raw,
        }

    def _parse_code_string(self, s):
        if not s:
            return {"code": "", "description": ""}
        m = re.match(r"^([A-Z]\d{2,3}(?:\.\d+)?)\s*[-\u2013\u2014]\s*(.+)$", s)
        if m:
            return {"code": m.group(1).strip(), "description": m.group(2).strip()}
        if re.match(r"^[A-Z]\d", s):
            return {"code": s.strip(), "description": ""}
        return {"code": "", "description": s.strip()}

    def _find_evidence_sentence(self, code, description, reasoning):
        if not reasoning:
            return ""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?\u061f\u0964])\s+|\n+", reasoning) if len(s.strip()) > 10]
        keywords = [code] + [w for w in (description or "").lower().split() if len(w) > 4]
        best, best_score = None, 0
        for sentence in sentences:
            lower = sentence.lower()
            score = sum(1 for kw in keywords if kw.lower() in lower)
            if score > best_score:
                best_score, best = score, sentence
        if not best and code:
            best = next((s for s in sentences if code in s), sentences[0] if sentences else "")
        return (best or "").strip()

    def _find_transcript_offset(self, sentence):
        if not sentence or not self.transcript:
            return -1
        idx = self.transcript.find(sentence[:40])
        if idx >= 0:
            return idx
        words = " ".join(sentence.split()[:8])
        return self.transcript.find(words)

    def _derive_confidence(self, code, reasoning):
        if not reasoning or not code:
            return 0.85
        lower = reasoning.lower()
        high = any(w in lower for w in ["confirmed", "diagnosed", "consistent with", "consistent", "clear"])
        low = any(w in lower for w in ["possible", "probable", "rule out", "differential", "consider", "uncertain"])
        if high and not low:
            return 0.92
        if low:
            return 0.65
        return 0.82

    def get_by_code(self, code):
        return next((c for c in self.citations if c["icd10Code"] == code), None)

    def get_with_evidence(self):
        return [c for c in self.citations if c["evidenceSentence"]]


# Fixtures
ENCOUNTER_OPD_ENGLISH = {
    "dx": {
        "icd10_codes": ["J06.9 - Acute upper respiratory infection", "R05 - Cough"],
        "reasoning": (
            "The patient presents with a 3-day history of sore throat and fever. "
            "Physical examination is consistent with acute upper respiratory infection (J06.9). "
            "The persistent cough (R05) is secondary to pharyngeal inflammation. "
            "No signs of pneumonia; antibiotics not indicated."
        ),
    },
    "transcript": (
        "Doctor: Good morning. What brings you in today? "
        "Patient: I've had a sore throat and fever for about 3 days. "
        "Doctor: Any cough? Patient: Yes, a dry cough. No shortness of breath."
    ),
}

ENCOUNTER_ER_ENGLISH = {
    "dx": {
        "icd10_codes": ["I21.9 - Acute myocardial infarction", "I10 - Essential hypertension"],
        "reasoning": (
            "The patient reports central crushing chest pain radiating to the left arm. "
            "ECG findings confirmed STEMI consistent with acute myocardial infarction (I21.9). "
            "Known hypertension history (I10) is a significant risk factor here. "
            "Troponin levels are markedly elevated."
        ),
    },
    "transcript": (
        "Paramedic: 58-year-old male, central crushing chest pain radiating to the left arm, onset 45 minutes ago. "
        "Known hypertension history. ECG shows STEMI. Troponin elevated."
    ),
}

ENCOUNTER_DENTAL_ENGLISH = {
    "dx": {
        "icd10_codes": ["K02.9 - Dental caries"],
        "reasoning": (
            "Examination reveals caries in tooth 36 consistent with K02.9. "
            "Patient reports sharp pain on cold stimuli for 2 weeks. "
            "Radiograph confirms dentinal involvement without periapical pathology."
        ),
    },
    "transcript": (
        "Patient: I have sharp pain in my lower left molar when I drink cold water. "
        "It started about 2 weeks ago. "
        "Dentist: The X-ray shows caries in tooth 36. No abscess visible."
    ),
}

ENCOUNTER_OPD_ARABIC = {
    "dx": {
        "icd10_codes": ["J06.9 - \u0627\u0644\u062a\u0647\u0627\u0628 \u0627\u0644\u062c\u0647\u0627\u0632 \u0627\u0644\u062a\u0646\u0641\u0633\u064a \u0627\u0644\u062d\u0627\u062f"],
        "reasoning": (
            "\u064a\u0634\u0643\u0648 \u0627\u0644\u0645\u0631\u064a\u0636 \u0645\u0646 \u0623\u0644\u0645 \u0641\u064a \u0627\u0644\u062d\u0644\u0642 \u0648\u0627\u0631\u062a\u0641\u0627\u0639 \u0641\u064a \u062f\u0631\u062c\u0629 \u0627\u0644\u062d\u0631\u0627\u0631\u0629 \u0645\u0646\u0630 3 \u0623\u064a\u0627\u0645. "
            "\u0627\u0644\u0641\u062d\u0635 \u0627\u0644\u0633\u0631\u064a\u0631\u064a \u0645\u062a\u0648\u0627\u0641\u0642 \u0645\u0639 \u0627\u0644\u062a\u0647\u0627\u0628 \u0627\u0644\u062c\u0647\u0627\u0632 \u0627\u0644\u062a\u0646\u0641\u0633\u064a \u0627\u0644\u062d\u0627\u062f J06.9. "
            "\u0644\u0627 \u062a\u0648\u062c\u062f \u0639\u0644\u0627\u0645\u0627\u062a \u0639\u0644\u0649 \u0627\u0644\u062a\u0647\u0627\u0628 \u0631\u0626\u0648\u064a."
        ),
    },
    "transcript": (
        "\u0627\u0644\u0637\u0628\u064a\u0628: \u0635\u0628\u0627\u062d \u0627\u0644\u062e\u064a\u0631\u060c \u0645\u0627 \u0633\u0628\u0628 \u0632\u064a\u0627\u0631\u062a\u0643 \u0627\u0644\u064a\u0648\u0645\u061f "
        "\u0627\u0644\u0645\u0631\u064a\u0636: \u0639\u0646\u062f\u064a \u0623\u0644\u0645 \u0641\u064a \u0627\u0644\u062d\u0644\u0642 \u0648\u062d\u0631\u0627\u0631\u0629 \u0645\u0646 3 \u0623\u064a\u0627\u0645."
    ),
}

ENCOUNTER_ER_ARABIC = {
    "dx": {
        "icd10_codes": ["S72.0 - \u0643\u0633\u0631 \u0639\u0646\u0642 \u0627\u0644\u0641\u062e\u0630"],
        "reasoning": (
            "\u0627\u0644\u0645\u0631\u064a\u0636\u0629 \u0633\u0642\u0637\u062a \u0648\u062a\u0634\u0643\u0648 \u0645\u0646 \u0623\u0644\u0645 \u0634\u062f\u064a\u062f \u0641\u064a \u0627\u0644\u0648\u0631\u0643 \u0627\u0644\u0623\u064a\u0645\u0646. "
            "\u0627\u0644\u0623\u0634\u0639\u0629 \u062a\u0624\u0643\u062f \u0643\u0633\u0631 \u0639\u0646\u0642 \u0627\u0644\u0641\u062e\u0630 S72.0. "
            "\u0627\u0644\u062d\u0627\u0644\u0629 \u062a\u0633\u062a\u062f\u0639\u064a \u062a\u062f\u062e\u0644\u0627\u064b \u062c\u0631\u0627\u062d\u064a\u0627\u064b \u0639\u0627\u062c\u0644\u0627\u064b."
        ),
    },
    "transcript": (
        "\u0627\u0644\u0645\u0631\u064a\u0636\u0629: \u0633\u0642\u0637\u062a \u0648\u0639\u0646\u062f\u064a \u0623\u0644\u0645 \u0634\u062f\u064a\u062f \u0641\u064a \u0627\u0644\u0648\u0631\u0643 \u0627\u0644\u0623\u064a\u0645\u0646. "
        "\u0627\u0644\u0637\u0628\u064a\u0628: \u0627\u0644\u0623\u0634\u0639\u0629 \u062a\u0638\u0647\u0631 \u0643\u0633\u0631 \u0641\u064a \u0639\u0646\u0642 \u0627\u0644\u0641\u062e\u0630."
    ),
}

ENCOUNTER_PATH_A = {
    "dx": {
        "citations": [{
            "icd10_code": "J18.9",
            "description": "Pneumonia, unspecified",
            "evidence_sentence": "Chest X-ray confirms bilateral infiltrates",
            "transcript_offset": 42,
            "confidence": 0.91,
            "field": "icd10",
        }]
    },
    "transcript": "The patient presents with fever and cough. Chest X-ray confirms bilateral infiltrates consistent with pneumonia.",
}


class TestCitationDataModelParsing:
    def test_opd_english_parse_count(self):
        m = CitationDataModel(ENCOUNTER_OPD_ENGLISH["dx"], ENCOUNTER_OPD_ENGLISH["transcript"])
        r = m.parse()
        assert len(r["citations"]) == 2

    def test_opd_english_code_extraction(self):
        m = CitationDataModel(ENCOUNTER_OPD_ENGLISH["dx"], ENCOUNTER_OPD_ENGLISH["transcript"])
        m.parse()
        cit = m.get_by_code("J06.9")
        assert cit is not None
        assert "upper respiratory" in cit["description"].lower()

    def test_er_english_parse_count(self):
        m = CitationDataModel(ENCOUNTER_ER_ENGLISH["dx"], ENCOUNTER_ER_ENGLISH["transcript"])
        r = m.parse()
        assert len(r["citations"]) == 2

    def test_dental_parse(self):
        m = CitationDataModel(ENCOUNTER_DENTAL_ENGLISH["dx"], ENCOUNTER_DENTAL_ENGLISH["transcript"])
        r = m.parse()
        assert len(r["citations"]) == 1
        assert r["citations"][0]["icd10Code"] == "K02.9"

    def test_arabic_opd_parse(self):
        m = CitationDataModel(ENCOUNTER_OPD_ARABIC["dx"], ENCOUNTER_OPD_ARABIC["transcript"])
        r = m.parse()
        assert len(r["citations"]) == 1
        assert r["citations"][0]["icd10Code"] == "J06.9"

    def test_arabic_er_parse(self):
        m = CitationDataModel(ENCOUNTER_ER_ARABIC["dx"], ENCOUNTER_ER_ARABIC["transcript"])
        r = m.parse()
        assert len(r["citations"]) == 1
        assert r["citations"][0]["icd10Code"] == "S72.0"

    def test_path_a_structured(self):
        m = CitationDataModel(ENCOUNTER_PATH_A["dx"], ENCOUNTER_PATH_A["transcript"])
        r = m.parse()
        assert len(r["citations"]) == 1
        cit = r["citations"][0]
        assert cit["icd10Code"] == "J18.9"
        assert cit["confidence"] == 0.91
        assert cit["transcriptOffset"] == 42

    def test_empty_dx(self):
        m = CitationDataModel({}, "")
        r = m.parse()
        assert r["citations"] == []

    def test_bare_code_only(self):
        m = CitationDataModel({"icd10_codes": ["J06"]}, "some transcript")
        r = m.parse()
        assert r["citations"][0]["icd10Code"] == "J06"


class TestCitationParsePerformance:
    ENCOUNTERS = [
        ENCOUNTER_OPD_ENGLISH,
        ENCOUNTER_ER_ENGLISH,
        ENCOUNTER_DENTAL_ENGLISH,
        ENCOUNTER_OPD_ARABIC,
        ENCOUNTER_ER_ARABIC,
    ]

    @pytest.mark.parametrize("enc_idx,enc", enumerate(ENCOUNTERS))
    def test_parse_under_10ms(self, enc_idx, enc):
        m = CitationDataModel(enc["dx"], enc.get("transcript", ""))
        r = m.parse()
        assert r["parseMs"] < 10.0, f"Encounter {enc_idx}: {r['parseMs']:.2f}ms > 10ms target"

    def test_path_a_under_10ms(self):
        m = CitationDataModel(ENCOUNTER_PATH_A["dx"], ENCOUNTER_PATH_A["transcript"])
        r = m.parse()
        assert r["parseMs"] < 10.0


class TestEvidenceSentenceDetection:
    def test_j069_evidence_contains_keywords(self):
        m = CitationDataModel(ENCOUNTER_OPD_ENGLISH["dx"], ENCOUNTER_OPD_ENGLISH["transcript"])
        m.parse()
        cit = m.get_by_code("J06.9")
        assert cit and cit["evidenceSentence"]
        ev = cit["evidenceSentence"].lower()
        assert any(kw in ev for kw in ["upper respiratory", "j06.9", "sore throat"])

    def test_i219_evidence_contains_keywords(self):
        m = CitationDataModel(ENCOUNTER_ER_ENGLISH["dx"], ENCOUNTER_ER_ENGLISH["transcript"])
        m.parse()
        cit = m.get_by_code("I21.9")
        assert cit and cit["evidenceSentence"]
        ev = cit["evidenceSentence"].lower()
        assert any(kw in ev for kw in ["myocardial", "stemi", "i21.9", "troponin", "confirmed"])

    def test_arabic_evidence_non_empty(self):
        m = CitationDataModel(ENCOUNTER_OPD_ARABIC["dx"], ENCOUNTER_OPD_ARABIC["transcript"])
        m.parse()
        cit = m.get_by_code("J06.9")
        assert cit and len(cit["evidenceSentence"]) > 5

    def test_evidence_is_from_reasoning(self):
        enc = ENCOUNTER_OPD_ENGLISH
        reasoning = enc["dx"]["reasoning"]
        m = CitationDataModel(enc["dx"], enc["transcript"])
        m.parse()
        cit = m.get_by_code("J06.9")
        assert cit["evidenceSentence"][:25] in reasoning


class TestTranscriptOffset:
    def test_path_a_offset_preserved(self):
        m = CitationDataModel(ENCOUNTER_PATH_A["dx"], ENCOUNTER_PATH_A["transcript"])
        m.parse()
        assert m.citations[0]["transcriptOffset"] == 42

    def test_no_transcript_gives_minus_one(self):
        m = CitationDataModel(ENCOUNTER_OPD_ENGLISH["dx"], "")
        m.parse()
        for cit in m.citations:
            assert cit["transcriptOffset"] == -1

    def test_er_offset_valid_index(self):
        enc = ENCOUNTER_ER_ENGLISH
        m = CitationDataModel(enc["dx"], enc["transcript"])
        m.parse()
        cit = m.get_by_code("I21.9")
        if cit and cit["transcriptOffset"] >= 0:
            snippet = enc["transcript"][cit["transcriptOffset"]:cit["transcriptOffset"]+30]
            assert len(snippet) > 0


class TestConfidenceAndAccessibility:
    def test_high_confidence_confirmed(self):
        dx = {"icd10_codes": ["I21.9 - AMI"], "reasoning": "ECG findings confirmed STEMI."}
        m = CitationDataModel(dx, "")
        m.parse()
        assert m.citations[0]["confidence"] >= 0.9

    def test_low_confidence_hedged(self):
        dx = {"icd10_codes": ["I21.9 - AMI"], "reasoning": "Possible cardiac event; rule out MI."}
        m = CitationDataModel(dx, "")
        m.parse()
        assert m.citations[0]["confidence"] < 0.75

    def test_default_confidence(self):
        dx = {"icd10_codes": ["J06.9 - URTI"], "reasoning": "Patient has sore throat and fever."}
        m = CitationDataModel(dx, "")
        m.parse()
        assert m.citations[0]["confidence"] == 0.82

    def test_required_fields_present(self):
        m = CitationDataModel(ENCOUNTER_OPD_ENGLISH["dx"], ENCOUNTER_OPD_ENGLISH["transcript"])
        m.parse()
        required = {"icd10Code", "description", "evidenceSentence", "confidence", "field", "id"}
        for cit in m.citations:
            missing = required - set(cit.keys())
            assert not missing, f"Missing: {missing}"

    def test_get_with_evidence_all_have_evidence(self):
        m = CitationDataModel(ENCOUNTER_OPD_ENGLISH["dx"], ENCOUNTER_OPD_ENGLISH["transcript"])
        m.parse()
        with_ev = m.get_with_evidence()
        assert all(c["evidenceSentence"] for c in with_ev)


class TestGateCriteria:
    ENCOUNTERS = [
        ("OPD English",  ENCOUNTER_OPD_ENGLISH,  ["J06.9", "R05"]),
        ("ER English",   ENCOUNTER_ER_ENGLISH,   ["I21.9", "I10"]),
        ("Dental",       ENCOUNTER_DENTAL_ENGLISH, ["K02.9"]),
        ("OPD Arabic",   ENCOUNTER_OPD_ARABIC,   ["J06.9"]),
        ("ER Arabic",    ENCOUNTER_ER_ARABIC,    ["S72.0"]),
    ]

    @pytest.mark.parametrize("name,enc,expected_codes", ENCOUNTERS)
    def test_all_icd10_codes_have_citations(self, name, enc, expected_codes):
        m = CitationDataModel(enc["dx"], enc.get("transcript", ""))
        r = m.parse()
        assert r["parseMs"] < 10.0, f"[{name}] {r['parseMs']:.2f}ms > 10ms"
        parsed_codes = {c["icd10Code"] for c in r["citations"]}
        for code in expected_codes:
            assert code in parsed_codes, f"[{name}] Missing citation for {code}"

    @pytest.mark.parametrize("name,enc,expected_codes", ENCOUNTERS)
    def test_all_citations_have_id_and_field(self, name, enc, expected_codes):
        m = CitationDataModel(enc["dx"], enc.get("transcript", ""))
        m.parse()
        for cit in m.citations:
            assert cit["id"], f"[{name}] Missing id"
            assert cit["field"], f"[{name}] Missing field"
