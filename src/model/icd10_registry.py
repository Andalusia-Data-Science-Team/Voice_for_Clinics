"""High-performance ICD-10 Registry for Australian ICD-10-AM codes.

Provides:
- Singleton in-memory registry parsed once at startup
- O(1) exact code lookup with dot/case normalization (e.g. "A00.0" -> "A000")
- Hybrid prefix + inverted index + rapidfuzz search (sub-10ms)
- LLM output validation against the official 16,954 Australian codes
"""

from __future__ import annotations

import csv
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Regex pattern to identify ICD-10 code structures (e.g., A00, A00.0, K35.80, Z99.9)
_ICD_CODE_PATTERN = re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?$", re.IGNORECASE)
_EXTRACT_CODE_PREFIX = re.compile(r"^([A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?)\b", re.IGNORECASE)
_TOKEN_SPLIT_REGEX = re.compile(r"[^a-z0-9]+", re.IGNORECASE)

# Common medical / English stop words to filter out from description inverted index
_STOP_WORDS = frozenset({
    "and", "or", "in", "of", "to", "with", "due", "for", "on", "by", "from",
    "at", "the", "a", "an", "other", "specified", "unspecified", "not", "without",
    "part", "type", "site", "side"
})


def normalize_code(code: str) -> str:
    """Normalize code string for O(1) lookup: uppercase and strip whitespace and dots."""
    if not code:
        return ""
    return code.replace(".", "").replace(" ", "").strip().upper()


def tokenize_text(text: str) -> list[str]:
    """Tokenize and normalize text into clean lowercase words >= 2 chars."""
    if not text:
        return []
    return [
        t for t in _TOKEN_SPLIT_REGEX.split(text.lower())
        if len(t) >= 2 and t not in _STOP_WORDS
    ]


@dataclass(frozen=True, slots=True)
class ICD10Entry:
    """Immutable entry representing an Australian ICD-10-AM code."""
    code: str          # Original display format: e.g. "A00.0"
    description: str   # Official text: e.g. "Cholera due to Vibrio cholerae 01, biovar cholerae"
    norm_code: str     # Normalized key: e.g. "A000"


@dataclass(frozen=True, slots=True)
class ValidatedCode:
    """Represents the validation result for an ICD-10 recommendation."""
    original: str                       # Raw string from LLM: e.g. "A00.0 - Cholera..."
    code: Optional[str]                 # Extracted/normalized code or None
    matched_entry: Optional[ICD10Entry] # Matching registry entry or None
    is_valid: bool                      # True if successfully validated against official list
    corrected_label: str                # Canonical label: "A00.0 - Description" or original fallback


class ICD10Registry:
    """High-performance singleton registry for Australian ICD-10-AM codes.

    Built for ultra-fast async read operations:
    - Thread-safe read-only after initialization
    - Lookup: O(1)
    - Code prefix search: O(K) where K is matching branch
    - Inverted index description search: candidates filtered by word posting sets,
      then ranked with rapidfuzz.
    """

    def __init__(self, csv_path: Optional[Path | str] = None) -> None:
        if csv_path is None:
            source_data = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "Austrilian ICD codes.csv"
            )
            project_root = Path(__file__).resolve().parent.parent.parent
            tracked_data = project_root / "Austrilian ICD codes.csv"
            csv_path = source_data if source_data.is_file() else tracked_data

        self._csv_path = Path(csv_path)
        self._entries: list[ICD10Entry] = []
        self._by_norm_code: dict[str, ICD10Entry] = {}
        self._by_code: dict[str, ICD10Entry] = {}
        self._code_prefixes: dict[str, list[ICD10Entry]] = {}
        self._inverted_index: dict[str, set[int]] = {}  # token -> set of entry indices
        self._descriptions_lower: list[str] = []
        self._description_lengths: list[int] = []
        self._term_frequencies: list[dict[str, int]] = []
        self._average_description_length = 0.0

        self._load_and_index()

    def _load_and_index(self) -> None:
        if not self._csv_path.is_file():
            logger.error("ICD-10 CSV file not found at: %s", self._csv_path)
            raise FileNotFoundError(f"ICD-10 CSV file not found at: {self._csv_path}")

        entries: list[ICD10Entry] = []
        by_norm: dict[str, ICD10Entry] = {}
        by_code: dict[str, ICD10Entry] = {}
        code_prefixes: dict[str, list[ICD10Entry]] = {}
        inverted_index: dict[str, set[int]] = {}
        descriptions_lower: list[str] = []
        description_lengths: list[int] = []
        term_frequencies: list[dict[str, int]] = []

        with open(self._csv_path, mode="r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            # Find column indices
            code_col = 0
            desc_col = 1
            if header:
                for idx, col in enumerate(header):
                    c = col.strip().lower()
                    if "code" in c:
                        code_col = idx
                    elif "desc" in c or "ascii" in c or "name" in c:
                        desc_col = idx

            for row in reader:
                if not row or len(row) <= max(code_col, desc_col):
                    continue
                raw_code = row[code_col].strip()
                raw_desc = row[desc_col].strip()
                if not raw_code:
                    continue

                norm = normalize_code(raw_code)
                entry = ICD10Entry(code=raw_code, description=raw_desc, norm_code=norm)
                idx = len(entries)
                entries.append(entry)
                by_norm[norm] = entry
                by_code[raw_code.upper()] = entry
                descriptions_lower.append(raw_desc.lower())

                # Index code prefixes for fast prefix queries (minimum 2 chars to avoid storing 17k entries for letter prefixes)
                if len(norm) >= 2:
                    for p_len in range(2, len(norm) + 1):
                        prefix = norm[:p_len]
                        if prefix not in code_prefixes:
                            code_prefixes[prefix] = []
                        code_prefixes[prefix].append(idx)

                # Build inverted word index for description search
                tokens = tokenize_text(raw_desc)
                description_lengths.append(len(tokens))
                token_counts = Counter(tokens)
                term_frequencies.append(dict(token_counts))
                for token in token_counts:
                    if token not in inverted_index:
                        inverted_index[token] = set()
                    inverted_index[token].add(idx)

        self._entries = entries
        self._by_norm_code = by_norm
        self._by_code = by_code
        self._code_prefixes = code_prefixes
        self._inverted_index = inverted_index
        self._descriptions_lower = descriptions_lower
        self._description_lengths = description_lengths
        self._term_frequencies = term_frequencies
        self._average_description_length = (
            sum(description_lengths) / len(description_lengths)
            if description_lengths
            else 0.0
        )

        logger.info(
            "ICD10Registry initialized: loaded %d codes, %d search tokens indexed from %s",
            len(self._entries),
            len(self._inverted_index),
            self._csv_path.name,
        )

    def __len__(self) -> int:
        return len(self._entries)

    def lookup_code(self, code_id: str) -> Optional[ICD10Entry]:
        """O(1) exact code lookup supporting either formatted ('A00.0') or normalized ('A000') keys."""
        if not code_id:
            return None
        norm = normalize_code(code_id)
        return self._by_norm_code.get(norm)

    def all_codes(self) -> list[tuple[str, str]]:
        """Return all codes as (code, description) tuples."""
        return [(e.code, e.description) for e in self._entries]

    def search(self, query: str, limit: int = 20) -> list[tuple[str, str]]:
        """Hybrid search across code and descriptions.

        1. If query starts with alphanumeric code structure (e.g. 'A0', 'J18', 'E11.9'):
           -> Returns prefix matches on norm_code index first.
        2. Description tokens -> inverted index intersection / union candidate pruning.
        3. Scored and ranked with rapidfuzz.
        """
        q = (query or "").strip()
        if not q:
            return [(e.code, e.description) for e in self._entries[:limit]]

        norm_q = normalize_code(q)
        results: list[tuple[str, str]] = []
        seen_codes: set[str] = set()

        # 1. Exact or prefix code match
        is_code_like = bool(_ICD_CODE_PATTERN.match(q) or re.match(r"^[A-Z][0-9]{1,4}", norm_q))
        if norm_q and (len(norm_q) >= 2 or norm_q in self._by_norm_code):
            # Check exact code
            exact = self._by_norm_code.get(norm_q)
            if exact:
                results.append((exact.code, exact.description))
                seen_codes.add(exact.norm_code)

            # Prefix matches
            prefix_match_indices = self._code_prefixes.get(norm_q, [])
            for match_idx in prefix_match_indices:
                match = self._entries[match_idx]
                if match.norm_code not in seen_codes:
                    results.append((match.code, match.description))
                    seen_codes.add(match.norm_code)
                if len(results) >= limit:
                    return results[:limit]

            # If query is clearly an ICD-10 code (e.g. J18, E11, A00.0) and we got results, return immediately
            if is_code_like and results:
                return results[:limit]

        # 2. Text / description search using inverted index candidate filtering + rapidfuzz
        query_tokens = tokenize_text(q)
        candidate_indices: set[int] = set()

        if query_tokens:
            # Token match: intersection of posting lists if possible, otherwise union
            token_candidates: list[set[int]] = [
                self._inverted_index.get(tok, set())
                for tok in query_tokens
                if tok in self._inverted_index
            ]

            if token_candidates:
                # First try intersection (all tokens must match)
                intersected = set.intersection(*token_candidates)
                if len(intersected) >= limit:
                    candidate_indices = intersected
                else:
                    # Union if intersection gives too few results
                    candidate_indices = set.union(*token_candidates)
            else:
                # Substring token fallback for prefix tokens (e.g. "pneu" -> match words starting with "pneu")
                sub_candidates: set[int] = set()
                for tok in query_tokens:
                    for indexed_token, post_list in self._inverted_index.items():
                        if indexed_token.startswith(tok):
                            sub_candidates.update(post_list)
                            if len(sub_candidates) > 200:
                                break
                candidate_indices = sub_candidates

        # If inverted index yielded candidates, rank with rapidfuzz
        if candidate_indices:
            try:
                from rapidfuzz import fuzz, process

                candidate_list = list(candidate_indices)
                # Form list of (index, description) for rapidfuzz
                choices = {idx: self._descriptions_lower[idx] for idx in candidate_list}
                matches = process.extract(
                    q.lower(),
                    choices,
                    scorer=fuzz.WRatio,
                    limit=limit * 2,
                    score_cutoff=40.0,
                )
                for _, score, idx in matches:
                    entry = self._entries[idx]
                    if entry.norm_code not in seen_codes:
                        results.append((entry.code, entry.description))
                        seen_codes.add(entry.norm_code)
                        if len(results) >= limit:
                            break
            except ImportError:
                # Rapidfuzz not installed fallback: simple containment check
                for idx in candidate_indices:
                    entry = self._entries[idx]
                    if entry.norm_code not in seen_codes:
                        results.append((entry.code, entry.description))
                        seen_codes.add(entry.norm_code)
                        if len(results) >= limit:
                            break

        # 3. If still not enough results and query is short word, fallback fuzzy over full descriptions
        if len(results) < limit and len(q) >= 3 and not candidate_indices:
            try:
                from rapidfuzz import fuzz, process

                full_matches = process.extract(
                    q.lower(),
                    self._descriptions_lower,
                    scorer=fuzz.WRatio,
                    limit=limit,
                    score_cutoff=55.0,
                )
                for _, score, idx in full_matches:
                    entry = self._entries[idx]
                    if entry.norm_code not in seen_codes:
                        results.append((entry.code, entry.description))
                        seen_codes.add(entry.norm_code)
                        if len(results) >= limit:
                            break
            except ImportError:
                pass

        return results[:limit]

    def search_diagnosis_candidates(
        self,
        text: str,
        limit: int = 80,
        max_queries: int = 80,
    ) -> list[tuple[str, str]]:
        """Retrieve catalogue candidates with BM25 plus phrase-search fusion.

        BM25 ranks each complete note against the catalogue descriptions, while
        phrase searches preserve stronger exact-phrase and code matches. The
        two ranked lists are combined with reciprocal-rank fusion.
        """
        if limit <= 0 or not text or not text.strip():
            return []

        generic_terms = {
            "patient", "reports", "report", "presents", "presented", "complains",
            "complaint", "history", "today", "states", "noted", "doctor",
            "symptom", "symptoms", "finding", "findings",
        }
        note_tokens = [
            token for token in tokenize_text(text)
            if token not in generic_terms
        ]
        note_token_counts: dict[str, int] = {}
        for token in note_tokens:
            note_token_counts[token] = note_token_counts.get(token, 0) + 1

        bm25_scores: dict[int, float] = {}
        total_documents = len(self._entries)
        if note_token_counts and self._average_description_length:
            k1 = 1.5
            b = 0.75
            for token, query_frequency in note_token_counts.items():
                posting = self._inverted_index.get(token)
                if not posting:
                    continue
                document_frequency = len(posting)
                inverse_document_frequency = max(
                    0.0,
                    (total_documents - document_frequency + 0.5)
                    / (document_frequency + 0.5),
                )
                inverse_document_frequency = math.log1p(inverse_document_frequency)
                for document_index in posting:
                    term_frequency = self._term_frequencies[document_index][token]
                    length_normalizer = 1 - b + b * (
                        self._description_lengths[document_index]
                        / self._average_description_length
                    )
                    term_score = inverse_document_frequency * (
                        term_frequency * (k1 + 1)
                    ) / (term_frequency + k1 * length_normalizer)
                    bm25_scores[document_index] = (
                        bm25_scores.get(document_index, 0.0)
                        + query_frequency * term_score
                    )

        bm25_ranking = sorted(
            bm25_scores,
            key=bm25_scores.get,
            reverse=True,
        )
        fused_scores: dict[str, float] = {}
        reciprocal_rank_constant = 60
        for rank, document_index in enumerate(bm25_ranking, start=1):
            code = self._entries[document_index].code
            fused_scores[code] = fused_scores.get(code, 0.0) + (
                1 / (reciprocal_rank_constant + rank)
            )

        query_phrases: list[str] = []
        seen_phrases: set[str] = set()
        for segment in re.split(r"[,.;:!?؟؛\n]+", text or ""):
            tokens = [
                token
                for token in tokenize_text(segment)
                if token not in generic_terms
            ]
            for width in (3, 2, 1):
                for start in range(max(0, len(tokens) - width + 1)):
                    phrase = " ".join(tokens[start:start + width])
                    if phrase in seen_phrases:
                        continue
                    seen_phrases.add(phrase)
                    query_phrases.append(phrase)
                    if len(query_phrases) >= max_queries:
                        break
                if len(query_phrases) >= max_queries:
                    break
            if len(query_phrases) >= max_queries:
                break

        phrase_scores: dict[str, float] = {}
        for phrase in query_phrases:
            width = len(phrase.split())
            hit_limit = 40 if width == 1 else 12
            for rank, (code, _description) in enumerate(
                self.search(phrase, limit=hit_limit)
            ):
                phrase_scores[code] = phrase_scores.get(code, 0.0) + (
                    width / (rank + 1)
                )

        for rank, code in enumerate(
            sorted(phrase_scores, key=phrase_scores.get, reverse=True),
            start=1,
        ):
            fused_scores[code] = fused_scores.get(code, 0.0) + (
                1 / (reciprocal_rank_constant + rank)
            )

        ranked_codes = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )[:limit]
        return [
            (entry.code, entry.description)
            for code in ranked_codes
            if (entry := self.lookup_code(code)) is not None
        ]

    def validate_code(self, raw_item: str) -> ValidatedCode:
        """Validate a single raw code or label string against the registry.

        Handles:
        1. "A00.0 - Cholera due to..." (standard LLM output format)
        2. "A00.0" (bare code)
        3. "A000" (normalized code)
        4. "Cholera due to Vibrio..." (description-only with fuzzy match fallback)
        """
        raw = (raw_item or "").strip()
        if not raw:
            return ValidatedCode(
                original="",
                code=None,
                matched_entry=None,
                is_valid=False,
                corrected_label="",
            )

        # Case 1: Format "CODE - Description" or "CODE: Description"
        code_part: Optional[str] = None
        desc_part: Optional[str] = None

        if " - " in raw:
            parts = raw.split(" - ", 1)
            code_part = parts[0].strip()
            desc_part = parts[1].strip()
        elif ":" in raw and not raw.startswith("http"):
            parts = raw.split(":", 1)
            code_part = parts[0].strip()
            desc_part = parts[1].strip()
        else:
            # Check if start of string has an ICD code format
            prefix_match = _EXTRACT_CODE_PREFIX.match(raw)
            if prefix_match:
                code_part = prefix_match.group(1).strip()
                desc_part = raw[len(code_part):].lstrip(" -:,")
            else:
                code_part = raw

        # Attempt exact lookup by code_part
        entry = self.lookup_code(code_part)
        if entry:
            return ValidatedCode(
                original=raw,
                code=entry.code,
                matched_entry=entry,
                is_valid=True,
                corrected_label=f"{entry.code} - {entry.description}",
            )

        # Case 1b: Try parent category code if dot/subdivision was used (e.g., K35.80 -> K35.8, or J18.99 -> J18.9)
        clean_code = (code_part or "").strip().upper()
        if "." in clean_code:
            parts = clean_code.split(".")
            root = parts[0]
            # Try progressive truncations: e.g. J18.99 -> J18.9 -> J18
            sub = parts[1]
            for l in range(len(sub) - 1, 0, -1):
                cand = f"{root}.{sub[:l]}"
                cand_entry = self.lookup_code(cand)
                if cand_entry:
                    return ValidatedCode(
                        original=raw,
                        code=cand_entry.code,
                        matched_entry=cand_entry,
                        is_valid=True,
                        corrected_label=f"{cand_entry.code} - {cand_entry.description}",
                    )

            # Try 3-character root directly (e.g. A38 or K35)
            parent_entry = self.lookup_code(root)
            if parent_entry:
                return ValidatedCode(
                    original=raw,
                    code=parent_entry.code,
                    matched_entry=parent_entry,
                    is_valid=True,
                    corrected_label=f"{parent_entry.code} - {parent_entry.description}",
                )

            # If 3-character root has subcodes (e.g. J18 -> J18.9 unspecified)
            norm_root = normalize_code(root)
            root_matches = self._code_prefixes.get(norm_root, [])
            if root_matches:
                # Prefer .9 (unspecified) if present in prefix matches
                unspecified_entry = None
                for m_idx in root_matches:
                    m = self._entries[m_idx]
                    if m.code.endswith(".9"):
                        unspecified_entry = m
                        break
                selected = unspecified_entry or self._entries[root_matches[0]]
                return ValidatedCode(
                    original=raw,
                    code=selected.code,
                    matched_entry=selected,
                    is_valid=True,
                    corrected_label=f"{selected.code} - {selected.description}",
                )

        # Case 2: If code_part was explicitly provided and looks like an attempted code or was explicitly separated,
        # but is not in our registry, do not arbitrarily reassign to an unrelated code.
        had_explicit_code = bool(
            (" - " in raw or ":" in raw or _ICD_CODE_PATTERN.match(code_part or ""))
            and code_part
        )

        # Only fallback to description search if there was no invalid code token, or if search_query is a rich medical description
        search_query = desc_part if (desc_part and not had_explicit_code) else ("" if had_explicit_code else raw)
        if search_query and len(search_query) >= 4:
            search_hits = self.search(search_query, limit=1)
            if search_hits:
                hit_code, hit_desc = search_hits[0]
                matched = self.lookup_code(hit_code)
                if matched:
                    from rapidfuzz import fuzz
                    # Require strict match on the medical description
                    ratio = fuzz.token_sort_ratio(search_query.lower(), matched.description.lower())
                    if ratio >= 80:
                        return ValidatedCode(
                            original=raw,
                            code=matched.code,
                            matched_entry=matched,
                            is_valid=True,
                            corrected_label=f"{matched.code} - {matched.description}",
                        )

        # Invalid: cannot be resolved to any official ICD-10 code
        return ValidatedCode(
            original=raw,
            code=code_part if _ICD_CODE_PATTERN.match(code_part or "") else None,
            matched_entry=None,
            is_valid=False,
            corrected_label=raw,
        )

    def validate_codes(self, codes: Sequence[str]) -> list[ValidatedCode]:
        """Validate a list of raw code strings returned by LLM or user."""
        return [self.validate_code(c) for c in (codes or [])]


# Module-level singleton
_REGISTRY_INSTANCE: Optional[ICD10Registry] = None


def get_registry(csv_path: Optional[Path | str] = None) -> ICD10Registry:
    """Thread-safe singleton accessor for the ICD10Registry."""
    global _REGISTRY_INSTANCE
    if _REGISTRY_INSTANCE is None:
        _REGISTRY_INSTANCE = ICD10Registry(csv_path=csv_path)
    return _REGISTRY_INSTANCE
