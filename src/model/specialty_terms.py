"""
Specialty-aware terminology correction.

Two independent things live here:
  1. SPECIALTIES / SPECIALTY_GLOSSARIES — the fixed list the doctor picks from,
     and a starter glossary of canonical terms per specialty (English + Arabic).
  2. correct_specialty_terms() — a deterministic (non-LLM) fuzzy-matching pass
     that scans transcript text for near-misses of those canonical terms
     (the kind of thing ASR garbles — "hypertention", "برونكيت" for
     "برونشيت") and swaps in the correct spelling.

This is intentionally separate from the LLM refinement step. The LLM prompt
gets told which specialty it's in (see utils/prompt.py) and uses its general
medical knowledge; this module is a cheap, deterministic safety net specific
to the glossary you maintain below, and — because it doesn't need an LLM
call — it also helps Doctor-mode transcripts, which skip LLM refinement
entirely.

IMPORTANT: the glossaries below are a starting point, not a clinically
validated vocabulary. Review and extend them (drug names, abbreviations,
local usage) before leaning on this in production — a wrong fuzzy correction
silently changes clinical meaning, so keep the threshold conservative and
spot-check corrected transcripts while you tune it.

Requires: rapidfuzz  (pip install rapidfuzz --break-system-packages)
"""

import logging
import re
from typing import Dict, List, Optional

from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

SPECIALTIES: List[str] = [
    "Dentistry",
    "Pediatric",
    "Internal Medicine",
    "Dermatology",
    "Ophthalmology",
    "General Surgery",
    "Pain Management",
    "OB & GYN",
    "Pulmonary",
    "Neurology",
    "ENT",
    "Rheumatology",
    "Urology",
    "Cardiology",
    "Orthopedics",
]

# Starter glossaries — English + Arabic canonical terms per specialty.
# Extend freely; order doesn't matter, duplicates across specialties are fine.
SPECIALTY_GLOSSARIES: Dict[str, List[str]] = {
    "Dentistry": [
        "caries", "periodontitis", "gingivitis", "root canal", "dental abscess",
        "malocclusion", "extraction", "crown", "bridge", "dental implant",
        "wisdom tooth", "plaque", "enamel", "dentin", "pulp", "occlusion",
        "bruxism", "orthodontics", "periodontal", "endodontic",
        "تسوس الأسنان", "التهاب اللثة", "التهاب دواعم السن", "علاج عصب",
        "خراج سني", "سوء الإطباق", "خلع", "تاج", "جسر", "ضرس", "زراعة أسنان",
        "ضرس العقل", "طبقة الجير", "مينا الأسنان", "العاج", "لب السن",
    ],
    "Pediatric": [
        "otitis media", "febrile seizure", "croup", "bronchiolitis", "failure to thrive",
        "immunization", "growth chart", "well-child visit", "colic", "jaundice",
        "developmental milestone", "hand-foot-mouth disease", "pertussis", "eczema flare",
        "التهاب الأذن الوسطى", "تشنج حراري", "الخانوق", "التهاب القصيبات",
        "تأخر النمو", "التطعيم", "اليرقان", "مرحلة نمو", "السعال الديكي",
    ],
    "Internal Medicine": [
        "hypertension", "diabetes mellitus", "hyperlipidemia", "atrial fibrillation",
        "chronic kidney disease", "hypothyroidism", "anemia", "congestive heart failure",
        "type 2 diabetes", "coronary artery disease", "metabolic syndrome", "gastritis",
        "hyperthyroidism", "dyslipidemia",
        "ارتفاع ضغط الدم", "داء السكري", "ارتفاع الدهون", "الرجفان الأذيني",
        "الفشل الكلوي المزمن", "قصور الغدة الدرقية", "فقر الدم", "قصور القلب",
        "التهاب المعدة",
    ],
    "Dermatology": [
        "eczema", "psoriasis", "atopic dermatitis", "urticaria", "melanoma",
        "basal cell carcinoma", "acne vulgaris", "rosacea", "vitiligo", "alopecia areata",
        "tinea corporis", "contact dermatitis", "seborrheic dermatitis",
        "الأكزيما", "الصدفية", "التهاب الجلد التأتبي", "الشرى", "الورم الميلانيني",
        "حب الشباب", "الوردية", "البهاق", "الثعلبة",
    ],
    "Ophthalmology": [
        "glaucoma", "cataract", "macular degeneration", "diabetic retinopathy",
        "conjunctivitis", "keratitis", "retinal detachment", "strabismus",
        "uveitis", "dry eye syndrome", "refractive error", "intraocular pressure",
        "الزرق", "الماء الأبيض", "الضمور البقعي", "اعتلال الشبكية السكري",
        "التهاب الملتحمة", "انفصال الشبكية", "الحول", "جفاف العين",
    ],
    "General Surgery": [
        "appendectomy", "cholecystectomy", "hernia repair", "laparotomy",
        "laparoscopic surgery", "bowel obstruction", "peritonitis", "abscess drainage",
        "wound dehiscence", "anastomosis", "cholelithiasis", "acute abdomen",
        "استئصال الزائدة الدودية", "استئصال المرارة", "إصلاح الفتق", "انسداد الأمعاء",
        "التهاب الصفاق", "تصريف الخراج", "حصى المرارة", "بطن حاد",
    ],
    "Pain Management": [
        "chronic pain", "neuropathic pain", "fibromyalgia", "nerve block",
        "epidural steroid injection", "radiculopathy", "sciatica", "myofascial pain",
        "opioid tolerance", "trigger point injection", "complex regional pain syndrome",
        "الألم المزمن", "الألم العصبي", "الفيبروميالجيا", "حصار عصبي",
        "حقنة الستيرويد فوق الجافية", "اعتلال الجذور", "عرق النسا",
    ],
    "OB & GYN": [
        "preeclampsia", "gestational diabetes", "ectopic pregnancy", "endometriosis",
        "polycystic ovary syndrome", "menorrhagia", "amenorrhea", "cervical dysplasia",
        "placenta previa", "postpartum hemorrhage", "fibroids", "pelvic inflammatory disease",
        "تسمم الحمل", "سكري الحمل", "الحمل خارج الرحم", "الانتباذ البطاني الرحمي",
        "متلازمة تكيس المبايض", "غزارة الطمث", "انقطاع الطمث", "المشيمة المنزاحة",
    ],
    "Pulmonary": [
        "chronic obstructive pulmonary disease", "asthma exacerbation", "pneumonia",
        "pulmonary embolism", "pleural effusion", "bronchitis", "pulmonary fibrosis",
        "tuberculosis", "sleep apnea", "spirometry", "interstitial lung disease",
        "الانسداد الرئوي المزمن", "نوبة ربو", "الالتهاب الرئوي", "الانصمام الرئوي",
        "الانصباب الجنبي", "التهاب الشعب الهوائية", "التليف الرئوي", "السل الرئوي",
        "انقطاع النفس النومي",
    ],
    "Neurology": [
        "migraine", "epilepsy", "stroke", "transient ischemic attack", "multiple sclerosis",
        "parkinson's disease", "peripheral neuropathy", "seizure disorder",
        "dementia", "bell's palsy", "myasthenia gravis", "vertigo",
        "الصداع النصفي", "الصرع", "السكتة الدماغية", "نوبة إقفارية عابرة",
        "التصلب المتعدد", "مرض باركنسون", "اعتلال الأعصاب الطرفية", "الخرف",
        "شلل بيل", "الدوار",
    ],
    "ENT": [
        "tonsillitis", "sinusitis", "otitis externa", "otitis media", "pharyngitis",
        "laryngitis", "deviated septum", "vertigo", "tinnitus", "hearing loss",
        "adenoid hypertrophy", "epistaxis", "allergic rhinitis", "vocal cord nodules",
        "sleep apnea", "mastoiditis",
        "التهاب اللوزتين", "التهاب الجيوب الأنفية", "التهاب الأذن الخارجية",
        "التهاب الأذن الوسطى", "التهاب البلعوم", "التهاب الحنجرة", "انحراف الحاجز الأنفي",
        "طنين الأذن", "فقدان السمع", "تضخم اللحمية", "نزيف الأنف", "حساسية الأنف",
    ],
    "Rheumatology": [
        "rheumatoid arthritis", "osteoarthritis", "systemic lupus erythematosus", "gout",
        "ankylosing spondylitis", "psoriatic arthritis", "sjogren's syndrome",
        "fibromyalgia", "vasculitis", "polymyalgia rheumatica", "scleroderma",
        "antinuclear antibody", "rheumatoid factor",
        "التهاب المفاصل الروماتويدي", "الفصال العظمي", "الذئبة الحمامية الجهازية", "النقرس",
        "التهاب الفقار اللاصق", "التهاب المفاصل الصدفي", "متلازمة شوغرن",
        "الفيبروميالجيا", "التهاب الأوعية الدموية", "تصلب الجلد",
    ],
    "Urology": [
        "urinary tract infection", "benign prostatic hyperplasia", "kidney stone",
        "urinary incontinence", "erectile dysfunction", "hematuria", "prostatitis",
        "urinary retention", "nephrolithiasis", "hydronephrosis", "overactive bladder",
        "testicular torsion", "varicocele", "prostate specific antigen",
        "التهاب المسالك البولية", "تضخم البروستاتا الحميد", "حصوة الكلى",
        "سلس البول", "ضعف الانتصاب", "بيلة دموية", "التهاب البروستاتا",
        "احتباس البول", "استسقاء الكلية", "دوالي الخصية",
    ],
    "Cardiology": [
        "myocardial infarction", "atrial fibrillation", "congestive heart failure",
        "coronary artery disease", "hypertension", "angina pectoris", "arrhythmia",
        "cardiomyopathy", "hyperlipidemia", "aortic stenosis", "mitral regurgitation",
        "deep vein thrombosis", "pulmonary embolism", "ejection fraction",
        "احتشاء عضلة القلب", "الرجفان الأذيني", "قصور القلب الاحتقاني",
        "مرض الشريان التاجي", "ارتفاع ضغط الدم", "الذبحة الصدرية", "اضطراب نظم القلب",
        "اعتلال عضلة القلب", "تضيق الأبهر", "قصور الصمام التاجي", "تجلط الأوردة العميقة",
    ],
    "Orthopedics": [
        "fracture", "osteoarthritis", "rotator cuff tear", "meniscus tear",
        "anterior cruciate ligament tear", "herniated disc", "osteoporosis",
        "carpal tunnel syndrome", "tendinitis", "bursitis", "scoliosis",
        "dislocation", "sprain", "osteomyelitis", "spinal stenosis",
        "كسر", "الفصال العظمي", "تمزق الكفة المدورة", "تمزق الغضروف الهلالي",
        "تمزق الرباط الصليبي الأمامي", "انزلاق غضروفي", "هشاشة العظام",
        "متلازمة النفق الرسغي", "التهاب الأوتار", "التهاب الجراب", "الجنف", "خلع",
    ],
}

# Cache of {specialty: {ngram_length: {normalized_term: canonical_term}}} —
# built once per specialty on first use. Keying by word count matters: it
# stops e.g. a 3-word window like "has atrial fibrilation" from being scored
# against a 2-word term like "atrial fibrillation" and silently absorbing the
# extra word ("has") into the replacement.
_normalized_cache: Dict[str, Dict[int, Dict[str, str]]] = {}

# Strips Arabic diacritics (tashkeel) and punctuation, keeping word chars in
# both scripts, then lowercases — so "Hypertension," and "hypertension" (or
# "الرَّبْو" and "الربو") compare equal before fuzzy scoring.
_TASHKEEL_RE = re.compile(r"[\u064B-\u0652]")
_STRIP_RE    = re.compile(r"[^\w\s\u0600-\u06FF]")
# Word-character tokens only (letters/digits/apostrophe, Latin + Arabic) —
# deliberately NOT \S+, so trailing punctuation attached to a word (the
# period on "الاوتار.") is never swallowed into a replacement.
_WORD_RE = re.compile(r"[A-Za-z0-9\u0600-\u06FF']+")

# Single-letter Arabic proclitics commonly attached directly to the next word
# with no space — و (and), ف (so/then), ب (with/by/in), ل (to/for), ك (like/as).
# e.g. "والتهاب" = "و" + "التهاب". Tried as an alternate reading during
# matching (see correct_specialty_terms) and, if it's the one that matches,
# reattached to the canonical replacement so it isn't lost.
_ARABIC_PROCLITICS = "وفبلك"


def _normalize(s: str) -> str:
    s = _TASHKEEL_RE.sub("", s)
    s = _STRIP_RE.sub("", s)
    return s.strip().lower()


def _strip_leading_proclitic(norm_phrase: str):
    """
    If norm_phrase's first word starts with a single-letter Arabic proclitic
    attached with no space, return (proclitic_char, rest_with_proclitic_removed).
    Otherwise (None, norm_phrase). Requires at least 2 chars left after
    stripping so we don't turn e.g. "و" alone into an empty candidate.
    """
    if len(norm_phrase) < 3:
        return None, norm_phrase
    first = norm_phrase[0]
    if first in _ARABIC_PROCLITICS and "\u0600" <= norm_phrase[1] <= "\u06FF":
        return first, norm_phrase[1:]
    return None, norm_phrase


def _glossary_for(specialty: str) -> Optional[Dict[int, Dict[str, str]]]:
    if specialty in _normalized_cache:
        return _normalized_cache[specialty]
    terms = SPECIALTY_GLOSSARIES.get(specialty)
    if not terms:
        return None
    by_len: Dict[int, Dict[str, str]] = {}
    for t in terms:
        norm = _normalize(t)
        if not norm:
            continue
        word_count = len(norm.split())
        by_len.setdefault(word_count, {})[norm] = t
    _normalized_cache[specialty] = by_len
    return by_len


def correct_specialty_terms(
    text: str,
    specialty: Optional[str],
    threshold: float = 87.0,
    max_ngram: int = 3,
) -> str:
    """
    Scan `text` for word/phrase spans (1-3 words) that are close-but-not-exact
    matches to a term in `specialty`'s glossary, and replace them with the
    canonical spelling. No-op if specialty is None/unknown or the text is empty.

    A candidate span is only ever scored against glossary terms with the SAME
    word count as the span — never against shorter or longer terms — so an
    extra or missing word (e.g. "has atrial fibrilation" vs. the 2-word term
    "atrial fibrillation") can't be silently absorbed or dropped by the
    replacement. Trailing punctuation attached to a word (a period, comma) is
    never part of the token, so it's never swallowed either. And because
    Arabic attaches a handful of one-letter conjunctions/prepositions directly
    to the next word with no space (e.g. "والتهاب" = "و" + "التهاب"), each
    span is also tried with that leading proclitic stripped off; if THAT
    version is what matches, the proclitic is reattached to the canonical
    replacement so it isn't lost.

    Matching runs longest n-gram first and skips tokens already claimed by an
    earlier (longer) match.

    `threshold` (0-100, rapidfuzz ratio) controls how close a span must be to
    count as a "near miss" worth correcting — kept conservative by default
    since an overzealous correction can silently change clinical meaning.
    """
    if not text or not specialty:
        return text

    glossary_by_len = _glossary_for(specialty)
    if not glossary_by_len:
        return text

    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return text

    n_tokens = len(matches)
    used = [False] * n_tokens
    replacements = []  # (start_char, end_char, canonical_term)

    for n in range(min(max_ngram, n_tokens), 0, -1):
        choices = glossary_by_len.get(n)
        if not choices:
            continue  # no glossary terms of this exact word count — nothing to compare
        choice_norms = list(choices.keys())

        for i in range(0, n_tokens - n + 1):
            if any(used[i:i + n]):
                continue
            span = matches[i:i + n]
            phrase = text[span[0].start():span[-1].end()]
            norm_phrase = _normalize(phrase)
            if not norm_phrase:
                continue

            # Try the phrase as-is, and (for Arabic) with a leading attached
            # proclitic stripped off — score both, keep whichever is closer.
            candidates = [(norm_phrase, "")]
            proclitic, stripped = _strip_leading_proclitic(norm_phrase)
            if proclitic:
                candidates.append((stripped, proclitic))

            best_score, best_canonical = -1.0, None
            for cand_text, reattach_prefix in candidates:
                if not cand_text:
                    continue
                res = process.extractOne(cand_text, choice_norms, scorer=fuzz.ratio)
                if res is None:
                    continue
                cand_norm, score, _ = res
                if cand_norm == cand_text:
                    continue  # already an exact match — nothing to correct here
                if score > best_score:
                    best_score = score
                    best_canonical = reattach_prefix + choices[cand_norm]

            if best_canonical is not None and best_score >= threshold:
                replacements.append((span[0].start(), span[-1].end(), best_canonical))
                for k in range(i, i + n):
                    used[k] = True

    if not replacements:
        return text

    replacements.sort(key=lambda r: r[0])
    out, cursor = [], 0
    for start, end, canonical in replacements:
        out.append(text[cursor:start])
        out.append(canonical)
        cursor = end
        logger.debug("[SpecialtyTerms] Corrected span -> '%s'", canonical)
    out.append(text[cursor:])
    return "".join(out)