"""Shared constants and conservative language-risk patterns."""

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MAX_PDF_BYTES = 25 * 1024 * 1024
DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 180
DEFAULT_RETRIEVAL_K = 6

# These patterns identify language that usually needs surrounding context. They
# are deliberately narrow: matching one is a review signal, not a conclusion.
AMBIGUITY_REGEX_PATTERNS: tuple[str, ...] = (
    r"\b(?:may|might|could)\s+(?:cover|apply|qualify|be payable)\b",
    r"\b(?:reasonable|customary|usual|medically necessary)\b",
    r"\b(?:as determined by|at (?:our|the insurer'?s) discretion)\b",
    r"\bat (?:the )?(?:company(?:'s|’s)?|insurer(?:'s|’s)?|our)\s+(?:sole\s+|absolute\s+)?discretion\b",
    r"\breasonable\s+care\b",
    r"\busual\s+and\s+customary\s+charges\b",
    r"\b(?:where appropriate|if applicable|as required)\b",
    r"\bsubject to\b",
    r"\bunless otherwise (?:stated|specified|agreed)\b",
    r"\b(?:generally|typically|ordinarily|normally)\b",
    r"\b(?:including but not limited to|among other things)\b",
    r"\bpre[ -]?existing condition\b",
    r"\bexperimental or investigational\b",
    r"\bmaterial (?:fact|change|misrepresentation)\b",
)

# Backwards-friendly alias for callers that use the shorter name.
AMBIGUITY_PATTERNS = AMBIGUITY_REGEX_PATTERNS

# Literal phrases are useful for inexpensive full-policy scans. Keep this list
# separate from the regexes so callers do not accidentally escape a regex and
# then search for its punctuation literally.
AMBIGUITY_PHRASES: tuple[str, ...] = (
    "reasonable and customary",
    "usual and customary",
    "medically necessary",
    "at our discretion",
    "at the insurer's discretion",
    "subject to",
    "unless otherwise stated",
    "pre-existing condition",
    "waiting period",
    "sub-limit",
    "co-payment",
    "deductible",
    "experimental or investigational",
    "not covered",
    "excluded",
)
RISK_PHRASES = AMBIGUITY_PHRASES
RISK_PATTERNS = AMBIGUITY_REGEX_PATTERNS

# Query expansion retains the original query and only appends related terms.
CLINICAL_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("phacoemulsification", "cataract surgery", "cataract extraction", "intraocular lens"),
    ("renal dialysis", "kidney dialysis", "hemodialysis", "haemodialysis", "peritoneal dialysis"),
    ("myocardial infarction", "heart attack", "acute coronary syndrome"),
    ("cerebrovascular accident", "stroke", "brain attack"),
    ("coronary artery bypass graft", "cabg", "heart bypass surgery"),
    ("total knee replacement", "knee arthroplasty", "joint replacement", "tkr"),
    ("computed tomography", "ct scan", "cat scan"),
    ("magnetic resonance imaging", "mri"),
)
