"""
cleaner.py — Vendor-name, domain, and country normalisation utilities.

Improvements over the original version
---------------------------------------
- Unicode NFKD normalisation + combining-mark removal  (ü → u, é → e …)
- ~40 international legal-suffix patterns stripped
- Country normalisation with a human-name → ISO-3166 alpha-2 look-up
- Flexible field mapping so the same helper works for CSV rows *and* user payloads
"""

import re
import unicodedata
import tldextract

# ---------------------------------------------------------------------------
# Legal suffixes — international set
# ---------------------------------------------------------------------------
_LEGAL_SUFFIXES = [
    # English
    "ltd",
    "limited",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "llc",
    "llp",
    "plc",
    "lp",
    # German / Austrian / Swiss
    "gmbh",
    "mbh",
    "ag",
    "kg",
    "ohg",
    "ug",
    "ev",
    "e v",
    # Scandinavian
    "as",
    "a s",
    "aps",
    "ab",
    "oyj",
    "oy",
    "hf",
    # French
    "sa",
    "sarl",
    "sas",
    "eurl",
    "sci",
    # Dutch / Belgian
    "nv",
    "bv",
    "vof",
    "cv",
    # Italian / Spanish / Portuguese
    "srl",
    "spa",
    "slu",
    "sl",
    "ltda",
    # Indian
    "pvt",
    "private",
    # Hungarian / Czech / Slovak
    "kft",
    "a s",
    "sro",
    # Other
    "pty",
    "pte",
    "bhd",
    "sdn",
]

LEGAL_SUFFIX_REGEX = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in _LEGAL_SUFFIXES) + r")\b\.?",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Country normalisation
# ---------------------------------------------------------------------------
_COUNTRY_NAME_MAP: dict[str, str] = {
    "india": "IN",
    "united states": "US",
    "usa": "US",
    "us": "US",
    "united kingdom": "GB",
    "uk": "GB",
    "germany": "DE",
    "deutschland": "DE",
    "france": "FR",
    "italy": "IT",
    "spain": "ES",
    "netherlands": "NL",
    "belgium": "BE",
    "switzerland": "CH",
    "sweden": "SE",
    "norway": "NO",
    "denmark": "DK",
    "finland": "FI",
    "austria": "AT",
    "australia": "AU",
    "canada": "CA",
    "japan": "JP",
    "china": "CN",
    "brazil": "BR",
    "south korea": "KR",
    "korea": "KR",
    "mexico": "MX",
    "singapore": "SG",
    "hong kong": "HK",
    "ireland": "IE",
    "portugal": "PT",
    "poland": "PL",
    "czech republic": "CZ",
    "czechia": "CZ",
    "hungary": "HU",
    "new zealand": "NZ",
    "south africa": "ZA",
    "israel": "IL",
    "united arab emirates": "AE",
    "uae": "AE",
    "saudi arabia": "SA",
    "taiwan": "TW",
    "indonesia": "ID",
    "malaysia": "MY",
    "philippines": "PH",
    "thailand": "TH",
    "vietnam": "VN",
    "argentina": "AR",
    "chile": "CL",
    "colombia": "CO",
    "peru": "PE",
    "turkey": "TR",
    "russia": "RU",
    "ukraine": "UA",
    "romania": "RO",
    "greece": "GR",
    "luxembourg": "LU",
    "iceland": "IS",
    "estonia": "EE",
    "latvia": "LV",
    "lithuania": "LT",
    "croatia": "HR",
    "serbia": "RS",
    "slovakia": "SK",
    "slovenia": "SI",
    "bulgaria": "BG",
    "egypt": "EG",
    "nigeria": "NG",
    "kenya": "KE",
    "pakistan": "PK",
    "bangladesh": "BD",
    "sri lanka": "LK",
    "mars": "",
}


# ── public helpers ────────────────────────────────────────────────────────


def clean_company_name(raw_name: str) -> str:
    """Normalise a company name for fuzzy comparison.

    1. NFKD-decompose → strip combining marks  (ü→u, é→e, …)
    2. Lower-case
    3. Replace all non-word/non-space chars with a space
    4. Strip legal suffixes
    5. Collapse whitespace
    """
    if not isinstance(raw_name, str) or not raw_name.strip():
        return ""

    # Unicode → ASCII-safe transliteration
    name = unicodedata.normalize("NFKD", raw_name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))

    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)  # punctuation → space
    name = LEGAL_SUFFIX_REGEX.sub("", name)  # strip legal suffixes
    name = re.sub(r"\d{4,}", "", name)  # drop long numeric IDs
    return re.sub(r"\s+", " ", name).strip()


offline_extractor = tldextract.TLDExtract(suffix_list_urls=())


def extract_root_domain(raw_url: str) -> str:
    """Extract `domain.tld` from a URL, email address, or bare domain."""
    if not isinstance(raw_url, str) or not raw_url.strip():
        return ""

    url = raw_url.strip()

    # Handle email addresses
    if "@" in url:
        url = url.split("@")[-1]

    # USE THE OFFLINE EXTRACTOR HERE
    extracted = offline_extractor(url)

    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()
    return url.lower().strip()


def normalize_country(country_str: str) -> str:
    """Return an upper-case ISO-3166 alpha-2 code.

    Accepts full country names ("India"), common aliases ("USA"),
    or already-valid 2-letter codes ("IN").
    """
    if not isinstance(country_str, str) or not country_str.strip():
        return ""

    key = country_str.strip().lower()

    # Direct look-up by name / alias
    if key in _COUNTRY_NAME_MAP:
        return _COUNTRY_NAME_MAP[key]

    # Already a 2-letter code?
    if len(key) == 2 and key.isalpha():
        return key.upper()

    return country_str.strip().upper()[:2]


def clean_record(raw_record: dict, field_map: dict | None = None) -> dict:
    """Clean an input payload according to *field_map*.

    Parameters
    ----------
    raw_record : dict
        The raw input — either a CSV row or a user-supplied payload.
    field_map : dict, optional
        Maps canonical keys (``name``, ``website``, ``country``) to the
        actual keys present in *raw_record*.  Defaults to
        ``{"name": "name", "website": "website", "country": "country"}``.

    Returns
    -------
    dict with keys ``clean_name``, ``clean_domain``, ``country``.
    """
    if field_map is None:
        field_map = {"name": "name", "website": "website", "country": "country"}

    return {
        "clean_name": clean_company_name(raw_record.get(field_map["name"], "")),
        "clean_domain": extract_root_domain(raw_record.get(field_map["website"], "")),
        "country": normalize_country(raw_record.get(field_map["country"], "")),
    }
