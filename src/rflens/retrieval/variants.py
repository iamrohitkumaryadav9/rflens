"""Clause references in a question, and the variant-clause rule.

**Clause lookup.** A question that says "clause 6.6.2.1" or "Table 7.3.1-1" names its target: the
chunks of that clause or table are fetched directly (like the band lookup), by spec when it is
known and otherwise in both specs. Only a number after the word clause / subclause / section or
Table counts, so "10 MHz" or "15 kHz" never looks like a clause.

**Variant clauses.** 3GPP adds a letter to a clause number for a variant of a requirement: 5.5H is
the operating-band clause of "LTE based 5G terrestrial broadcast", 6.2.4A.7 is an A-MPR clause for
carrier aggregation. They repeat the wording of the base clause, so in a vector search they displace
it (75% of the TS 38.101-1 chunks and 26% of the TS 36.101 chunks belong to a variant clause). A
variant clause is moved behind the others unless the question mentions the topic of its letter. The
letters mean different things in the two specs; the topics below were read off the clause titles
(`make` a query of clause, title and chunk counts by letter; the counts are in docs/design).
A clause with two letters (5.2E.1F) needs both topics. A letter without an entry is never excused.
"""

import re

CLAUSE_RE = re.compile(
    r"\b(?:sub)?(?:clauses?|sections?)\s+(\d+[A-Z]?(?:\.\d+[A-Z]?)*)", re.IGNORECASE
)
TABLE_RE = re.compile(r"\btables?\s+(\d+[A-Z]?(?:\.\d+[A-Z]?)*-\d+[a-z]?)", re.IGNORECASE)
LETTER_RE = re.compile(r"\d+([A-Z]+)")

CA = r"\bCA\b|\bCA_|carrier aggregation|aggregat|inter-band|intra-band|contiguous|\bUL CA\b"
SPATIAL = r"spatial multiplexing|single-layer|\bMIMO\b|demodulation|\bTx antenna|\bRx\b"
MIMO = r"\bMIMO\b|\bUL[- ]MIMO\b"

# (spec, clause letter) -> regular expressions; one match anywhere in the question excuses the
# letter. Read off the clause titles of the loaded specs.
TOPICS: dict[tuple[str, str], tuple[str, ...]] = {
    # TS 36.101 (LTE / E-UTRA)
    ("36.101", "A"): (CA,),  # carrier aggregation, CA_NS_xx
    ("36.101", "B"): (MIMO, SPATIAL),  # UL-MIMO, spatial multiplexing performance
    ("36.101", "C"): (SPATIAL,),  # spatial multiplexing with 2 Tx / 4Rx
    ("36.101", "D"): (r"ProSe|sidelink|proximity|D2D|PSSS|PSBCH",),
    ("36.101", "E"): (
        r"category M|\bM1\b|\bM2\b|eMTC|\bMTC\b|category 0|1bis|bandwidth[- ]reduced",
    ),
    ("36.101", "F"): (r"NB-IoT|\bNB1\b|\bNB2\b|narrowband|\bNB-",),
    ("36.101", "G"): (r"V2X|vehicle",),
    ("36.101", "H"): (r"broadcast|PMCH|5G terrestrial|LTE based 5G",),
    ("36.101", "I"): (SPATIAL,),
    ("36.101", "J"): (SPATIAL,),
    ("36.101", "K"): (r"aerial|UAV|drone",),
    # TS 38.101-1 (NR FR1)
    ("38.101-1", "A"): (CA,),
    ("38.101-1", "B"): (r"NR-DC|dual connectivity",),
    ("38.101-1", "C"): (r"\bSUL\b|supplementary uplink",),
    ("38.101-1", "D"): (MIMO,),
    ("38.101-1", "E"): (r"V2X|sidelink|\bSL\b|\bSL-U\b|vehicle|SLCA",),
    ("38.101-1", "F"): (r"additional",),  # "additional" requirements (NS_xx regions)
    ("38.101-1", "G"): (r"Tx diversity|transmit diversity",),
    ("38.101-1", "H"): (r"(?=.*(?:MIMO))(?=.*(?:\bCA\b|carrier aggregation|UL CA))",),
    ("38.101-1", "I"): (r"RedCap|reduced capability",),
    ("38.101-1", "J"): (r"\bATG\b|air[- ]to[- ]ground",),
    ("38.101-1", "K"): (r"aerial|UAV|drone",),
    ("38.101-1", "L"): (
        r"(?=.*(?:Tx diversity|transmit diversity))(?=.*(?:\bCA\b|UL CA|carrier))",
    ),
    ("38.101-1", "M"): (r"LP-WUS|\bWUR\b|wake[- ]?up",),
}


def extract_refs(text: str) -> tuple[list[str], list[str]]:
    """(clause numbers, table numbers) the text names, in order, without duplicates."""
    clauses = list(dict.fromkeys(m.group(1).upper() for m in CLAUSE_RE.finditer(text)))
    tables = list(dict.fromkeys(m.group(1) for m in TABLE_RE.finditer(text)))
    return clauses, tables


def clause_letters(clause: str) -> tuple[str, ...]:
    """The variant letters of a clause number (5.2E.1F -> E, F); none for a base clause or an
    annex clause (A.3.9.2)."""
    if not clause or clause[0].isalpha():
        return ()
    return tuple(LETTER_RE.findall(clause))


def mentions_topic(spec: str, letter: str, text: str) -> bool:
    patterns = TOPICS.get((spec, letter), ())
    return any(re.search(p, text, re.IGNORECASE | re.DOTALL) for p in patterns)


def is_downranked(spec: str, clause: str, text: str) -> bool:
    """A variant clause whose topic the text does not mention."""
    letters = clause_letters(clause)
    return bool(letters) and not all(mentions_topic(spec, letter, text) for letter in letters)
