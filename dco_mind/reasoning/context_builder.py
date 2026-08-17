


import re
from dco_mind.utils.helpers import _get_dynamic_stopwords
from numpy import dot
from numpy.linalg import norm

_UNICODE_MAP = {
    '\u2265': '>=',   # ≥
    '\u2264': '<=',   # ≤
    '\u2260': '!=',   # ≠
    '\u2248': '~=',   # ≈
    '\u00b1': '+-',   # ±
    '\u00d7': 'x',    # ×
    '\u00f7': '/',    # ÷
    '\u2192': '->',   # →
    '\u2190': '<-',   # ←
    '\u2014': '-',    # —
    '\u2013': '-',    # –
    '\u2019': "'",    # '
    '\u2018': "'",    # '
    '\u201c': '"',    # "
    '\u201d': '"',    # "
    '\u2022': '-',    # •
    '\u00e9': 'e',    # é
    '\u00e8': 'e',    # è
    '\u00ea': 'e',    # ê
    '\u00e0': 'a',    # à
    '\u00e2': 'a',    # â
    '\u00fc': 'u',    # ü
    '\u00f6': 'o',    # ö
    '\u00e4': 'a',    # ä
    '\u00df': 'ss',   # ß
    '\u00b0': ' deg', # °
    '\u03b1': 'alpha',
    '\u03b2': 'beta',
    '\u03b3': 'gamma',
    '\u03c3': 'sigma',
    '\u03bc': 'mu',
    '\u03c0': 'pi',
}



def normalize_text(text: str) -> str:
    """
    Normalize unicode symbols to ASCII-friendly equivalents.
    Applied to both chunks (in node_chunk) and questions (in node_qa).
    Prevents silent retrieval mismatches when document uses special characters.
    """
    if not text:
        return text
    for char, replacement in _UNICODE_MAP.items():
        text = text.replace(char, replacement)
    # Collapse multiple spaces created by replacements
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text


# ============================================================
# TEXT UTILITIES
# ============================================================

def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def reorder_by_question(question: str, chunks: list) -> list:
    """
    Reorder chunks by relevance to question using keyword overlap.
    Gives bonus when a number from the question appears in the
    first line of a chunk (structural signal for navigational queries).
    """
    if not chunks:
        return chunks

    stopwords    = _get_dynamic_stopwords()
    q_words      = set(w.lower() for w in question.split()
                       if w.lower() not in stopwords and len(w) > 2)
    q_numbers    = set(re.findall(r'\b\d+\b', question))

    def score(chunk: str) -> int:
        chunk_lower  = chunk.lower()
        kw_score     = sum(1 for w in q_words if w in chunk_lower)
        num_score    = sum(3 for n in q_numbers
                          if re.search(rf'\b{re.escape(n)}\b', chunk_lower))

        first_line = chunk.strip().split('\n')[0].lower() if chunk.strip() else ""
        first_line_num_bonus = sum(
            5 for n in q_numbers
            if re.search(rf'\b{re.escape(n)}\b', first_line)
        )

        total          = kw_score + num_score + first_line_num_bonus
        repeated_spans = _count_repeated_spans(chunk)
        return total - repeated_spans

    scored = sorted(chunks, key=score, reverse=True)

    top_score = score(scored[0]) if scored else 0
    if top_score > 0:
        print(f"[Reorder] top chunk score={top_score} | "
              f"numbers={list(q_numbers)} | "
              f"keywords={list(q_words)[:4]}")
    return scored


def _count_repeated_spans(chunk: str, min_len: int = 4) -> int:
    words  = chunk.lower().split()
    seen   = {}
    repeat = 0
    for i in range(len(words) - min_len + 1):
        span = " ".join(words[i:i + min_len])
        if span in seen:
            repeat += 1
            if repeat == 1:
                print(f"[Reorder] Tiebreaker applied — repeated span: '{span}'")
        seen[span] = i
    return repeat



def clean_context_for_llm(chunks: list) -> list:
    """Remove duplicate chunks before sending to LLM."""
    seen    = set()
    cleaned = []
    for chunk in chunks:
        fp = chunk.strip()[:80]
        if fp not in seen:
            seen.add(fp)
            cleaned.append(chunk)
    return cleaned


def classify_from_context(question: str, chunks: list) -> str:
    """
    Determine answer type from retrieved chunks + question structure.
    NAVIGATIONAL and POSITIONAL are handled upstream in node_qa before
    this function is called.

    Returns one of: FULL_SUMMARY | MULTIPART_QA | FACTUAL_QA | VERIFICATION_QA
    """
    q_low = question.strip().lower()
    words = q_low.split()

    # Verification — polar question structure
    if re.match(
        r'^(is|was|were|does|did|can|should|would|could|has|have|had|are|do|will)\b',
        q_low
    ):
        return "VERIFICATION_QA"

    # Summary
    if (
        len(words) > 12 and "?" not in question
    ) or re.match(
        r'^(summarize|summarise|give\s+a|provide\s+a|generate\s+a|'
        r'write\s+a|create\s+a)\b', q_low
    ):
        return "FULL_SUMMARY"

    
    list_chunk_count = 0
    for chunk in chunks:
        lines   = chunk.strip().split("\n")
        numbered = sum(
            1 for line in lines
            if re.match(r'^\s*(\d+[\.\)]|[-•*])\s+\w', line.strip())
        )
        if numbered >= 2:
            list_chunk_count += 1

    stopwords     = _get_dynamic_stopwords()
    content_words = [w for w in words if len(w) > 3 and w not in stopwords]

    chunks_with_hits = sum(
        1 for chunk in chunks
        if sum(1 for w in content_words if w in chunk.lower()) >= 2
    )

    top_chunk_lines = chunks[0].strip().split("\n") if chunks else []
    short_answer_lines = [
        line for line in top_chunk_lines
        if 2 <= len(line.split()) <= 8
        and any(w in line.lower() for w in content_words)
    ]
    has_short_span = len(short_answer_lines) > 0

    if list_chunk_count >= 2:
        return "MULTIPART_QA"



    if chunks_with_hits <= 2 and has_short_span:
        return "FACTUAL_QA"

    return "FACTUAL_QA"





def normalize_answer(answer: str) -> str:
    """Minimal normalization — strip whitespace and double spaces."""
    if not answer:
        return answer
    answer = answer.strip()
    answer = re.sub(r'\s{2,}', ' ', answer)
    return answer























