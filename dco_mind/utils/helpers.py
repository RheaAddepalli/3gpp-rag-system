import re
import hashlib


def clean_chunk_text(text: str) -> str:
  
    text = re.sub(r'^\s*\d+\s*[-—]+\s*[—-]?\s*', '', text, flags=re.MULTILINE)
  
    text = re.sub(r'\bpage\s+\d+\s*(of\s+\d+)?\b', '', text, flags=re.IGNORECASE)

    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'\n{2,}', '\n---\n', text)
  
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def validate_and_correct_span(answer: str, question: str,
                               context_chunks: list) -> str:
    """
    Post-generation span validation.
    1. Find anchor chunk by scoring identifier presence
    2. Extract local window (line-level) around identifiers
    3. Validate answer against window
    4. Re-extract by stripping question tokens if validation fails
    """
    if not answer or not context_chunks:
        return answer

    answer_norm = re.sub(r'[^a-z0-9 ]', '', answer.lower())
    numbers     = re.findall(r'\b\d+\b', question)

    # ── STRICT GUARD: answer already confirmed in context ─────
    for chunk in context_chunks:
        for line in chunk.split("\n"):
            line_norm = re.sub(r'[^a-z0-9 ]', '', line.lower())
            if answer_norm in line_norm:
                print(f"[Validate] ✅ Answer found in context — skipping correction")
                return answer  # BUG FIX: was return None

    # ── Step 1: extract identifiers ───────────────────────────
    q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
    keywords = [w for w in q_tokens if len(w) > 3]

    # ── Step 2: find anchor chunk ─────────────────────────────
    best_chunk = None
    best_score = 0

    for chunk in context_chunks:
        chunk_lower = chunk.lower()
        score       = 0

        for num in numbers:
            for kw in keywords:
                pattern = (
                    rf'\b{re.escape(kw)}\s*{re.escape(num)}\b'
                    rf'|\b{re.escape(num)}\s*{re.escape(kw)}\b'
                )
                if re.search(pattern, chunk_lower):
                    score += 5
                    break
            else:
                if re.search(rf'\b{re.escape(num)}\b', chunk_lower):
                    score += 1

        for kw in keywords:
            if kw in chunk_lower:
                score += 1

        if score > best_score:
            best_score = score
            best_chunk = chunk

    if not best_chunk or best_score == 0:
        return None 

    # ── Step 3: extract local window ─────────────────────────
    lines  = best_chunk.split("\n")
    window = ""

    for line in lines:
        line_lower = line.lower()
        num_hit = any(re.search(rf'\b{n}\b', line_lower) for n in numbers)
        kw_hit  = any(kw in line_lower for kw in keywords)

        if num_hit and kw_hit:
            window = line.strip()
            break


    if not window:
        sentences = re.split(r'(?<=[.!?\n])', best_chunk)
        for sent in sentences:
            sent_lower = sent.lower()
            if (any(re.search(rf'\b{n}\b', sent_lower) for n in numbers) or
                    any(kw in sent_lower for kw in keywords)):
                window = sent.strip()
                break

    if not window:
        return answer

    # ── Step 4: validate against window ──────────────────────
    window_norm = re.sub(r'[^a-z0-9 ]', '', window.lower())

    if answer_norm in window_norm:
        print(f"[Validate] ✅ Answer grounded in window")
        return answer

    print(f"[Validate] ❌ Answer not in window — rejecting correction")
    return None


# ============================================================
# UTILITIES
# ============================================================

def get_pdf_hash(pdf_path: str) -> str:
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} sec"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins} min {secs:.2f} sec"


def _get_dynamic_stopwords(chunks: list = None) -> set:
    """
    If chunks are provided, derive stopwords from the corpus:
    words appearing in >70% of chunks are document-level noise.
    Falls back to a minimal universal function-word set.
    """
    base = {
        "what", "which", "who", "where", "when", "why", "how",
        "is", "are", "was", "were", "be", "been", "the", "a", "an",
        "of", "in", "on", "at", "to", "for", "and", "or", "but",
        "this", "that", "it", "its", "do", "did", "does", "has",
        "have", "had", "will", "would", "could", "should", "with",
        "from", "by", "if", "not", "tell", "me", "give", "about",
    }
    if not chunks or len(chunks) < 5:
        return base

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        vec   = TfidfVectorizer(max_features=5000)
        X     = vec.fit_transform(chunks)
        words = vec.get_feature_names_out()
        doc_freq = np.diff(X.tocsc().indptr)
        threshold = 0.70 * len(chunks)
        corpus_stopwords = set(words[doc_freq >= threshold])
        return base | corpus_stopwords
    except Exception:
        return base


def clean_artifacts(text: str) -> str:
    if not text:
        return text

    # Priority 1: extract what follows the last "Input:" label
    if "Input:" in text:
        candidate = text.split("Input:")[-1].strip()
        if candidate and not candidate.lower().startswith(
                ("thought:", "action:", "input:")):
            return candidate.strip().strip('"').strip("'")

    # Priority 2: scan line by line for Input: and collect continuation
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("input:"):
            rest = line.split(":", 1)[-1].strip()
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if not next_line:
                    continue
                if next_line.lower().startswith(("thought:", "action:", "input:")):
                    break
                rest += " " + next_line
            return rest.strip().strip('"').strip("'")

    # Priority 3: strip known ReAct prefixes
    clean_lines = []
    skip_prefixes = ("thought:", "action:", "if answer", "if more",
                     "final_answer", "search_more", "not_found")
    for line in lines:
        if line.strip().lower().startswith(skip_prefixes):
            continue
        clean_lines.append(line)

    result = "\n".join(clean_lines).strip()

    # Priority 4: strip inline artifact keywords
    result = re.sub(
        r'(?i)^(final_answer|search_more|not_found)\s*[:\-—]?\s*',
        '', result
    ).strip()

    return result.strip().strip('"').strip("'")


def _is_cop_out_answer(answer: str, question: str) -> bool:
    """
    Detect non-answers dynamically:
    1. Answer echoes the question (too short + substring of question)
    2. Semantic refusal patterns (regex on structure, not vocabulary)
    3. Empty or trivially short
    """
    ans_clean = answer.lower().strip()
    q_clean   = question.lower().strip()

    if not ans_clean or len(ans_clean) < 2:
        return True

    if len(ans_clean) < 20 and ans_clean in q_clean:
        return True

    if ans_clean == q_clean or ans_clean.startswith(q_clean[:20]):
        return True

    ans_tokens = set(re.findall(r'\b\w{2,}\b', ans_clean))
    q_tokens   = set(re.findall(r'\b\w{2,}\b', q_clean))
    if ans_tokens and ans_tokens.issubset(q_tokens):
        return True

    try:
        from dco_mind.models.llm import call_llama
        verdict = call_llama(
            f"Does this answer refuse, say it cannot answer, or fail to provide "
            f"any real information?\nAnswer: {answer}\nReply YES or NO only.",
            num_ctx=256, temperature=0.0
        ).strip().upper()
        if "YES" in verdict:
            return True
    except Exception:
        pass

    return False


def normalize_answer(answer: str) -> str:
    """Minimal normalization — strip whitespace and artifacts."""
    if not answer:
        return answer
    answer = answer.strip()
    answer = re.sub(r'\s{2,}', ' ', answer)
    return answer









