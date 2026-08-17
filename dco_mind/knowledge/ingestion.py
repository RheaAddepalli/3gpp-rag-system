
import re
import time

from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz
import pytesseract
from PIL import Image


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dco_mind.generation.response_generator import SECTION_SUMMARY_PROMPT, MERGE_PROMPT

from dco_mind.config.settings import (
    SUMMARY_CHUNK_SIZE, RAG_CHUNK_SIZE,
    CHUNK_OVERLAP, MAX_WORKERS, RAPTOR_BATCH_CHARS
)

from dco_mind.models.llm import call_llama, call_llama_streaming
from dco_mind.events.events import emit_event


_summary_cache: dict = {}

# ============================================================
# CORE: TF-IDF EXTRACTIVE SUMMARY
# ============================================================
def extractive_summary(chunk: str, top_n: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    sentences = [s for s in sentences
                 if not re.match(r'^\[\d+\]', s)
                 and not re.match(r'^\d+\.\s+[A-Z]', s)
                 and s.count('[') < 3]
    if not sentences:
        return chunk[:1000]
    if len(sentences) <= top_n:
        return " ".join(sentences)
    try:
        vectorizer   = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentences)
        scores       = np.array(tfidf_matrix.sum(axis=1)).flatten()
        top_indices  = sorted(np.argsort(scores)[-top_n:].tolist())
        return " ".join(sentences[i] for i in top_indices)
    except Exception:
        return " ".join(sentences[:top_n])


# ============================================================
# DUAL CHUNKING
# ============================================================
def merge_short_lines(text: str) -> str:
    lines = text.split("\n")
    merged = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

      
        if (
            0 < len(line) < 80
            and not line.endswith(".")
            and i + 1 < len(lines)
        ):
            next_line = lines[i + 1].strip()

            if next_line and len(next_line) > 20:
                merged.append(f"{line} — {next_line}")
                i += 2
                continue

        merged.append(line)
        i += 1

    return "\n".join(merged)
def clause_aware_chunk(text: str):
    """
    Generic clause-aware chunker for 3GPP technical specifications.

    Strategy:
      1. Identify the document TOC.
      2. Extract valid clause numbers from the TOC.
      3. Locate the actual document body.
      4. Locate those known clause numbers in the body — verified
         against the TOC title (word-overlap), not just structurally.
      5. Keep each clause together whenever possible.
      6. Split only very large clauses internally.
    """

    generic_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


    def _title_words(s: str) -> set:
        return set(w for w in re.findall(r'[a-z]{3,}', s.lower()))

    def _title_matches(candidate_text: str, expected_title: str) -> bool:
        expected_words = _title_words(expected_title)
        if not expected_words:
            return True  
        candidate_words = _title_words(candidate_text[:180])
        overlap = expected_words & candidate_words
        required = 1 if len(expected_words) <= 2 else 2
        return len(overlap) >= required



    def _strip_running_header_noise(clause_text: str, number: str, title: str) -> str:
        """
        Strips a running page-header repeated once per page
        (e.g. '3 Abbreviations and Definitions ..........'
        appearing at the top of every page in that section),
        plus any bare dot-leader runs left anywhere in the text.
        Runs GLOBALLY across the whole clause, not just at the
        start — a multi-page clause carries one copy of this
        per page.
        """
        text = clause_text

        escaped_number = re.escape(number)
        escaped_title = re.escape(title)
        running_header = re.compile(
            r'\b' + escaped_number + r'\s+' + escaped_title,
            re.IGNORECASE
        )
        text = running_header.sub(' ', text)

        text = re.sub(r'(?:\.\s?){4,}', ' ', text)

        text = re.sub(
            r'ETSI\s+ETSI\s+TS\s+\d{3}\s+\d{3}\s+V\d+\.\d+\.\d+\s+'
            r'\(\d{4}-\d{2}\)\s+\d+\s+3GPP\s+TS\s+\d+\.\d+\s+'
            r'version\s+\d+\.\d+\.\d+\s+Release\s+\d+',
            ' ',
            text
        )

        # Collapse whitespace created by the removals above.
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # ========================================================
    # 1. Split document into extracted pages
    # ========================================================

    page_blocks = re.split(r'--- PAGE (\d+) ---', text)

    pages = []

    i = 1
    while i < len(page_blocks) - 1:
        try:
            page_number = int(page_blocks[i])
            page_content = page_blocks[i + 1].strip()

            pages.append({
                "number": page_number,
                "text": page_content
            })
        except Exception:
            pass

        i += 2

    if len(pages) < 10:
        print(
            "[Clause Chunk] Page structure not found "
            "— generic fallback"
        )
        return generic_splitter.split_text(text)

    # ========================================================
    # 2. Find actual document body
    # ========================================================

    body_start_index = None

    for idx, page in enumerate(pages):

        page_text = page["text"]

        if re.search(
            r'\b1\s+Scope\s+The\s+present\s+document\b',
            page_text,
            re.IGNORECASE
        ):
            body_start_index = idx
            break

    if body_start_index is None:
        print(
            "[Clause Chunk] Could not locate actual "
            "document body — generic fallback"
        )
        return generic_splitter.split_text(text)

    body_start_page = pages[body_start_index]["number"]

    print(
        f"[Clause Chunk] Document body starts at page "
        f"{body_start_page}"
    )

    # ========================================================
    # 3. Extract TOC text
    # ========================================================

    toc_text = "\n".join(
        page["text"]
        for page in pages[:body_start_index]
    )

    # ========================================================
    # 4. Extract valid clause numbers from TOC
    # ========================================================

    toc_clause_pattern = re.compile(
        r'(?<![\w.])'
        r'(\d{1,2}(?:\.\d+){0,6})'
        r'\s+'
        r'([A-Z][A-Za-z0-9][^\n]{2,150}?)'
        r'\s+'
        r'(\d{1,4})'
        r'(?=\s|$)'
    )

    toc_clauses = []
    seen_toc_numbers = set()

    for match in toc_clause_pattern.finditer(toc_text):

        number = match.group(1)
        title = match.group(2).strip()
        page_number = match.group(3)

        title = re.sub(r'(?:\.\s?){3,}.*$', '', title).strip()

        if number in {
            "38.300",
            "38.331",
            "15.9.0",
            "19.0.0"
        }:
            continue

        if number in seen_toc_numbers:
            continue

        if len(title) > 150:
            continue

        seen_toc_numbers.add(number)

        toc_clauses.append({
            "number": number,
            "title": title,
            "page": int(page_number)
        })

    print(
        f"[Clause Chunk] Detected "
        f"{len(toc_clauses)} clause numbers from TOC"
    )

    # ========================================================
    # 5. Safety fallback
    # ========================================================

    if len(toc_clauses) < 10:

        print(
            "[Clause Chunk] Insufficient TOC structure "
            "— generic fallback"
        )

        body_text = "\n\n".join(
            f"--- PAGE {p['number']} ---\n{p['text']}"
            for p in pages[body_start_index:]
        )

        return generic_splitter.split_text(body_text)

    # ========================================================
    # 6. Build body text
    # ========================================================

    body_pages = pages[body_start_index:]

    body_text = "\n\n".join(
        f"--- PAGE {p['number']} ---\n{p['text']}"
        for p in body_pages
    )



    body_matches = []

    search_position = 0

    for clause in toc_clauses:

        number = clause["number"]
        expected_title = clause["title"]

        escaped_number = re.escape(number)

        pattern = re.compile(
            r'(?<![\w.])'
            + escaped_number +
            r'\s+'
            r'(?=[A-Za-z])',
            re.MULTILINE
        )

        found = None

        for match in pattern.finditer(
            body_text,
            search_position
        ):

            start = match.start()

            before = body_text[
                max(0, start - 100):start
            ]

            if re.search(
                r'(?:clause|sub-clause|section)\s*$',
                before,
                re.IGNORECASE
            ):
                continue

            after = body_text[
                match.end():
                match.end() + 180
            ]

            title_match = re.match(
                r'([A-Za-z][A-Za-z0-9][^\n]{2,150})',
                after
            )

            if not title_match:
                continue

            candidate_title_text = title_match.group(1).strip()

            # ---- THE FIX: verify against TOC title ----
            if not _title_matches(candidate_title_text, expected_title):
                continue

            found = match
            break

        if found is not None:

            body_matches.append({
                "number": number,
                "title": clause["title"],
                "start": found.start()
            })

            search_position = found.end()

    print(
        f"[Clause Chunk] Located "
        f"{len(body_matches)} clause headers in body"
    )

    # ========================================================
    # 8. Safety check
    # ========================================================

    if len(body_matches) < 10:

        print(
            "[Clause Chunk] Too few body clauses "
            "— generic fallback"
        )

        return generic_splitter.split_text(body_text)

    # ========================================================
    # 9. Remove duplicate clause positions
    # ========================================================

    unique_matches = []
    seen_positions = set()

    for match in body_matches:

        key = (
            match["number"],
            match["start"]
        )

        if key in seen_positions:
            continue

        seen_positions.add(key)
        unique_matches.append(match)

    body_matches = unique_matches



    if body_matches:
        last_start = body_matches[-1]["start"]
        annex_match = re.search(r'\bAnnex\s+[A-Z]\b', body_text[last_start:])
        if annex_match:
            annex_boundary = last_start + annex_match.start()
            print(
                f"[Clause Chunk] Trimming trailing Annex section "
                f"(was {len(body_text)} chars, now {annex_boundary})"
            )
            body_text = body_text[:annex_boundary]

    # ========================================================
    # 10. Build clause-aware chunks
    # ========================================================

    rag_chunks = []

    for idx, clause in enumerate(body_matches):

        start = clause["start"]

        if idx + 1 < len(body_matches):
            end = body_matches[idx + 1]["start"]
        else:
            end = len(body_text)

        clause_text = body_text[
            start:end
        ].strip()

        if not clause_text:
            continue

        page_numbers = [
            int(x)
            for x in re.findall(
                r'--- PAGE (\d+) ---',
                clause_text
            )
        ]

        if page_numbers:

            page_start = page_numbers[0]
            page_end = page_numbers[-1]

            if page_start == page_end:
                page_info = f" | Page {page_start}"
            else:
                page_info = (
                    f" | Pages "
                    f"{page_start}-{page_end}"
                )

        else:
            page_info = ""

        clean_clause = re.sub(
            r'--- PAGE \d+ ---',
            '',
            clause_text
        )

        clean_clause = re.sub(
            r'\s+',
            ' ',
            clean_clause
        ).strip()

    
        clean_clause = _strip_running_header_noise(
            clean_clause,
            clause["number"],
            clause["title"]
        )

        if not clean_clause:
            continue

        # Use the REAL TOC title in the header now, not a
        # placeholder — this also makes citations meaningful.
        header = (
            f"[3GPP Clause {clause['number']}{page_info}]\n"
            f"{clause['number']} {clause['title']}"
        )

        if len(clean_clause) <= RAG_CHUNK_SIZE:

            rag_chunks.append(
                f"{header}\n{clean_clause}"
            )

            continue

        pieces = generic_splitter.split_text(
            clean_clause
        )

        for piece in pieces:

            if not piece.strip():
                continue

            rag_chunks.append(
                f"{header}\n{piece}"
            )

    print(
        f"[Clause Chunk] "
        f"{len(body_matches)} clauses → "
        f"{len(rag_chunks)} RAG chunks"
    )

    return rag_chunks
def semantic_chunk(text: str):
   
    clean_text = re.sub(r'--- PAGE \d+ ---\n?', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    summary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=SUMMARY_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    summary_chunks = summary_splitter.split_text(clean_text)

    # RAG chunks — clause-aware
    rag_chunks = clause_aware_chunk(text)

    print(
        f"[Chunk] Summary chunks: {len(summary_chunks)} "
        f"| RAG chunks: {len(rag_chunks)}"
    )

    return summary_chunks, rag_chunks
# ============================================================
# CORE: RAPTOR HIERARCHICAL SUMMARY
# ============================================================
def raptor_summarize(chunks: list, doc_type: str):
    if not chunks:
        return "No content to summarize.", 0, 0

    map_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures        = [ex.submit(extractive_summary, chunk, 3) for chunk in chunks]
        mini_summaries = [f.result() for f in futures if f.result().strip()]
    map_time = time.time() - map_start
    print(f"[RAPTOR] Map (TF-IDF): {len(mini_summaries)} chunks in {map_time:.1f}s")

    if not mini_summaries:
        return "Could not generate summary.", map_time, 0

    reduce_start  = time.time()
    BATCH_CHARS = RAPTOR_BATCH_CHARS
    batches       = []
    current_batch = ""
    for summary in mini_summaries:
        if len(current_batch) + len(summary) > BATCH_CHARS and current_batch:
            batches.append(current_batch.strip())
            current_batch = summary + "\n\n"
        else:
            current_batch += summary + "\n\n"
    if current_batch.strip():
        batches.append(current_batch.strip())

    request_id = getattr(raptor_summarize, '_request_id', "")

    partial_summaries = []
    print(f"[RAPTOR] Reduce: {len(batches)} batches → LLaMA")

    if len(batches) == 1:
        emit_event(request_id, "stream_start", "✍️ Generating summary...")
        final_summary, _ = call_llama_streaming(
            SECTION_SUMMARY_PROMPT.format(text=batches[0]),
            request_id, temperature=0.7)
        partial_summaries = [final_summary]
    else:
        for i, batch in enumerate(batches):
            prompt = SECTION_SUMMARY_PROMPT.format(text=batch)
            result = call_llama(prompt, temperature=0.7)
            partial_summaries.append(result)
            print(f"[RAPTOR] Batch {i+1}/{len(batches)} done")
            emit_event(request_id, "agent_action",
                       f"📝 Processed section {i+1}/{len(batches)}...")

        merged = "\n\n---\n\n".join(partial_summaries)
        emit_event(request_id, "stream_start", "✍️ Generating final summary...")
        final_summary, _ = call_llama_streaming(
            MERGE_PROMPT.format(summaries=merged[:12000]),
            request_id, temperature=0.7)

    reduce_time = time.time() - reduce_start
    total_calls = len(batches) + (1 if len(partial_summaries) > 1 else 0)
    print(f"[RAPTOR] Reduce done: {reduce_time:.1f}s | {total_calls} LLaMA calls | Total: {map_time+reduce_time:.1f}s")
    return final_summary, map_time, reduce_time









# extraction


from dco_mind.config.settings import MAX_WORKERS
from dco_mind.utils.helpers import get_pdf_hash
_extraction_cache: dict = {}

# ============================================================
# CORE: EXTRACT SINGLE PAGE
# ============================================================
def extract_page(args):
    page, page_num = args
    try:
        text = page.get_text().strip()
        if len(text) > 10:
            return page_num, text

        if page.rect.width < 10 or page.rect.height < 10:
            return page_num, ""

        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        try:
            t = pytesseract.image_to_string(img).strip()
            real_words = len([w for w in t.split() if len(w) > 2 and w.isalpha()])
            if real_words > 5:
                return page_num, t
        except Exception:
            pass

        best_text = ""
        best_len  = 0
        for angle in [90, 180, 270]:
            rotated = img.rotate(angle, expand=True)
            try:
                t = pytesseract.image_to_string(rotated).strip()
                real_words = len([w for w in t.split() if len(w) > 2 and w.isalpha()])
                if real_words > best_len:
                    best_len  = real_words
                    best_text = t
            except Exception:
                continue

        return page_num, best_text.strip()
    except Exception:
        return page_num, ""


def extract_pdf_parallel(pdf_path: str):
    pdf_hash = get_pdf_hash(pdf_path)
    if pdf_hash in _extraction_cache:
        text, page_count = _extraction_cache[pdf_hash]
        print(f"[Extract] ✅ Cache hit — skipping OCR ({page_count} pages)")
        return text, page_count

    doc        = fitz.open(pdf_path)
    page_count = len(doc)
    pages      = [(doc[i], i) for i in range(page_count)]
    results    = {}
    workers    = min(MAX_WORKERS, page_count)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(extract_page, p): p[1] for p in pages}
        for future in as_completed(futures):
            page_num, text    = future.result()
            results[page_num] = text
    doc.close()

    page_texts = []
    for i in sorted(results.keys()):
        t = results[i].strip()
        if t:
            t = re.sub(r'[^\x00-\x7F]+', ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
            page_texts.append(f"--- PAGE {i} ---\n{t}")

    full_text = "\n\n".join(page_texts)
    _extraction_cache[pdf_hash] = (full_text, page_count)
    print(f"[Extract] Done — {page_count} pages, {len(full_text)} chars")
    return full_text, page_count







