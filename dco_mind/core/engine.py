

import time
import re

from dco_mind.core.state import DocState
from dco_mind.generation.response_generator import QA_PROMPT

from dco_mind.models.llm import call_llama_streaming
from dco_mind.models.embeddings import build_faiss_index
from dco_mind.retrieval.reranker import rerank_docs, protect_exact_matches

from dco_mind.knowledge.ingestion import (
    extract_pdf_parallel, _extraction_cache,
    semantic_chunk, raptor_summarize, _summary_cache
)

from dco_mind.retrieval.adaptive_search import multi_query_retrieve

from dco_mind.cognition.query_brain import (
    react_agent, clean_reasoning_answer
)

from dco_mind.events.events import emit_event

from dco_mind.utils.helpers import (
    get_pdf_hash, clean_artifacts, clean_chunk_text, normalize_answer
)

from dco_mind.reasoning.context_builder import (
    classify_from_context,
    reorder_by_question,
    normalize_text,
)

from dco_mind.evaluation.metrics import (
    compute_answer_grounding,
    compute_retrieval_score,
    compute_context_precision,
    compute_recall_at_k,
    compute_confidence,
    compute_grounding_score,
    semantic_similarity,
)
# ============================================================
# LOCAL HELPERS
# ============================================================

def clean_context_for_llm(chunks):
    cleaned_chunks = []
    for chunk in chunks:
        lines = chunk.split("\n")
        filtered_lines = []
        for line in lines:
            line_strip = line.strip()
            is_question_like = (
                line_strip.endswith("?") or
                (len(line_strip.split()) < 15 and "?" in line_strip)
            )
            if is_question_like:
                continue
            filtered_lines.append(line)
        cleaned_chunks.append("\n".join(filtered_lines))
    return cleaned_chunks




# ============================================================
# NUMERIC INTENT DETECTION
# ============================================================

def _extract_numbers(text: str) -> list:
    return re.findall(r'\b\d+\b', text)


def _detect_numeric_intent(question: str, chunks: list) -> str:
    """
    Detect NAVIGATIONAL or POSITIONAL from chunk structure.
    Generic — no hardcoded section words.
    """
    numbers = _extract_numbers(question)
    if not numbers:
        return "NONE"

    q_words = set(question.lower().split())

    for chunk in chunks:
        lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        if not lines:
            continue
   
        first_line = " ".join(lines[0].split()[:10]).lower()
        line_words = set(first_line.split())
        overlap    = len(q_words & line_words)
        num_found  = any(re.search(rf'\b{re.escape(num)}\b', first_line) for num in numbers)

        if num_found:
            print(f"[NavDebug] first_line='{first_line}' | overlap={overlap} | words={len(first_line.split())}")

        if num_found and overlap >= 2: 
            return "NAVIGATIONAL"
    for chunk in chunks:
        lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
        short_lines = [l for l in lines if len(l.split()) <= 12]
        if len(short_lines) >= 3:
            first = lines[0].lower() if lines else ""
            if not any(re.search(rf'\b{re.escape(num)}\b', first) for num in numbers):
                return "POSITIONAL"

    return "NONE"


def _navigate_full_chunks(question: str, all_raw_chunks: list) -> str:
    numbers = _extract_numbers(question)
    if not numbers:
        return ""

    q_words    = set(question.lower().split())
    candidates = []

    for chunk in all_raw_chunks:
        lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        if not lines:
            continue

        first_line  = " ".join(lines[0].split()[:10])
        first_lower = first_line.lower()

        for num in numbers:
            if not re.search(rf'\b{re.escape(num)}\b', first_lower):
                continue

            line_words = set(first_lower.split())
            overlap    = len(q_words & line_words)

            if overlap >= 2:
                # generic scoring (NO hardcoding)
                has_number = num in first_lower
                score = overlap + (2 if has_number else 0)

                candidates.append((first_line, score))

    if not candidates:
        return ""

   
    candidates.sort(key=lambda x: x[1], reverse=True)

    best_line, best_score = candidates[0]

    if len(candidates) > 1:
        second_score = candidates[1][1]
    else:
        second_score = 0

    if best_score <= second_score:
        print("[NAV] ❌ No clear winner → fallback")
        return ""

    if best_score < 3:
        print("[NAV] ⚠️ Weak match → fallback")
        return ""
    return best_line





# ============================================================
# REFUSAL — semantic version (local)
# ============================================================

_REFUSAL_ANCHOR = "this information is not available in the provided context"


def _is_refusal_semantic(text: str) -> bool:
    if not text or len(text.strip()) < 2:
        return True
    sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
    print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
    return sim > 0.72

# ============================================================
# CITATION GUARD
# ============================================================

def _extract_cited_clauses(text: str) -> list:
    """
    Extract citations in the format:
        Source: TS 38.331, Clause 4.2.1
    """
    if not text:
        return []

    pattern = r"Source:\s*(TS\s+\d+\.\d+)\s*,\s*Clause\s+([\d.]+)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    return [
        (spec.strip(), clause.strip())
        for spec, clause in matches
    ]


def _citation_exists_in_context(answer: str, context_chunks: list) -> bool:
    """
    Verify that every cited source/clause in the answer
    actually exists in the retrieved context.
    """
    citations = _extract_cited_clauses(answer)

    if not citations:
        return False

    context = "\n".join(context_chunks).lower()

    for spec, clause in citations:
        spec_norm = re.sub(r"\s+", " ", spec.lower()).strip()

        clause_pattern = rf"\[3gpp clause\s+{re.escape(clause.lower())}(?:\s*\|[^\]]*)?\]"

        if spec_norm not in context:
            return False

        if not re.search(clause_pattern, context):
            return False

    return True
# ============================================================
# LANGGRAPH NODES
# ============================================================

def node_extract(state: DocState) -> DocState:
    extract_start    = time.time()
    text, page_count = extract_pdf_parallel(state["pdf_path"])
    extract_time     = time.time() - extract_start
    state["extracted_text"] = text
    state["page_count"]     = page_count
    state["char_count"]     = len(text)
    state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
    state["metrics"]["pages_processed"]      = page_count
    state["metrics"]["characters_processed"] = len(text)
    state["metrics"]["words_processed"]      = len(text.split())
    return state


def node_chunk(state: DocState) -> DocState:
    text = state["extracted_text"]
    state["query_type"] = "QA"
    question = state.get("question", "").strip().lower()
    summary_pattern = (
    r'^(summarize|summarise|summary|'
    r'give me a summary|'
    r'provide a summary|'
    r'summarise this|'
    r'generate a summary)\b'
)
    if re.match(summary_pattern, question) or (len(question.split()) > 12 and "?" not in question):
        state["query_type"] = "FULL_SUMMARY"

    summary_chunks, rag_chunks = semantic_chunk(text)

    # FIX 4 — normalize all chunks after chunking
    # Converts unicode symbols so retrieval doesn't fail on symbol mismatches
    rag_chunks     = [normalize_text(c) for c in rag_chunks]
    summary_chunks = [normalize_text(c) for c in summary_chunks]

    state["summary_chunks"] = summary_chunks
    state["chunks"]         = rag_chunks

    state["metrics"]["summary_chunks"] = len(summary_chunks)
    state["metrics"]["chunks_created"] = len(rag_chunks)
    # state["metrics"]["doc_type"]       = state.get("doc_type", "general")
    state["metrics"]["query_type"]     = state["query_type"]
    print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
    return state




def node_summarize(state: DocState) -> DocState:
    pdf_hash  = get_pdf_hash(state["pdf_path"])
    cache_key = pdf_hash  # ❌ removed doc_type dependency

    # ── Cache check ──────────────────────────────────────────
    if cache_key in _summary_cache:
        cached = _summary_cache[cache_key]
        print("[Summary] ✅ Cache hit")
        emit_event(
            state.get("request_id", ""),
            "agent_action",
            "⚡ Summary loaded from cache instantly!"
        )
        state["answer"] = cached["summary"]
        state["metrics"].update(cached["metrics"])
        state["metrics"]["type"] = "summary"
        return state

    # ── Generate summary ─────────────────────────────────────
    summary_start = time.time()
    raptor_summarize._request_id = state.get("request_id", "")

    summary, map_time, reduce_time = raptor_summarize(
        state["summary_chunks"],  
        state.get("doc_type", "general")
    )
    summary_time = time.time() - summary_start

    # ── Store result ─────────────────────────────────────────
    state["answer"] = summary

    metrics_snapshot = {
        "summary_time_sec":     round(summary_time, 2),
        "summary_length_words": len(summary.split()),
        "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
        "map_time_sec":         round(map_time, 2),
        "reduce_time_sec":      round(reduce_time, 2),
        "llm_calls":            3,
    }

    state["metrics"].update(metrics_snapshot)
    state["metrics"]["type"] = "summary"

    # ── Cache result ─────────────────────────────────────────
    _summary_cache[cache_key] = {
        "summary": summary,
        "metrics": metrics_snapshot
    }

    print(f"[Summary] Done ({len(summary.split())} words)")
    return state

def node_qa(state: DocState) -> DocState:
    print("\n[DEBUG STATE]")
    print(f"original_question = {repr(state.get('original_question'))}")
    print(f"session_id = {repr(state.get('session_id'))}")
    qa_start_t = time.time()
    question   = state["question"]
    request_id = state.get("request_id", "")
    all_chunks = state["chunks"]
    session_id = state.get("session_id", "")
    if not session_id:
        session_id = "default_session"
    original_q = state.get("original_question", question)

    question = normalize_text(question)
    numeric_intent = "NONE"

    all_raw = [
        d if isinstance(d, str) else d.page_content
        for d in all_chunks
    ]

    if state.get("faiss_index") is not None:
        faiss_index = state["faiss_index"]
        print("[QA] ✅ Using precomputed FAISS index (fixed corpus mode)")
    else:
        pdf_hash    = get_pdf_hash(state["pdf_path"])
        faiss_index = build_faiss_index(all_chunks, pdf_hash)

    print(f"[QA] START | {len(all_chunks)} chunks")

    recall_score      = 0.0
    retrieval_score   = 0.0
    context_precision = 0.0
    grounding         = 0.0
    grounding_score = 0.0
    semantic_ground = 0.0
    grounding_now = 0.0
    force_not_found = False
    react_ans = ""
    llm_calls         = 0
    model_used        = "llama"
    confidence        = 0.0
    decision_type     = "accepted"
    retrieved         = []
    retrieved_texts   = []
    answer            = ""
    rewrite_triggered = False
    rewritten_question = ""
    pre_rewrite_docs = []
    post_rewrite_docs = []
    refusal_phrases = [
    "not present",
    "not mentioned",
    "not available",
    "no information",
    "cannot find",
    "context does not mention",
    "don't have any information",
    "not in the document",
    "cannot answer",
    "no mention",
    "there is no mention",
    "does not mention",
    "cannot determine",
    "don't see any information",
    "don't see any mention",
    "no mention of",
    "i don't see",
    "there is no",
    "i don't have",
    "does not explicitly mention",
    "does not specifically",
    "not explicitly mentioned",
]


    # ── STEP 1: RETRIEVE — clean question only ────────────────
    retrieved = multi_query_retrieve(
        question, faiss_index,
        k=50,
        all_chunks=all_chunks,
        query_type="FACTUAL_QA"
    )
    initial_retrieved = retrieved.copy()
    pre_rewrite_docs = [
    {
        "content": d.page_content if hasattr(d, "page_content") else str(d),
        "metadata": getattr(d, "metadata", {})
    }
    for d in initial_retrieved
]

    # 🔥 STEP 2 — RERANK + SAFE FALLBACK
    retrieved, reranker_top, _ = rerank_docs(
        question, retrieved, top_k=8, apply_pruning=True
    )

    retrieved = protect_exact_matches(
        question, retrieved, all_chunks, top_k=8
    )

    if not retrieved or len(retrieved) < 3:
        print("[QA] ⚠️ Reranker too aggressive → fallback to initial chunks")
        retrieved = initial_retrieved[:8]

    retrieved_texts = [
        d.page_content if hasattr(d, "page_content") else str(d)
        for d in retrieved
    ]
    if reranker_top < -5:
        print("[Guard] ⚠️ Weak reranker score — continuing")

    # ── STEP 3: METRICS ───────────────────────────────────────
    retrieval_score   = compute_retrieval_score(question, retrieved)
    context_precision = compute_context_precision(question, retrieved)
    recall_score      = compute_recall_at_k(
        question, retrieved, all_chunks, k=len(retrieved)
    )

    is_numeric_question = len(_extract_numbers(question)) > 0
    if recall_score < 40:
        print("[QA] ⚠️ Expanding retrieval (low recall)")
        expanded = multi_query_retrieve(
            question, faiss_index, k=30,
            all_chunks=all_chunks, query_type="FACTUAL_QA"
        )
        if len(expanded) > len(retrieved):
            retrieved = expanded
            retrieved_texts = [
                d.page_content if hasattr(d, "page_content") else str(d)
                for d in retrieved
            ]
        numeric_intent = _detect_numeric_intent(question, retrieved_texts)
        is_numeric_question = len(_extract_numbers(question)) > 0
        if numeric_intent == "NONE" and is_numeric_question:
            print("[Navigate] Retrying intent detection on full document...")
            numeric_intent = _detect_numeric_intent(question, all_raw)
    else:
        print("[QA] ✅ High recall — trusting retrieved context")

    if numeric_intent == "NAVIGATIONAL":
        print("[Routing] Numeric intent → NAVIGATIONAL")
        title = _navigate_full_chunks(question, all_raw)
        if title:
            print(f"[Navigate] Extracted title: '{title}'")
            grounding  = compute_answer_grounding(title, retrieved_texts, question)
            try:
                confidence = compute_confidence(
                    reranker_top=reranker_top,
                    recall_score=recall_score,
                    answer=answer,
                    context_chunks=retrieved_texts,
                    question=question
                )
            except Exception as e:
                print("\n[CONFIDENCE CRASH]")
                print(f"answer={repr(answer)}")
                print(f"question={repr(question)}")
                print(f"retrieved_texts type={type(retrieved_texts)}")
                print(f"retrieved_texts len={len(retrieved_texts) if retrieved_texts else 0}")
                print(f"ERROR={e}")
                raise
            qa_time    = time.time() - qa_start_t
            state["answer"]     = title
            state["query_type"] = "NAVIGATIONAL"
            _write_metrics(state, "navigational", "navigational",
                           grounding, confidence, retrieval_score,
                           context_precision, recall_score, llm_calls,
                           retrieved, qa_time)
            return state
        print("[Navigate] Falling back to QA")

    # ── Normal routing ────────────────────────────────────────
    query_type = classify_from_context(original_q, retrieved_texts)
    state["query_type"] = query_type
    print(f"[Routing] Context-based → {query_type}")

    # ── STEP 4: ANSWER GENERATION ─────────────────────────────
    if query_type == "FULL_SUMMARY":
        llm_context_chunks = clean_context_for_llm(retrieved_texts)
        
        ranked = reorder_by_question(
            question,
            llm_context_chunks
        )
        context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
        structured_context = "\n".join(
            f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
        )
        emit_event(request_id, "stream_start", "✍️ Generating answer...")
        answer, _ = call_llama_streaming(
            QA_PROMPT.format(
                context=structured_context[:2500],
                question=question
            ),
            request_id=request_id, temperature=0.0
        )
        if answer is None:                         
            answer = ""                             
        answer     = clean_artifacts(str(answer)).strip()   
        model_used = "llama_summary"
        llm_calls  = 1
        grounding_score = compute_answer_grounding(answer, retrieved_texts, question)
        semantic_ground = semantic_similarity(answer, retrieved_texts)

    elif query_type == "MULTIPART_QA":
        llm_context_chunks = clean_context_for_llm(retrieved_texts)

        ranked = reorder_by_question(
            question,
            llm_context_chunks
        )
        context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
        emit_event(request_id, "agent_start", f"🤖 MULTIPART | {len(retrieved)} chunks")
        emit_event(request_id, "stream_start", "✍️ Generating answer...")
        react_ans, _ = call_llama_streaming(
            QA_PROMPT.format(
                context=context[:4000],
                question=question
            ),
            request_id=request_id, temperature=0.0
        )
        if react_ans is None:                       
            react_ans = ""                          
        react_ans = clean_artifacts(str(react_ans)).strip()  
        answer = react_ans

        model_used = "llama_multipart"
        llm_calls  = 1
        grounding_score = compute_answer_grounding(answer, retrieved_texts, question)
        semantic_ground = semantic_similarity(answer, retrieved_texts)

    else:
        if not retrieved_texts:
            print("[QA] ❌ No context → NOT FOUND")
            state["answer"] = "This information is not present in the document."
            _write_metrics(state, "not_found", "no_context",
                        0.0, 0.0, retrieval_score, context_precision,
                        recall_score, llm_calls, retrieved,
                        time.time() - qa_start_t)
            return state 

        cleaned_texts = clean_context_for_llm(retrieved_texts)
        context = "\n".join(cleaned_texts)
   

        if (
            query_type in ("FACTUAL_QA", "VERIFICATION_QA")
            and recall_score >= 60
        ):
            print("[QA] ⚡ Direct QA (no ReAct)")
        

            if query_type == "FACTUAL_QA":

                prompt = f"""Answer the question strictly using the provided context.

Rules:
1. Use only information supported by the context.
2. For conceptual questions, give a clear and concise explanation.
3. For factual fields such as names, numbers, dates, or values, give the exact information from the context.
4. Do not hallucinate or use outside knowledge.
5. If the context does not contain enough information to answer the question, output ONLY:
"This information is not present in the document."
6. When answering, cite the relevant 3GPP source at the end of the answer.
7. Use the source information already present in the context, such as:
   [Source: TS 38.331]
   [3GPP Clause 4.2.1 | Pages 21-22]
8. Format citations like:
   Source: TS 38.331, Clause 4.2.1
9. If multiple clauses/specifications support the answer, cite each relevant source.
10. Do not invent clause numbers or sources that are not present in the context.

Context:
{context[:3500]}

Question:
{original_q}

Answer:
"""
            else:
                prompt = f"""You are answering a question using ONLY the retrieved 3GPP context below.

Context:
{context[:3500]}

Question:
{original_q}

Rules:
1. Use the retrieved context as the authoritative source.
2. Read ALL provided context carefully before deciding that information is absent.
3. Answer the user's question directly if the answer is explicitly stated in the context.
4. For Yes/No or verification questions, answer Yes or No when the answer can be directly determined from facts explicitly stated in the context.
5. The wording of the question does NOT need to appear verbatim in the context.
6. You may connect facts that are explicitly stated in the context to determine the answer.
7. Do NOT use outside knowledge.
8. Do NOT invent facts, entities, relationships, clause numbers, or sources.
9. Do NOT say "information is not present" merely because the exact answer sentence is not present.
10. Only say:
"This information is not present in the document."
when the retrieved context genuinely does not contain enough evidence to answer the question.
11. For Yes/No questions, start with "Yes" or "No", then briefly explain using the retrieved evidence.
12. For factual questions, provide the specific fact requested rather than a related fact from the same topic.
13. Cite the relevant 3GPP source at the end of the answer.
14. Use source information already present in the context, such as:
   [Source: TS 38.331]
   [3GPP Clause 4.2.1 | Pages 21-22]
15. Never invent a citation.

Answer:
"""

            react_ans, _ = call_llama_streaming(
                prompt,
                request_id=request_id,
                temperature=0.0
            )
            llm_calls += 1
            model_used = "llama_direct"
         
            FOLLOWUP_WORDS = {
    "it", "this", "that",
    "he", "she", "they",
    "them", "his", "her", "their"
}

            is_followup = (
                len(question.split()) <= 8 and
                any(w in question.lower().split() for w in FOLLOWUP_WORDS)
            )

            is_refusal = any(
                p in react_ans.lower()
                for p in refusal_phrases
            )

        else:
            print(f"[QA] 🤖 Recall={recall_score:.1f}% or type={query_type} "
                  f"→ falling back to ReAct agent")
            react_ans, model_used, _, _, _, _, react_calls, _ = react_agent(
                question,
                faiss_index,
                query_type,
                all_chunks,
                request_id,
                recall_score
            )
            llm_calls += react_calls

        print("\n[DEBUG RAW LLM OUTPUT]")
        print(f"type={type(react_ans)}")
        print(f"value={repr(react_ans)}")
        if react_ans is None:
            print("[ERROR] react_ans is None")
            react_ans = ""

        react_ans = clean_artifacts(
            str(react_ans)
        ).strip().strip('"').strip("'")
        print(f"[QA] Answer: '{react_ans[:60]}'")

        words         = react_ans.split()
        content_words = [w for w in words if len(w) > 2 and not w.isdigit()]

        print(f"[DEBUG] react_ans before refusal check: repr={repr(react_ans)}")


      

        # ============================================================
        # HARD HALLUCINATION / CITATION GUARD
        # ============================================================

        is_refusal_detected = any(
            p in react_ans.lower()
            for p in refusal_phrases
        )

        if query_type == "FACTUAL_QA":

            # Case 1: LLM explicitly refused
            if is_refusal_detected:
                print("[Citation Guard] ⚠️ Refusal detected → abstaining")

                force_not_found = True
                answer = "This information is not present in the provided 3GPP specifications."
                react_ans = answer
                decision_type = "not_found"

            # Case 2: LLM answered → citation is mandatory
            else:
                citations = _extract_cited_clauses(react_ans)

                if not citations:
                    print("[Citation Guard] ❌ No valid clause citation → abstaining")

                    force_not_found = True
                    answer = "This information is not present in the provided 3GPP specifications."
                    decision_type = "citation_missing"

                elif not _citation_exists_in_context(
                    react_ans,
                    retrieved_texts
                ):
                    print("[Citation Guard] ❌ Citation not grounded in retrieved context → abstaining")

                    force_not_found = True
                    answer = "This information is not present in the provided 3GPP specifications."
                    decision_type = "citation_invalid"

                else:
                    print(
                        f"[Citation Guard] ✅ Citation present and grounded: "
                        f"{citations}"
                    )

        grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
        semantic_ground = semantic_similarity(
    react_ans,
    retrieved_texts
)
        print(
    f"[Grounding Fusion] "
    f"lexical={grounding_score:.3f} "
    f"semantic={semantic_ground:.3f} "
    f"recall={recall_score:.1f}"
)
      
        if force_not_found:

            answer = "This information is not present in the document."
            decision_type = "not_found"

        elif not react_ans or react_ans.strip() == "":
            answer        = "This information is not present in the document."
            decision_type = "empty_answer"

        else:
            if query_type == "VERIFICATION_QA":
                normalized = react_ans.strip().lower()
        

                if recall_score >= 40:
                    answer        = react_ans
                    decision_type = "accepted"
                else:
                    answer        = "This information is not present in the document."
                    decision_type = "low_recall"

            elif query_type == "FACTUAL_QA":

                allow_memory_override = (
                    rewrite_triggered
                    and recall_score >= 80
                    and confidence >= 0.45
                )

                grounded_enough = (
                    grounding_score >= 45
                    or semantic_ground >= 0.45
                    or (
                        recall_score >= 80
                        and semantic_ground >= 0.30
                    )
                    or allow_memory_override
                )

                if grounded_enough:
                    answer = react_ans
                    decision_type = "accepted"

                else:
                    answer = "This information is not present in the document."
                    decision_type = "low_grounding"

            else:
                ans_words = len(react_ans.strip().split())
                if recall_score < 25:
                    answer        = "This information is not present in the document."
                    decision_type = "low_recall"
                elif ans_words <= 3:
                    if recall_score >= 40:
                        answer        = react_ans
                        decision_type = "accepted"
                    else:
                        answer        = "This information is not present in the document."
                        decision_type = "weak_short"
                elif grounding_score < 40 and recall_score < 40:
                    answer        = "This information is not present in the document."
                    decision_type = "low_grounding"
                elif grounding_score >= 50 or recall_score >= 50:
                    answer        = react_ans
                    decision_type = "accepted"
                else:
                    answer        = "This information is not present in the document."
                    decision_type = "uncertain"

   
    if query_type == "VERIFICATION_QA":
        grounding = recall_score
    else:
        grounding = max(grounding_score, semantic_ground * 100)


    confidence = compute_confidence(
    reranker_top=reranker_top,
    recall_score=recall_score,
    answer=answer,
    context_chunks=retrieved_texts,
    question=question
)
    qa_time    = time.time() - qa_start_t

    print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
          f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
          f"confidence={confidence:.3f} | decision={decision_type}")

    if not answer.strip():
        answer = "Could not find a relevant answer in the PDF."
    else:
        answer = normalize_answer(answer)

    state["answer"] = answer
    state["rewrite_triggered"] = rewrite_triggered
    state["rewritten_query"] = rewritten_question
    state["pre_rewrite_docs"] = pre_rewrite_docs
    state["post_rewrite_docs"] = post_rewrite_docs
    _write_metrics(state, model_used, decision_type, grounding,
                   confidence, retrieval_score, context_precision,
                   recall_score, llm_calls, retrieved, qa_time)

    return state

def node_validate(state: DocState) -> DocState:
    answer = state["answer"]
    retry  = state.get("retry_count", 0)

    # ── Retry for empty/very weak answers ────────────────────
    if len(answer.strip()) < 3 and retry < 2:
        state["retry_count"] = retry + 1
        state["answer"]      = ""
        return state

    total_time    = time.time() - state["start_time"]
    output_words  = len(answer.split())
    output_tokens = output_words * 1.3

    extract_time  = state["metrics"].get("extraction_time_sec", 0)
    llm_time      = max(total_time - extract_time, 1)
    tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0

    m = state["metrics"]

    # ── Core metrics ─────────────────────────────────────────
    m["response_time_sec"]    = round(total_time, 2)
    m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
    m["pages_processed"]      = state.get("page_count", 0)
    m["characters_processed"] = state.get("char_count", 0)
    m["words_processed"]      = len(state.get("extracted_text", "").split())

    # ── Type-specific metrics ────────────────────────────────
    if m.get("type") == "summary":
        m["summary_time_sec"]     = m.get("summary_time_sec", 0)
        m["summary_length_words"] = len(answer.split())

    if m.get("type") == "qa":
        m["qa_time_sec"]      = m.get("qa_time_sec", 0)
        m["confidence_score"] = m.get("confidence_score", 0)

    # ── Performance ──────────────────────────────────────────
    m["ttft_sec"]        = round(total_time, 2)
    m["e2e_latency_sec"] = round(total_time, 2)
    m["tps"]             = tps



    # ── Context info ─────────────────────────────────────────
    m["query_type"]     = state.get("query_type", "")
    m["chunks_created"] = m.get("chunks_created", 0)
    m["retry_count"]    = retry

    # ── Model + retrieval metrics ────────────────────────────
    m["model_used"]        = m.get("model_used", "llama_react")
    m["llm_calls"]         = m.get("llm_calls", 0)
    m["retrieval_score"]   = m.get("retrieval_score", 0)
    m["context_precision"] = m.get("context_precision", 0)
    m["answer_grounding"]  = m.get("answer_grounding", 0)
    m["recall_at_k"]       = m.get("recall_at_k", 0)

    # ── Summary-specific ─────────────────────────────────────
    if m.get("type") == "summary":
        m["parallel_workers"] = m.get("parallel_workers", 0)
        m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
        m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)

    # ── QA-specific ──────────────────────────────────────────
    if m.get("type") == "qa":
        m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
        m["decision_type"]    = m.get("decision_type", "accepted")
        m["confidence_raw"]   = m.get("confidence_raw", 0.0)

    state["metrics"] = m
    return state




def _write_metrics(state, model_used, decision_type, grounding,
                   confidence, retrieval_score, context_precision,
                   recall_score, llm_calls, retrieved, qa_time):
    m = state.setdefault("metrics", {})
    m["qa_time_sec"]       = round(qa_time, 2)
    m["confidence_score"]  = round(confidence * 100, 2)
    m["retrieval_score"]   = retrieval_score
    m["context_precision"] = context_precision
    m["answer_grounding"]  = grounding
    m["recall_at_k"]       = recall_score
    m["llm_calls"]         = llm_calls
    m["model_used"]        = model_used
    m["chunks_retrieved"]  = len(retrieved)
    # m["retrieved_docs"] = retrieved
    m["retrieved_docs"] = [
        {
            "content": d.page_content if hasattr(d, "page_content") else str(d),
            "metadata": getattr(d, "metadata", {})
        }
        for d in retrieved
    ]
    m["type"]              = "qa"
    m["decision_type"]     = decision_type
    m["confidence_raw"]    = round(confidence, 4)
    m["rewrite_triggered"] = state.get(
    "rewrite_triggered",
    False
)

    m["rewritten_query"] = state.get(
        "rewritten_query",
        ""
    )

    m["pre_rewrite_docs"] = state.get(
        "pre_rewrite_docs",
        []
    )

    m["post_rewrite_docs"] = state.get(
        "post_rewrite_docs",
        []
    )



def run_pipeline(question: str, pdf_path: str, session_id: str):
    if not session_id:
        session_id = "default_session"
    state = {
        "question": question,
        "pdf_path": pdf_path,
        "session_id": session_id,
        "start_time": time.time(),
        "metrics": {},
    }

    # Execute pipeline manually
    state = node_extract(state)
    state = node_chunk(state)
    state = node_qa(state)
    state = node_validate(state)

    # 🔥 IMPORTANT: expose for evaluation
    state["retrieved_docs"] = state["metrics"].get("retrieved_docs", [])
    state["all_chunks"] = state.get("chunks", [])

    return state








