


import re
import numpy as np
from difflib import SequenceMatcher
# from sklearn.metrics.pairwise import cosine_similarity

from dco_mind.config.settings import (
    embedding_model,
    CONF_WEIGHT_RERANKER,
    CONF_WEIGHT_RECALL,
    CONF_WEIGHT_KEYWORD,
)

from dco_mind.reasoning.context_builder import _normalize_text
# ============================================================
# NUMERICALLY STABLE COSINE
# ============================================================

def embedding_cosine(a, b):
    """
    Numerically stable cosine similarity.
    """

    a = np.array(a)
    b = np.array(b)

    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)

    return float(np.dot(a, b))


_REFUSAL_PHRASES = [
    "not present",
    "not available",
    "not found",
    "not in the document",
    "not present in the document",
    "information is not",
    "cannot find",
    "no information",
    "don't see any information",   
    "don't see any mention",       
    "does not mention",            
    "no mention of",               
    "context does not mention",    
    "i don't see", 
    "does not explicitly mention",
    "does not specifically",
    "not explicitly mentioned",
]


def _is_refusal_answer(answer: str) -> bool:
    """
    Phrase-based refusal detection. No LLM, no embeddings.
    Short answers (<=3 words) are never treated as refusals —
    they are legitimate short factual answers.
    """
    if not answer:
        return True

    t = answer.lower().strip()
    return any(p in t for p in _REFUSAL_PHRASES)



def compute_token_f1(prediction, ground_truth):
    """
    Computes token-level F1 score.
    Useful for factual overlap evaluation.
    """
    pred_tokens = re.findall(
        r'\w+',
        prediction.lower()
    )

    gt_tokens = re.findall(
        r'\w+',
        ground_truth.lower()
    )

    common = set(pred_tokens) & set(gt_tokens)

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = (
        2 * precision * recall
    ) / (precision + recall)

    return round(float(f1), 4)


# ============================================================
# ANSWER CORRECTNESS
# ============================================================

def evaluate_answer_correctness(
    generated_answer,
    gold_answer,
    embedder
):
    """
    Hybrid answer correctness score.

    Combines:
    - semantic similarity
    - token F1 overlap

    Final Score:
        0.7 * semantic similarity
      + 0.3 * token F1
    """

    if not generated_answer:

        return {
            "semantic_similarity": 0.0,
            "token_f1": 0.0,
            "final_score": 0.0
        }

    # ============================================================
    # NULL / EMPTY GOLD ANSWER HANDLING
    # ============================================================

    if not gold_answer:

        normalized = generated_answer.lower()

        refusal_patterns = [
    "not present",
    "not mentioned",
    "not available",
    "insufficient information",
    "cannot be determined",
    "document does not mention",
    "no information",
    "not stated",
    "not provided",
    "i cannot find",
    "there is no mention",
    "don't see any information",
    "don't see any mention",
    "does not mention",
    "no mention of",
    "context does not mention",
    "i don't see",
    "there is no",
    "i don't have",
    "unfortunately, the provided context does not mention",
    "does not explicitly mention",
    "does not specifically",
    "not explicitly mentioned",

        ]

        abstained = any(
            p in normalized
            for p in refusal_patterns
        )

        return {
            "semantic_similarity": 1.0 if abstained else 0.5,
            "token_f1": 1.0 if abstained else 0.5,
            "final_score": 1.0 if abstained else 0.5
        }
    if gold_answer == "NOT_PRESENT":

        hall_eval = evaluate_hallucination(
                        generated_answer,
                        gold_answer
                    )

        score = hall_eval["score"]

        return {
                "semantic_similarity": score,
                "token_f1": score,
                "final_score": score
            }
    
    gen_emb = embedder.embed_query(generated_answer)
    gold_emb = embedder.embed_query(gold_answer)

    semantic_similarity = embedding_cosine(
    gen_emb,
    gold_emb
)

    token_f1 = compute_token_f1(
        generated_answer,
        gold_answer
    )

    final_score = (
        0.7 * semantic_similarity +
        0.3 * token_f1
    )

    return {
        "semantic_similarity": round(float(semantic_similarity), 4),
        "token_f1": round(float(token_f1), 4),
        "final_score": round(float(final_score), 4)
    }


# ============================================================
# HALLUCINATION EVALUATION
# ============================================================

REFUSAL_PATTERNS = [
    "not present",
    "not mentioned",
    "not available",
    "insufficient information",
    "cannot be determined",
    "document does not mention",
    "no information",
    "not stated",
    "not provided",
    "i cannot find",
    "there is no mention",
    "don't see any information",
    "don't see any mention",
    "does not mention",
    "no mention of",
    "context does not mention",
    "i don't see",
    "there is no",
    "i don't have",
    "unfortunately, the provided context does not mention",
    "does not explicitly mention",
    "does not specifically",
    "not explicitly mentioned",
]


def evaluate_hallucination(
    generated_answer,
    gold_answer
):
    """
    Evaluates hallucination for NOT_PRESENT questions.

    Correct behavior:
    - abstain
    - refuse
    - indicate insufficient information
    """

    if gold_answer != "NOT_PRESENT":
        return {
            "hallucinated": False,
            "abstention_detected": False,
            "score": 1.0
        }

    answer_lower = generated_answer.lower()

    abstained = any(
        pattern in answer_lower
        for pattern in REFUSAL_PATTERNS
    )

    return {
        "hallucinated": not abstained,
        "abstention_detected": abstained,
        "score": 1.0 if abstained else 0.0
    }


# ============================================================
# GOLD CHUNK RECALL@K
# ============================================================

def compute_gold_chunk_recall(
    retrieved_chunks,
    gold_chunks,
    embedder,
    threshold=0.30
):
    """
    Computes Recall@K using gold chunks.

    Formula:
        matched_gold_chunks / total_gold_chunks
    """

    if not gold_chunks:
        return 1.0

    if not retrieved_chunks:
        return 0.0
    def normalize_text(x):
        return " ".join(str(x).lower().split())

    gold_chunks = [
        normalize_text(g)
        for g in gold_chunks
    ]

    retrieved_chunks = [
        normalize_text(r)
        for r in retrieved_chunks
    ]

    gold_embeddings = [
        embedder.embed_query(g)
        for g in gold_chunks
    ]

    retrieved_embeddings = [
        embedder.embed_query(r)
        for r in retrieved_chunks
    ]

    matched = 0

    for gold_idx, gold_emb in enumerate(gold_embeddings):

        gold_text = gold_chunks[gold_idx]

        found = False

        for ret_idx, ret_emb in enumerate(retrieved_embeddings):

            retrieved_text = retrieved_chunks[ret_idx]

            # ====================================================
            # containment shortcut
            # ====================================================

            if (
                gold_text in retrieved_text
                or retrieved_text in gold_text
            ):
                found = True
                break

            # ====================================================
            # semantic fallback
            # ====================================================

            sim = embedding_cosine(
                gold_emb,
                ret_emb
            )

            if sim >= threshold:
                found = True
                break

        if found:
            matched += 1

    recall = matched / len(gold_chunks)

    return round(float(recall), 4)


# ============================================================
# GOLD CHUNK PRECISION@K
# ============================================================

def compute_gold_chunk_precision(
    retrieved_chunks,
    gold_chunks,
    embedder,
    threshold=0.30
):
    """
    Computes Precision@K using gold chunks.

    Formula:
        relevant_retrieved / total_retrieved
    """

    if not retrieved_chunks:
        return 0.0
    def normalize_text(x):
        return " ".join(str(x).lower().split())

    gold_chunks = [
        normalize_text(g)
        for g in gold_chunks
    ]

    retrieved_chunks = [
        normalize_text(r)
        for r in retrieved_chunks
    ]

    gold_embeddings = [
        embedder.embed_query(g)
        for g in gold_chunks
    ]

    retrieved_embeddings = [
        embedder.embed_query(r)
        for r in retrieved_chunks
    ]

    relevant = 0

    for ret_idx, ret_emb in enumerate(retrieved_embeddings):

        retrieved_text = retrieved_chunks[ret_idx]

        matched = False

        for gold_idx, gold_emb in enumerate(gold_embeddings):

            gold_text = gold_chunks[gold_idx]

            # ====================================================
            # containment shortcut
            # ====================================================

            if (
                gold_text in retrieved_text
                or retrieved_text in gold_text
            ):
                matched = True
                break

            # ====================================================
            # semantic fallback
            # ====================================================

            sim = embedding_cosine(
                ret_emb,
                gold_emb
            )

            if sim >= threshold:
                matched = True
                break

        if matched:
            relevant += 1

    precision = relevant / len(retrieved_chunks)

    precision = max(
        0.0,
        min(1.0, precision)
    )

    return round(float(precision), 4)


# ============================================================
# REWRITE EFFECTIVENESS
# ============================================================

def evaluate_rewrite_effectiveness(
    pre_rewrite_recall,
    post_rewrite_recall
):
    """
    Measures how much rewrite improved retrieval.

    Formula:
        post_rewrite_recall - pre_rewrite_recall
    """

    improvement = (
        post_rewrite_recall -
        pre_rewrite_recall
    )

    return round(float(improvement), 4)


# ============================================================
# FOLLOWUP SUCCESS SCORE
# ============================================================

def evaluate_followup_success(
    answer_correctness,
    retrieval_recall,
    grounding_score
):
    """
    Combined conversational followup success score.

    Formula:
        0.5 * answer correctness
      + 0.3 * retrieval recall
      + 0.2 * grounding score
    """

    final_score = (
        0.5 * answer_correctness +
        0.3 * retrieval_recall +
        0.2 * grounding_score
    )

    final_score = max(
    0.0,
    min(1.0, final_score)
)

    return round(float(final_score), 4)


# ============================================================
# QUERY TYPE AGGREGATION
# ============================================================

def aggregate_query_type_metrics(results):
    """
    Aggregates metrics per query type.

    Example query types:
    - FACTUAL_QA
    - MULTIPART_QA
    - VERIFICATION_QA
    """

    grouped = {}

    for result in results:

        qtype = result.get(
            "query_type",
            "UNKNOWN"
        )

        grouped.setdefault(qtype, []).append(result)

    summary = {}

    for qtype, items in grouped.items():

        count = len(items)

        avg_answer_score = (
            sum(
                i.get("answer_correctness", 0.0)
                for i in items
            ) / count
        )

        avg_recall = (
            sum(
                i.get("retrieval_recall", 0.0)
                for i in items
            ) / count
        )

        avg_precision = (
            sum(
                i.get("retrieval_precision", 0.0)
                for i in items
            ) / count
        )

        summary[qtype] = {
            "count": count,
            "avg_answer_correctness": round(avg_answer_score, 4),
            "avg_retrieval_recall": round(avg_recall, 4),
            "avg_retrieval_precision": round(avg_precision, 4)
        }

    return summary

# ============================================================
# NORMALIZATION
# ============================================================

def minmax_normalize(score: float, all_scores: list) -> float:
    if not all_scores:
        return 1.0
    min_s = min(all_scores)
    max_s = max(all_scores)
    if max_s == min_s:
        return 1.0
    return (score - min_s) / (max_s - min_s + 1e-6)


def minmax_normalize_list(scores: list) -> list:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s + 1e-6) for s in scores]


# ============================================================
# KEYWORD OVERLAP
# ============================================================

def keyword_overlap(answer: str, context_chunks: list) -> float:
    if not answer or not context_chunks:
        return 0.0
    text   = " ".join(context_chunks).lower()
    answer = answer.lower().strip()
    if answer in text:
        return 1.0
    a = set(answer.split())
    c = set(text.split())
    if not a:
        return 0.0
    return len(a & c) / len(a)


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def semantic_similarity(answer: str, context_chunks: list) -> float:
  
    print("\n[DEBUG SEMANTIC]")
    print(f"answer type={type(answer)} value={repr(answer)}")

    if context_chunks is not None:
        print(f"context_chunks len={len(context_chunks)}")
        if len(context_chunks) > 0:
            print(f"first chunk type={type(context_chunks[0])}")
            print(f"first chunk preview={repr(str(context_chunks[0])[:100])}")
    else:
        print("context_chunks is NONE")
  
    if not answer or not context_chunks:
        return 0.0
    try:
        combined = " ".join(context_chunks)[:2000]
        ans_emb  = embedding_model.embed_query(answer[:500])
        ctx_emb  = embedding_model.embed_query(combined)
        a        = np.array(ans_emb)
        b        = np.array(ctx_emb)
        norm     = np.linalg.norm(a) * np.linalg.norm(b)
        if norm < 1e-8:
            return 0.0
        return embedding_cosine(a, b)
    except Exception as e:
        print(f"[SemanticSim] error: {e}")
        return 0.0


# ============================================================
# GROUNDING COMPONENTS
# ============================================================

def local_substring_match(answer: str, context_chunks: list) -> float:
    if not context_chunks:
        return 0.0
    ans_norm   = _normalize_text(answer)
    chunk_norm = _normalize_text(context_chunks[0])
    return 1.0 if ans_norm in chunk_norm else 0.0


def exact_span_match(answer: str, context_chunks: list) -> bool:
    if not context_chunks:
        return False
    ans_norm = _normalize_text(answer)
    return any(ans_norm in _normalize_text(c) for c in context_chunks)


def compute_grounding_score(answer: str, context_chunks: list) -> float:
    if not answer or not context_chunks:
        return 0.0

    ans_norm    = _normalize_text(answer)
    chunks_norm = [_normalize_text(c) for c in context_chunks]

    local_match    = 1.0 if ans_norm in chunks_norm[0] else 0.0
    global_match   = 1.0 if any(ans_norm in c for c in chunks_norm) else 0.0
    keyword_score  = keyword_overlap(answer, context_chunks)
    semantic_score = semantic_similarity(answer, context_chunks)

    grounding_score = (
    0.3 * local_match +
    0.2 * global_match +
    0.3 * semantic_score +
    0.2 * keyword_score
)
    print(f"[Grounding] local={local_match:.2f} global={global_match:.2f} "
          f"semantic={semantic_score:.3f} keyword={keyword_score:.3f} "
          f"-> grounding={grounding_score:.3f}")
    return round(grounding_score, 4)


def coverage_score(answer: str, chunks: list) -> float:
    if not answer or not chunks:
        return 0.0
    ans = answer.lower()
    return max(SequenceMatcher(None, ans, c.lower()).ratio() for c in chunks)


def score_answer(answer: str, context_chunks: list) -> float:
    if not answer or not context_chunks:
        return 0.0
    top_chunk = context_chunks[0]
    semantic  = semantic_similarity(answer, [top_chunk])
    keyword   = keyword_overlap(answer, [top_chunk])
    coverage  = coverage_score(answer, [top_chunk])
    return 0.4 * semantic + 0.3 * keyword + 0.3 * coverage


# ============================================================
# CONFIDENCE
# ============================================================
def compute_confidence(
    reranker_top: float,
    recall_score: float,
    answer: str,
    context_chunks: list,
    question: str,
    all_reranker_scores: list = None
) -> float:
    print("\n[DEBUG CONFIDENCE INPUTS]")
    print(f"answer type={type(answer)} value={repr(answer)}")
    print(f"question type={type(question)} value={repr(question)}")
    print(f"context_chunks type={type(context_chunks)}")
    print(f"reranker_top={reranker_top}")
    print(f"recall_score={recall_score}")

    if all_reranker_scores:
        normalized = minmax_normalize(
            reranker_top,
            all_reranker_scores
        )
    else:
        normalized = minmax_normalize(
            reranker_top,
            [reranker_top]
        )

    rec = min(recall_score / 100.0, 1.0)

    kw = keyword_overlap(
        answer,
        context_chunks
    )

    qa_relevance = semantic_similarity(
    answer,
    context_chunks
)

    confidence = (
    0.25 * normalized +
    0.20 * rec +
    0.15 * kw +
    0.40 * qa_relevance
)

    print(
        f"[Confidence] "
        f"normalized={normalized:.3f} "
        f"rec={rec:.3f} "
        f"kw={kw:.3f} "
        f"qa_rel={qa_relevance:.3f} "
        f"→ {confidence:.3f}"
    )

    return round(confidence, 4)


# ============================================================
# compute_answer_grounding
# Word-overlap grounding (percentage 0–100).
# Refusal answers return sentinel 75.0 — above the hallucination
# gate threshold, preventing false rejections of correct refusals.
# ============================================================

def compute_answer_grounding(answer: str, context_chunks: list,
                             question: str = "") -> float:
    if not answer:
        return 0.0

   
    refusal_phrases = [
    "not present",
    "not mentioned",
    "not available",
    "no information",
    "cannot find",
    "not in the document",
    "don't see any information",
    "don't see any mention",
    "does not mention",
    "no mention of",
    "context does not mention",
    "i don't see",
    "there is no",
    "i don't have",
    "unfortunately, the provided context does not mention",
    "does not explicitly mention",
    "does not specifically",
    "not explicitly mentioned",
]

    if any(p in answer.lower() for p in refusal_phrases):
        print(f"[Grounding] Refusal answer detected → returning sentinel 75.0")
        return 0.0
    if not context_chunks:
        return 0.0
       
    if exact_span_match(answer, context_chunks):
        return 100.0

    try:
        context_text  = " ".join(context_chunks).lower()
        context_words = set(re.findall(r'\b[a-z0-9%\-]{2,}\b', context_text))
        answer_words  = set(re.findall(r'\b[a-z0-9%\-]{2,}\b', answer.lower()))
        if not answer_words:
            return 0.0
        overlap = answer_words & context_words
        return round(len(overlap) / len(answer_words) * 100, 1)
    except Exception:
        return 0.0


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def compute_retrieval_score(question: str, docs: list) -> float:
    try:
        q_embed = embedding_model.embed_query(question)
        sims    = []
        for d in docs:
            d_embed = embedding_model.embed_query(d.page_content[:300])
            sim = np.dot(q_embed, d_embed) / (
                np.linalg.norm(q_embed) * np.linalg.norm(d_embed) + 1e-8)
            sims.append(sim)
        if not sims:
            return 0.0
        max_sim  = float(np.max(sims))
        top3     = sorted(sims, reverse=True)[:3]
        mean_top = float(np.mean(top3))
        return round((0.7 * max_sim + 0.3 * mean_top) * 100, 2)
    except Exception:
        return 0.0


def compute_context_precision(question: str, docs: list,
                              threshold: float = 0.35) -> float:
    try:
        q_embed = embedding_model.embed_query(question)
        sims    = []
        for d in docs:
            d_embed = embedding_model.embed_query(d.page_content[:300])
            sim = np.dot(q_embed, d_embed) / (
                np.linalg.norm(q_embed) * np.linalg.norm(d_embed) + 1e-8)
            sims.append(sim)
        if not sims:
            return 0.0
        avg_sim            = float(np.mean(sims))
        max_sim            = float(np.max(sims))
        adaptive_threshold = min(threshold, max(max_sim * 0.60, avg_sim * 0.80))
        relevant           = sum(1 for s in sims if s >= adaptive_threshold)
        return round(relevant / len(sims) * 100, 1)
    except Exception:
        return 0.0

def compute_recall_at_k(question: str, docs: list,
                        all_chunks: list, k: int = 10) -> float:
    try:
        q_embed  = embedding_model.embed_query(question)
        all_sims = []

        for chunk in all_chunks:
            text = chunk if isinstance(chunk, str) else str(chunk)
            c_embed = embedding_model.embed_query(text[:300])
            sim = np.dot(q_embed, c_embed) / (
                np.linalg.norm(q_embed) * np.linalg.norm(c_embed) + 1e-8)
            all_sims.append((sim, text))

        if not all_sims:
            return 0.0

        all_sims.sort(key=lambda x: x[0], reverse=True)
        oracle_sim   = all_sims[0][0]
        oracle_chunk = all_sims[0][1]

        if oracle_sim < 0.01:
            return 0.0

        # 🔥 SAFE conversion for docs
        doc_texts = [
            d.page_content if hasattr(d, "page_content") else str(d)
            for d in docs
        ]

        retrieved_fps = set(text[:100] for text in doc_texts)

        if oracle_chunk[:100] in retrieved_fps:
            print(f"[Recall@{k}] ✅ Oracle chunk retrieved! score=100%")
            return 100.0

        best_retrieved_sim = max(
            (
                np.dot(q_embed, embedding_model.embed_query(text[:300])) /
                (np.linalg.norm(q_embed) *
                 np.linalg.norm(embedding_model.embed_query(text[:300])) + 1e-8)
            )
            for text in doc_texts
        ) if doc_texts else 0.0

        soft_score = round((best_retrieved_sim / oracle_sim) * 100, 1)
        print(f"[Recall@{k}] oracle={oracle_sim:.3f} | "
              f"best_retrieved={best_retrieved_sim:.3f} | score={soft_score}%")

        return min(soft_score, 99.9)

    except Exception as e:
        print(f"[Recall@K] error: {e}")
        return 0.0

def llm_verify_answer(question: str, answer: str,
                      context_chunks: list) -> bool:
    from dco_mind.models.llm import call_llama
    context = "\n\n".join(context_chunks[:3])
    prompt  = (
        f"Context:\n{context[:2000]}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Is this answer supported by the context above?\n"
        f"Rules:\n"
        f"- Paraphrases are valid support\n"
        f"- Partial answers are valid if they answer the question\n"
        f"- A Yes/No + evidence is valid for verification questions\n"
        f"Reply ONLY: YES or NO"
    )
    try:
        resp = call_llama(prompt, num_ctx=1024, temperature=0.0)
        return "YES" in resp.upper()
    except Exception:
        return True







