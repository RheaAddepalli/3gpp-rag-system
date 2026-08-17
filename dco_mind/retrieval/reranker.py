
from dco_mind.config.settings import reranker, RERANKER_PRUNE_MARGIN, FACTUAL_TOP_K
from dco_mind.retrieval.adaptive_search import exact_match_retrieve
# ============================================================
# RERANK WITH RELATIVE PRUNING + TOP-K CAP
# Returns (docs, top_score, all_scores)
# ============================================================
def rerank_docs(question: str, docs: list,
                top_k: int = FACTUAL_TOP_K,
                apply_pruning: bool = True) -> tuple:
    if not docs:
        return docs, -99.0, []

    try:
        pairs      = [(question, d.page_content[:512]) for d in docs]
        scores_arr = reranker.predict(pairs)
        all_scores = [float(s) for s in scores_arr]
        ranked     = sorted(zip(all_scores, docs), key=lambda x: x[0], reverse=True)
        top_score  = float(ranked[0][0])
      
        if top_score < -5:
            print(f"[Reranker] ⚠️ Very low scores → keeping top chunks instead of rejecting")

            safe_k = max(3, min(top_k, len(ranked)))
            result_docs = [d for _, d in ranked[:safe_k]]

            return result_docs, top_score, all_scores
    
        if apply_pruning:
          
            margin = RERANKER_PRUNE_MARGIN
            pruned = [(s, d) for s, d in ranked if s >= top_score - margin]
      

            MIN_CHUNKS = max(3, min(len(ranked), top_k))

            if len(pruned) < MIN_CHUNKS:
                print(f"[Reranker] ⚠️ Too few chunks after pruning ({len(pruned)}) → expanding")

                # fallback to top-N ranked instead of margin pruning
                result_docs = [d for _, d in ranked[:MIN_CHUNKS]]
            else:
                result_docs = [d for _, d in pruned]
            bottom_score = pruned[-1][0] if pruned else -999

            print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks after pruning | "
                f"top={top_score:.3f} | margin={margin} | "
                f"bottom kept={bottom_score:.3f}")
        else:
            result_docs = [d for _, d in ranked[:top_k]]
            print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks (no pruning) | "
                  f"top={top_score:.3f}")

        return result_docs, top_score, all_scores

    except Exception as e:
        print(f"[Reranker] error: {e} — returning original order")
        return docs[:top_k], -99.0, []


def protect_exact_matches(question: str, reranked_docs: list,
                          all_chunks: list,
                          top_k: int = FACTUAL_TOP_K) -> list:
    if not all_chunks:
        return reranked_docs

    exact_docs = exact_match_retrieve(question, all_chunks)
    if not exact_docs:
        return reranked_docs

    reranked_fps = set(d.page_content[:120].strip() for d in reranked_docs)
    missing = [d for d in exact_docs
               if d.page_content[:120].strip() not in reranked_fps]

    if not missing:
        print(f"[Protect] All exact-match chunks already in reranked set ✅")
        return reranked_docs

    print(f"[Protect] Injecting {len(missing)} exact-match chunks dropped by reranker")

    protected = reranked_docs + missing
    expanded_k = max(top_k, len(missing) + len(reranked_docs))
    expanded_k = min(expanded_k, FACTUAL_TOP_K + len(missing)) # this is very imp just by changinng this from this expanded_k = min(expanded_k, 5) i got crct retreival  

    print(f"[Protect] Context expanded: {top_k} → {expanded_k} chunks")
    return protected[:expanded_k]














