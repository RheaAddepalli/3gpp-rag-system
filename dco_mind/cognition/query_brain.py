import re


from dco_mind.models.llm import call_llama, call_llama_streaming
from dco_mind.generation.response_generator import REACT_PROMPT, QA_PROMPT

from dco_mind.events.events import emit_event

from dco_mind.utils.helpers import (
    clean_artifacts, 
    clean_chunk_text
)

from dco_mind.reasoning.context_builder import (
    reorder_by_question
)

from dco_mind.evaluation.metrics import (
    compute_answer_grounding,
    compute_retrieval_score,
    compute_context_precision,
)


def clean_reasoning_answer(answer: str, question: str) -> str:
    """
    Remove answers that are pure echoes of the question.
    Generic — no hardcoded trigger words.
    """
    words = answer.strip().split()

    # Never wipe short answers — they may be legitimate facts
    if len(words) <= 3:
        return answer

    q_tokens = set(re.findall(r'\b\w{3,}\b', question.lower()))
    a_tokens = set(re.findall(r'\b\w{3,}\b', answer.lower()))
    new_tokens = a_tokens - q_tokens

    if len(new_tokens) < 3:
        print(f"[ReasoningFilter] ❌ Answer adds only {len(new_tokens)} new tokens → wiping")
        return ""

    return answer



# ============================================================
# ReAct AGENT
# ============================================================
def react_agent(question, faiss_index, query_type, all_chunks, request_id, recall_score):
    grounding = 0.0
    """
    Returns:
        answer, model_used, steps, retrieval_score,
        context_precision, grounding, llm_calls, context_chunks
    """
    MAX_STEPS  = 3
    scratchpad = ""
    model_used = "llama_react"
    llm_calls  = 0

    # ── Retrieval ─────────────────────────────────────────────
    context_chunks = [
        d if isinstance(d, str) else d.page_content
        for d in all_chunks
    ]

    retrieval_score = compute_retrieval_score(question, context_chunks)
    context_precision = compute_context_precision(question, context_chunks)
   

    print(
        f"[ReAct] Starting | {query_type} | {len(context_chunks)} chunks | "
        f"retrieval={retrieval_score:.1f}%"
    )

    emit_event(
        request_id,
        "agent_start",
        f"🤖 Agent starting | {query_type} | {len(context_chunks)} chunks"
    )

    # ── Step loop ─────────────────────────────────────────────
    for step in range(MAX_STEPS):
        ranked_chunks = reorder_by_question(question, context_chunks)
        top_chunks    = ranked_chunks[:7]

        print("\n========== DEBUG: TOP CHUNKS ==========")
        for i, chunk in enumerate(top_chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(chunk[:300].replace("\n", " "))
        print("======================================\n")

        context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)

        print("\n[DEBUG] ===== CLEANED CONTEXT PREVIEW =====")
        print(context[:200])
        print("=========================================\n")
        print(f"[DEBUG] Full context sent to LLM:\n{context[:2500]}")
        raw = call_llama(
            REACT_PROMPT.format(
                question=question,
                context=context[:2500],
                scratchpad=scratchpad if scratchpad else "None yet"
            ),
            temperature=0.0
        )
        llm_calls += 1
        print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

        # ── Parse LLM output ──────────────────────────────────
        action       = ""
        action_input = ""
        lines        = raw.split("\n")

        thought_text = ""
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("Thought:"):
                thought_text = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action_raw   = line.replace("Action:", "").strip()
                action_lower = action_raw.lower()
                if "final" in action_lower or "answer" in action_lower:
                    action = "final_answer"
            
                    if ":" in action_raw:
                        inline = action_raw.split(":", 1)[1].strip()
                        if inline:
                            action_input = inline
                elif "search" in action_lower or "more" in action_lower:
                    action = "search_more"
                else:
                    action       = "final_answer"
                    action_input = action_raw
                    print(f"[ReAct] ⚠️ Answer rescued from Action field: '{action_raw[:60]}'")
            elif line.startswith("Input:"):
                if not action_input:
                    action_input = line.replace("Input:", "").strip()
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if not next_line:
                            continue
                        if next_line.startswith(("Thought:", "Action:", "Input:")):
                            break
                        action_input += " " + next_line

        if action_input:
            action_input = clean_artifacts(action_input)
        if not action:
            action = "final_answer"
            action_input = ""
   
        action_input = clean_artifacts(action_input) if action_input else ""
        if not action_input or len(action_input.strip()) < 2:

            if thought_text:

                if query_type == "VERIFICATION_QA":
                    confidence = round(recall_score / 100, 3)
                    print("[ReAct] ⚠️ Converting Thought → Yes/No using signals")

                    if recall_score >= 50:
                        action_input = "Yes"
                    else:
                        action_input = "No"

                else:
  
                    action_input = thought_text
        if "final_answer" in action:

            print(f"[DEBUG] action_input raw: repr={repr(action_input)}")
            answer = action_input.strip() if action_input else ""
            print(f"[DEBUG] answer after strip: repr={repr(answer)}")
            if answer.strip().lower() in [
    "this information is not present in the document.",
    "not present",
    "not found"
]:
                if retrieval_score > 30: 
                    print("[ReAct] ⚠️ Likely wrong refusal → retrying with QA prompt")
                    ranked = reorder_by_question(question, context_chunks)
                    ctx = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])
                    answer = call_llama(
                        QA_PROMPT.format(
                            context=ctx[:2500],
                            question=question
                        ),
                        temperature=0.0
                    )
                    llm_calls += 1
            if not answer:
                answer = "This information is not present in the document."
            elif answer.lower() == question.strip().lower():
                answer = "This information is not present in the document."
            else:
                cleaned = clean_reasoning_answer(answer, question)
                if cleaned:
                    answer = cleaned

            print(f"[DEBUG] answer before grounding: repr={repr(answer)}")
         
            grounding = compute_answer_grounding(answer, context_chunks, question)

            return (
                answer,
                "llama_react",
                step + 1,
                retrieval_score,
                context_precision,
                grounding,
                llm_calls,
                context_chunks
            )
           

        elif "search" in action:
            print("[ReAct] ❌ Skipping search → NOT FOUND")

            return (
                "This information is not present in the document.",
                "llama_react_fail",
                step + 1,
                retrieval_score,
                context_precision,
                0.0,
                llm_calls,
                context_chunks
            )

        else:
            if len(raw) > 20:
                grounding = compute_answer_grounding(raw, context_chunks)
                return (
                    raw, "llama_react_direct", step + 1,
                    retrieval_score, context_precision,
                    grounding, llm_calls, context_chunks
                )

    emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
    ranked  = reorder_by_question(question, context_chunks)
    ctx     = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])

    emit_event(request_id, "stream_start", "✍️ Generating final answer...")
    final, ttft = call_llama_streaming(
        QA_PROMPT.format(
            context=ctx[:2500],
            question=question
        ),
        request_id=request_id,
        temperature=0.0
    )
    llm_calls += 1
    grounding = compute_answer_grounding(final, context_chunks)

    return (
        final, "llama_react_final", MAX_STEPS,
        retrieval_score, context_precision,
        grounding, llm_calls, context_chunks
    )














