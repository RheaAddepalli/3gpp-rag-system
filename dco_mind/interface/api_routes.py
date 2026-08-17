import os
import json
import time
from flask import Response
from flask import request, jsonify
import sys
import contextlib
import datetime

from dco_mind.models.llm import call_llama, _ttft_tracker
from dco_mind.evaluation.metrics import (
    evaluate_answer_correctness,
    evaluate_hallucination,
    compute_gold_chunk_recall,
    compute_gold_chunk_precision,
    evaluate_rewrite_effectiveness,
    evaluate_followup_success,
    aggregate_query_type_metrics
)
from dco_mind.core.state import DocState
from dco_mind.core.engine import node_qa, node_validate

from dco_mind.knowledge.ingestion import _extraction_cache, _summary_cache

from dco_mind.models.embeddings import _faiss_cache

from dco_mind.events.events import emit_event, cleanup_queue
from dco_mind.config.settings import OLLAMA_MODEL, device

class BackendTee:
    def __init__(self, original, logfile):
        self.original = original
        self.logfile = logfile

    def write(self, text):
        self.original.write(text)
        self.original.flush()

        if text:
            self.logfile.write(text)
            self.logfile.flush()

    def flush(self):
        self.original.flush()
        self.logfile.flush()

def register_routes(app, workflow, all_chunks, faiss_index):
    """Register all Flask routes onto the given app instance.

    all_chunks / faiss_index are the precomputed, fixed 3GPP
    corpus (both specs merged, source-tagged) — loaded once at
    startup in app.py and passed in here for /ask to use directly.
    """

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "model":  OLLAMA_MODEL,
            "device": "cuda" if device == 0 else "cpu",
            "faiss_cached":     len(_faiss_cache),
            "extracted_pdfs":   len(_extraction_cache),
            "summaries_cached": len(_summary_cache),
            "corpus_chunks":    len(all_chunks),
        })

    @app.route("/evaluate", methods=["POST"])
    def evaluate():

        try:
            data        = request.json or {}
            session_name = data.get("session_name", "").strip()
            if session_name:
                import builtins, datetime as _dt
                _LOG_DIR = r"C:\xampp\htdocs\GenAI-Doc-old\dco_mind\evaluation\results\backend_logs"
                _new_log = open(os.path.join(_LOG_DIR, f"run_logs_backend_{session_name}.txt"), "a", encoding="utf-8", buffering=1)
                _rp = builtins._real_print
                def _tee(*args, **kwargs):
                    _rp(*args, **kwargs)
                    msg = kwargs.get("sep", " ").join(str(a) for a in args) + kwargs.get("end", "\n")
                    _new_log.write(f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}")
                    _new_log.flush()
                builtins.print = _tee

            pdf_path    = data.get("pdf_path", "").strip()
            run_desc = (
            session_name
            if session_name
            else data.get("run_description", f"run_{int(time.time())}")
        )

            if not pdf_path or not os.path.exists(pdf_path):
                return jsonify({"error": f"PDF not found: {pdf_path}"}), 404

            pdf_name = os.path.basename(pdf_path).lower().strip()
            print(f"[DEBUG] PDF NAME: {pdf_name}")

            dataset_map = {
                "rhea resume-ziroh labs.pdf.pdf": "datasets/resume2.json",
                "the-story-of-doctor-dolittle.pdf": "datasets/story2.json",
                "ml.pdf": "datasets/ml2.json"
            }
            print(f"[DEBUG] DATASET MAP KEYS: {list(dataset_map.keys())}")

            dataset_file = dataset_map.get(pdf_name)

            if not dataset_file:
                return jsonify({"error": f"No dataset mapping for {pdf_name}"}), 404

            dataset_file = os.path.join(os.path.dirname(__file__), "..", dataset_file)
            dataset_file = os.path.abspath(dataset_file)

            print(f"[DEBUG] FULL DATASET PATH: {dataset_file}")

            if not os.path.exists(dataset_file):
                return jsonify({"error": f"Dataset file missing: {dataset_file}"}), 404

            with open(dataset_file, "r") as f:
                dataset = json.load(f)

                questions = (
                dataset
                if isinstance(dataset, list)
                else dataset.get("questions", [])
            )
            if not questions:
                return jsonify({"error": "No questions found in grounding_dataset.json"}), 400

            results       = []
            pass_count    = 0
            partial_count = 0
            fail_count    = 0

            print(f"[Evaluate] Starting evaluation run: {run_desc}")
            print(f"[Evaluate] PDF: {pdf_path}")
            print(f"[Evaluate] Questions: {len(questions)}")

            for item in questions:
                    q_id       = item.get("id", "?")
                    depends_on = item.get("depends_on")
                    is_followup = item.get("followup_question", False)

                    query_type = item.get("query_type", "FACTUAL_QA")
                    is_roberta = item.get("roberta_test", False)

                    if item.get("skip", False):
                        print(f"[Evaluate] Q{q_id}: ⏭️ SKIPPED")
                        continue

                    question = item.get("question", "")
                    print(f"[Evaluate] Q{q_id}: {question[:60]}...")

                    initial_state: DocState = {
                        "pdf_path":       pdf_path,
                        "question":       question,
                        "original_question": question,
                        "extracted_text": "",
                        "chunks":         [],
                        "summary_chunks": [],
                        "faiss_index":    None,
                        "answer":         "",
                        "metrics":        {},
                        "doc_type":       "general",
                        "query_type":     "",
                        "retry_count":    0,
                        "start_time":     time.time(),
                        "page_count":     0,
                        "char_count":     0,
                        "request_id":     ""
                    }

                    try:
                        result     = workflow.invoke(initial_state)
                        answer     = result.get("answer", "")
                        metrics    = result.get("metrics", {})
                        model_used = metrics.get("model_used", "unknown")
                        recall_k   = metrics.get("recall_at_k", 0)
                        grounding  = metrics.get("answer_grounding", 0)
                        confidence = metrics.get("confidence_score", 0)
                        print(f"[Evaluate] Q{q_id} actual answer: {answer[:300]}")
                    except Exception as invoke_err:
                        print(f"[Evaluate] Q{q_id} workflow error: {invoke_err}")
                        answer     = f"ERROR: {str(invoke_err)}"
                        model_used, recall_k, grounding, confidence = "error", 0, 0, 0
                        metrics = {}
                    gold_answer = item.get("gold_answer", "")
                    gold_chunks = item.get("gold_chunks", [])

                    retrieved_docs = metrics.get("retrieved_docs", [])
                    print(f"[DEBUG] retrieved_docs count = {len(retrieved_docs)}")
                    print(f"[DEBUG] gold_chunks count = {len(gold_chunks)}")

                    retrieved_texts = [
                        d["content"] if isinstance(d, dict)
                        else str(d)
                        for d in retrieved_docs
                    ]

                    if not _faiss_cache:
                        raise RuntimeError("FAISS cache is empty during evaluation")

                    embedder = _faiss_cache[
                        next(iter(_faiss_cache))
                    ].embedding_function

                    answer_eval = evaluate_answer_correctness(
                        generated_answer=answer,
                        gold_answer=gold_answer,
                        embedder=embedder
                    )

                    answer_score = answer_eval["final_score"]
                    gold_norm = gold_answer.strip().lower()
                    ans_norm = answer.strip().lower()

                    if gold_norm and gold_norm in ans_norm:
                        answer_score = max(answer_score, 0.90)

                    hall_eval = evaluate_hallucination(
                        generated_answer=answer,
                        gold_answer=gold_answer
                    )

                    hallucination_detected = hall_eval["hallucinated"]

                    retrieval_recall = compute_gold_chunk_recall(
                        retrieved_chunks=retrieved_texts,
                        gold_chunks=gold_chunks,
                        embedder=embedder,
                        threshold=0.30
                    )

                    retrieval_precision = compute_gold_chunk_precision(
                        retrieved_chunks=retrieved_texts,
                        gold_chunks=gold_chunks,
                        embedder=embedder,
                        threshold=0.30
                    )

                    rewrite_gain = 0.0
                    followup_score = None

                    if is_followup:
                        pre_docs = metrics.get("pre_rewrite_docs", [])
                        post_docs = metrics.get("post_rewrite_docs", [])

                        pre_texts = [
                            d["content"] if isinstance(d, dict)
                            else str(d)
                            for d in pre_docs
                        ]

                        post_texts = [
                            d["content"] if isinstance(d, dict)
                            else str(d)
                            for d in post_docs
                        ]

                        pre_recall = compute_gold_chunk_recall(
                            pre_texts, gold_chunks, embedder, threshold=0.30
                        )
                        post_recall = compute_gold_chunk_recall(
                            post_texts, gold_chunks, embedder, threshold=0.30
                        )
                        rewrite_gain = evaluate_rewrite_effectiveness(
                            pre_recall, post_recall
                        )

                        grounding_score = grounding / 100
                        followup_score = evaluate_followup_success(
                            answer_correctness=answer_score,
                            retrieval_recall=retrieval_recall,
                            grounding_score=grounding_score
                        )

                    if gold_answer == "NOT_PRESENT":
                        verdict = "PASS" if not hallucination_detected else "FAIL"
                    else:
                        if answer_score >= 0.55:
                            verdict = "PASS"
                        elif answer_score >= 0.30:
                            verdict = "PARTIAL"
                        else:
                            verdict = "FAIL"

                    if verdict == "PASS":       pass_count += 1
                    elif verdict == "PARTIAL":  partial_count += 1
                    else:                       fail_count += 1

                    verdict_icon = "✅" if verdict == "PASS" else "⚠️" if verdict == "PARTIAL" else "❌"
                    print(
                        f"[Evaluate] Q{q_id} {verdict_icon} {verdict} | "
                        f"answer_score={answer_score:.4f} | "
                        f"recall={retrieval_recall:.4f} | "
                        f"precision={retrieval_precision:.4f} | "
                        f"confidence={confidence:.1f}%"
                )

                    results.append({
                        "id":               q_id,
                        "question":         question,
                        "query_type":       query_type,
                        "verdict":          verdict,
                        "gold_answer": gold_answer,
                        "answer_score": round(answer_score, 4),
                        "retrieval_recall": round(retrieval_recall, 4),
                        "retrieval_precision": round(retrieval_precision, 4),
                        "followup_score": round(followup_score, 4) if followup_score is not None else None,
                        "followup_question": is_followup,
                        "rewrite_triggered": metrics.get("rewrite_triggered", False),
                        "depends_on": depends_on,
                        "hallucination_detected": hallucination_detected,
                        "rewrite_gain": round(rewrite_gain, 4),
                        "rewritten_query": metrics.get("rewritten_query", ""),
                        "query_rewrite": item.get("query_rewrite", ""),
                        "pass_numeric": (
                            1 if verdict == "PASS"
                            else 0.5 if verdict == "PARTIAL"
                            else 0
                        ),
                        "model_used": model_used,
                        "recall_at_k":      round(recall_k, 1),
                        "answer_grounding": round(grounding, 1),
                        "confidence":       round(confidence, 1),
                        "roberta_test":     is_roberta,
                        "actual_answer": answer,
                        "retrieved_docs": retrieved_texts,
                        "tests":            item.get("tests", ""),
                        "conversation_id":  None,
                        "turn":             None,
                    })

            total = len(results)
            if total == 0:
                return jsonify({"error": "All questions were skipped or failed"}), 400

            pass_rate = round(pass_count / total * 100, 1)
            query_type_summary = aggregate_query_type_metrics(results)
            run_summary = {
                "run_id":          run_desc,
                "date":            time.strftime("%Y-%m-%d %H:%M:%S"),
                "pdf":             os.path.basename(pdf_path),
                "total_questions": total,
                "pass":            pass_count,
                "query_type_summary": query_type_summary,
                "partial":         partial_count,
                "fail":            fail_count,
                "pass_rate":       pass_rate,
                "results":         results
            }

            results_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "evaluation",
    "results",
    "final_grounding_results.json"
)
            try:
                if os.path.exists(results_path):
                    with open(results_path, "r") as f:
                        existing = json.load(f)
                else:
                    existing = {"sessions": []}

                session_found = False
                for session in existing["sessions"]:
                    if session["name"] == run_desc:
                        run_number = len(session["runs"]) + 1
                        run_summary["run_number"] = run_number
                        session["runs"].append(run_summary)
                        session["num_runs"] = len(session["runs"])
                        session_found = True
                        break

                if not session_found:
                    run_summary["run_number"] = 1
                    existing["sessions"].append({
                        "name": run_desc,
                        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "pdf": os.path.basename(pdf_path),
                        "num_runs": 1,
                        "runs": [run_summary]
                    })

                with open(results_path, "w") as f:
                    json.dump(existing, f, indent=2)

                print(f"[Evaluate] Results saved to final_grounding_results.json")
            except Exception as save_err:
                print(f"[Evaluate] Warning: could not save results: {save_err}")

            print(f"[Evaluate] ✅ Done | Pass={pass_count} Partial={partial_count} "
                  f"Fail={fail_count} | Pass rate={pass_rate}%")

            return jsonify({
                "run_id":          run_desc,
                "total_questions": total,
                "pass":            pass_count,
                "partial":         partial_count,
                "fail":            fail_count,
                "pass_rate":       f"{pass_rate}%",
                "results":         results
            })

        except Exception as fatal_err:
            print(f"[Evaluate] ❌ Fatal error: {fatal_err}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Fatal evaluation error: {str(fatal_err)}"}), 500

    @app.route("/ask", methods=["POST"])
    def ask():
        data = request.json or {}

        question = data.get("question", "").strip()
        request_id = data.get("request_id", "")

        evaluation_run = data.get("evaluation_run")
        evaluation_question_id = data.get(
            "evaluation_question_id",
            ""
        )

        if not question:
            return jsonify({"error": "Missing question"}), 400

        backend_log_file = None
        original_stdout = None

        if evaluation_run is not None:
            backend_log_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "evaluation",
                    "results",
                    "backend_logs"
                )
            )

            os.makedirs(backend_log_dir, exist_ok=True)

            backend_log_path = os.path.join(
                backend_log_dir,
                f"rag_backend_40_{evaluation_run}.log"
            )

            backend_log_file = open(
                backend_log_path,
                "a",
                encoding="utf-8",
                buffering=1
            )

            original_stdout = sys.stdout

            backend_log_file.write(
                "\n"
                + "=" * 70
                + "\n"
                + f"QUESTION {evaluation_question_id} | "
                f"RUN {evaluation_run}\n"
                + f"TIME: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                + f"QUESTION: {question}\n"
                + "=" * 70
                + "\n"
            )

            sys.stdout = BackendTee(
                original_stdout,
                backend_log_file
            )

        original_question = question

        emit_event(request_id, "workflow_start", "⚙️ Searching 3GPP knowledge base...")

        state: DocState = {
            "pdf_path":          "",
            "question":          question,
            "original_question": original_question,
            "extracted_text":    "",
            "chunks":            all_chunks,
            "summary_chunks":    [],
            "faiss_index":       faiss_index,
            "answer":            "",
            "metrics":           {},
            "doc_type":          "general",
            "query_type":        "",
            "retry_count":       0,
            "start_time":        time.time(),
            "page_count":        0,
            "char_count":        0,
            "request_id":        request_id,
        }
        try:
            request_start = time.time()

            state = node_qa(state)
            state = node_validate(state)

            m = state["metrics"]

            total_e2e = round(
                time.time() - request_start,
                2
            )

            m["e2e_latency_sec"] = total_e2e

            real_ttft = _ttft_tracker.pop(
                request_id,
                None
            )

            m["ttft_sec"] = (
                real_ttft
                if real_ttft is not None
                else total_e2e
            )

            emit_event(
                request_id,
                "done",
                "✅ Complete!"
            )

            cleanup_queue(request_id)

            return jsonify({
                "answer": state["answer"],
                "metrics": m,
                "rewritten_query": None
            })

        except Exception as e:

            return jsonify({
                "answer": f"Error: {str(e)}",
                "metrics": {}
            })

        finally:

            if original_stdout is not None:
                sys.stdout = original_stdout

            if backend_log_file is not None:

                backend_log_file.write(
                    "\n"
                    + "-" * 70
                    + "\n"
                    + f"END {evaluation_question_id} | "
                        f"RUN {evaluation_run}\n"
                    + "-" * 70
                    + "\n"
                )

                backend_log_file.close()

    @app.route("/stream", methods=["GET"])
    def stream():
        request_id = request.args.get("request_id", "")

        def event_generator():
            import tempfile, time
            event_file = os.path.join(tempfile.gettempdir(), f"docmind_{request_id}.jsonl")
            timeout    = time.time() + 300
            seen_lines = 0
            while time.time() < timeout:
                if os.path.exists(event_file):
                    with open(event_file, "r") as f:
                        lines = f.readlines()
                    for line in lines[seen_lines:]:
                        seen_lines += 1
                        yield f"data: {line.strip()}\n\n"
                        if json.loads(line.strip()).get("type") == "done":
                            return
                else:
                    yield f"data: {json.dumps({'type': 'heartbeat', 'message': ''})}\n\n"
                time.sleep(0.3)

        return Response(event_generator(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})






















