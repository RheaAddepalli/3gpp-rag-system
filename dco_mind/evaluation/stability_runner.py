

import requests
import json
import time
import os
import sys
import re
import datetime


# ============================================================
# CONFIG
# ============================================================

FLASK_URL = "http://127.0.0.1:5000/ask"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_FILE = os.path.join(
    BASE_DIR,
    "datasets",
    "rag_eval_dataset_40_grounded.json"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "evaluation",
    "results"
)

RESULT_FILE_PREFIX = "rag_results_40_"

LOG_DIR = RESULTS_DIR

BACKEND_LOG_DIR = os.path.join(
    RESULTS_DIR,
    "backend_logs"
)

REQUEST_TIMEOUT = 7200

ANSWER_MATCH_THRESHOLD = 0.6


NOT_PRESENT_PHRASES = [
    "not_present",
    "not present",
    "not found",
    "not specified",
    "not mentioned",
    "not stated",
    "not defined",
    "not covered",
    "not available in",
    "no information",
    "no relevant information",
    "no mention",
    "no such",
    "cannot find",
    "can't find",
    "could not find",
    "couldn't find",
    "unable to find",
    "does not specify",
    "does not contain",
    "does not mention",
    "does not provide",
    "doesn't specify",
    "doesn't contain",
    "doesn't mention",
    "outside the scope",
    "out of scope",
    "not supported by the provided",
    "not answerable",
    "unsupported",
    "unanswerable",
    "no data",
    "no evidence",
]

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "with",
    "does", "do", "did", "this", "that", "these", "those", "it", "its",
    "as", "by", "from", "yes", "no",
}


# ============================================================
# LOGGING
# ============================================================

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass


os.makedirs(LOG_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, "run_logs_40.txt")
_log_file = open(log_path, "a", encoding="utf-8")

_log_file.write(
    "\n"
    + "=" * 80
    + "\n"
    + f"RUN START : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    + "=" * 80
    + "\n"
)

sys.stdout = Tee(sys.stdout, _log_file)
sys.stderr = Tee(sys.stderr, _log_file)


# ============================================================
# DATASET
# ============================================================

def load_dataset():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}"
        )

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError(
            "Expected the evaluation dataset to be a JSON list."
        )

    print(f"[Dataset] Loaded {len(dataset)} questions")

    return dataset


def validate_dataset(dataset):
    if len(dataset) != 40:
        raise ValueError(
            f"Expected exactly 40 questions, found {len(dataset)}."
        )

    hallucination_count = sum(
        1 for q in dataset if bool(q.get("hallucination_test")) is True
    )
    answerable_count = sum(
        1 for q in dataset if bool(q.get("hallucination_test")) is False
    )

    print(f"[Dataset] Answerable (hallucination_test=false) : {answerable_count}")
    print(f"[Dataset] Hallucination tests (hallucination_test=true) : {hallucination_count}")

    if answerable_count != 20 or hallucination_count != 20:
        raise ValueError(
            "Dataset split is not 20 answerable / 20 hallucination-test "
            f"questions (got {answerable_count} / {hallucination_count})."
        )


# ============================================================
# BACKEND RESPONSE HELPERS
# ============================================================

def get_value(data, *keys, default=None):
    """
    Try several possible response-key names so the runner stays compatible
    with small backend response changes.
    """
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key in data:
            return data[key]
    return default


def normalize_answer(data):
    return get_value(
        data,
        "answer",
        "generated_answer",
        "final_answer",
        "actual_answer",
        "response",
        "result",
        default=""
    ) or ""


# ============================================================
# GRADING
# ============================================================

def normalize_text(s):
    s = str(s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def answer_indicates_not_present(answer_text):
    """
    True when the generated answer correctly signals that the requested
    information is not present / not found / unsupported.
    """
    text = normalize_text(answer_text)

    if not text:
        return False

    for phrase in NOT_PRESENT_PHRASES:
        if normalize_text(phrase) in text:
            return True

    return False


def compare_answer_to_gold(generated_answer, gold_answer):
    """
    Heuristic correctness check for answerable questions: PASS when a
    sufficient fraction of the gold answer's significant words appear in
    the generated answer.
    """
    gen_norm = normalize_text(generated_answer)
    gold_norm = normalize_text(gold_answer)

    if not gold_norm:
        return False

    gold_tokens = [
        t for t in gold_norm.split()
        if t not in STOPWORDS and len(t) > 2
    ]

    if not gold_tokens:
        return gold_norm in gen_norm

    matched = sum(1 for t in gold_tokens if t in gen_norm)
    ratio = matched / len(gold_tokens)

    return ratio >= ANSWER_MATCH_THRESHOLD


# ============================================================
# ONE QUESTION
# ============================================================

def run_question(question):
    question_id = question["id"]
    is_hallucination_test = bool(question.get("hallucination_test"))
    gold_answer = question.get("gold_answer", "")

    payload = {
    "question": question["question"],
    "session_id": None,
    "evaluation_run": RUN_NUMBER,
    "evaluation_question_id": question_id
}

    started = time.time()

    try:
        response = requests.post(
            FLASK_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        elapsed = round(time.time() - started, 2)

        if response.status_code != 200:
            return {
                "id": question_id,
                "question": question["question"],
                "hallucination_test": is_hallucination_test,
                "gold_answer": gold_answer,
                "generated_answer": "",
                "status": "http_error",
                "http_status": response.status_code,
                "hallucination": None,
                "result": "FAIL",
                "confidence": None,
                "answer_grounding": None,
                "recall_at_k": None,
                "latency_sec": elapsed,
                "source": extract_source(question),
            }

        try:
            data = response.json()
        except Exception as e:
            return {
                "id": question_id,
                "question": question["question"],
                "hallucination_test": is_hallucination_test,
                "gold_answer": gold_answer,
                "generated_answer": "",
                "status": "invalid_json",
                "error": str(e),
                "hallucination": None,
                "result": "FAIL",
                "confidence": None,
                "answer_grounding": None,
                "recall_at_k": None,
                "latency_sec": elapsed,
                "source": extract_source(question),
            }

        metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
        generated_answer = normalize_answer(data)

        if is_hallucination_test:
            correctly_abstained = answer_indicates_not_present(generated_answer)
            hallucinated = not correctly_abstained
            result = "PASS" if correctly_abstained else "FAIL"
        else:
            passed = compare_answer_to_gold(generated_answer, gold_answer)
            hallucinated = None
            result = "PASS" if passed else "FAIL"

        record = {
            "id": question_id,
            "question": question["question"],
            "hallucination_test": is_hallucination_test,
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
            "status": "ok",
            "hallucination": hallucinated,
            "result": result,
            "confidence": get_value(
                metrics, "confidence_score", "confidence", default=None
            ),
            "answer_grounding": get_value(
                metrics, "answer_grounding", "grounding_score", "grounding",
                default=None
            ),
            "recall_at_k": get_value(
                metrics, "recall_at_k", "retrieval_recall", "recall",
                default=None
            ),
            "latency_sec": elapsed,
            "source": extract_source(question),
        }

        return record

    except requests.exceptions.Timeout:
        elapsed = round(time.time() - started, 2)
        return {
            "id": question_id,
            "question": question["question"],
            "hallucination_test": is_hallucination_test,
            "gold_answer": gold_answer,
            "generated_answer": "",
            "status": "timeout",
            "hallucination": None,
            "result": "FAIL",
            "confidence": None,
            "answer_grounding": None,
            "recall_at_k": None,
            "latency_sec": elapsed,
            "source": extract_source(question),
        }

    except Exception as e:
        elapsed = round(time.time() - started, 2)
        return {
            "id": question_id,
            "question": question["question"],
            "hallucination_test": is_hallucination_test,
            "gold_answer": gold_answer,
            "generated_answer": "",
            "status": "error",
            "error": str(e),
            "hallucination": None,
            "result": "FAIL",
            "confidence": None,
            "answer_grounding": None,
            "recall_at_k": None,
            "latency_sec": elapsed,
            "source": extract_source(question),
        }


def extract_source(question):
    """
    Only returns source metadata that actually exists in the dataset entry.
    Hallucination-test questions never require this.
    """
    source = {}
    for key in ("pdf", "pdf_page", "document_page", "clause"):
        if question.get(key) not in (None, ""):
            source[key] = question.get(key)
    return source or None


# ============================================================
# PER-QUESTION PRINTING
# ============================================================

def print_question_result(record):
    print("\n" + "-" * 80)
    print(f"{record['id'].upper()} | {record['question']}\n")

    if record["hallucination_test"]:
        print("Type         : HALLUCINATION")
        print(
            f"Hallucination: "
            f"{'TRUE' if record.get('hallucination') else 'FALSE'}"
        )
        print(f"Answer       : {record.get('generated_answer', '')}")
        print(f"Result       : {record.get('result', 'FAIL')}")
    else:
        print("Type        : ANSWERABLE")
        source = record.get("source")
        if source:
            parts = []
            if source.get("pdf"):
                parts.append(str(source["pdf"]))
            if source.get("pdf_page") is not None:
                parts.append(f"Page {source['pdf_page']}")
            if source.get("clause"):
                parts.append(f"Clause {source['clause']}")
            if parts:
                print(f"Source      : {' | '.join(parts)}")
        print(f"Answer      : {record.get('generated_answer', '')}")
        print(f"Result      : {record.get('result', 'FAIL')}")

    print("-" * 80)


# ============================================================
# REPORT
# ============================================================

def build_report(records, total_elapsed):
    total = len(records)
    passed = sum(1 for r in records if r.get("result") == "PASS")
    failed = total - passed

    hallucination_records = [r for r in records if r.get("hallucination_test")]
    failed_to_detect = sum(
        1 for r in hallucination_records if r.get("hallucination") is True
    )

    passed_ids = [r["id"] for r in records if r.get("result") == "PASS"]
    failed_ids = [r["id"] for r in records if r.get("result") != "PASS"]
    hallucination_failed_ids = [
        r["id"] for r in hallucination_records if r.get("hallucination") is True
    ]

    def avg(field):
        values = [
            r[field] for r in records
            if isinstance(r.get(field), (int, float))
        ]
        return (sum(values) / len(values)) if values else None

    return {
        "total_questions": total,
        "passed": passed,
        "failed": failed,
        "accuracy_pct": round(passed / total * 100, 1) if total else 0.0,

        "hallucination_tests": len(hallucination_records),
        "failed_to_detect": failed_to_detect,
        "hallucination_rate_pct": round(
            failed_to_detect / len(hallucination_records) * 100, 1
        ) if hallucination_records else 0.0,

        "passed_ids": passed_ids,
        "failed_ids": failed_ids,
        "hallucination_failed_ids": hallucination_failed_ids,

        "average_recall_at_k": avg("recall_at_k"),
        "average_grounding": avg("answer_grounding"),
        "average_confidence": avg("confidence"),
        "average_latency_sec": avg("latency_sec"),

        "total_time_sec": round(total_elapsed, 1),
    }


def _pct(value):
    if value is None:
        return "N/A"
    v = value * 100 if value <= 1 else value
    return f"{v:.1f}%"


def print_full_report(report, records):
    print("\n")
    print("=" * 80)
    print("                    DCO MIND — 40 QUESTION EVALUATION")
    print("=" * 80)
    print()
    print(f"Pass        : {report['passed']} / {report['total_questions']} ({report['accuracy_pct']:.1f}%)")
    print(f"Fail        : {report['failed']} / {report['total_questions']} ({100 - report['accuracy_pct']:.1f}%)")
    print(f"Total       : {report['total_questions']}")
    print(f"Accuracy    : {report['accuracy_pct']:.1f}%")
    print(f"Time        : {report['total_time_sec']}s")
    print()
    print(f"Hallucination Tests : {report['hallucination_tests']}")
    print(
        f"Failed to Detect    : {report['failed_to_detect']} / "
        f"{report['hallucination_tests']} ({report['hallucination_rate_pct']:.1f}%)"
    )
    print(f"Hallucination Rate   : {report['hallucination_rate_pct']:.1f}%")
    print()
    print(f"Passed Questions : [{', '.join(report['passed_ids'])}]")
    print(f"Failed Questions : [{', '.join(report['failed_ids'])}]")
    print()
    print(f"Hallucination Failed Questions : [{', '.join(report['hallucination_failed_ids'])}]")
    print()
    print("=" * 80)
    print("                         QUESTION REPORT")
    print("=" * 80)

    for record in records:
        print_question_result(record)

    print("\n")
    print("=" * 80)
    print("                         SUMMARY METRICS")
    print("=" * 80)
    print()
    print(f"Pass                : {report['passed']} / {report['total_questions']} ({report['accuracy_pct']:.1f}%)")
    print(f"Fail                : {report['failed']} / {report['total_questions']} ({100 - report['accuracy_pct']:.1f}%)")
    print(f"Total               : {report['total_questions']}")
    print(f"Accuracy            : {report['accuracy_pct']:.1f}%")
    print(f"Time                : {report['total_time_sec']}s")
    print()
    print(f"Hallucination Tests : {report['hallucination_tests']}")
    print(
        f"Failed to Detect    : {report['failed_to_detect']} / "
        f"{report['hallucination_tests']} ({report['hallucination_rate_pct']:.1f}%)"
    )
    print(f"Hallucination Rate  : {report['hallucination_rate_pct']:.1f}%")
    print()
    print(f"Average Recall@K    : {_pct(report['average_recall_at_k'])}")
    print(f"Average Grounding   : {_pct(report['average_grounding'])}")
    print(f"Average Confidence  : {_pct(report['average_confidence'])}")
    print()
    print("=" * 80)



def get_next_run_number():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n = 1

    while True:
        result_path = os.path.join(
            RESULTS_DIR,
            f"{RESULT_FILE_PREFIX}{n}.json"
        )

        backend_log_path = os.path.join(
            BACKEND_LOG_DIR,
            f"rag_backend_40_{n}.log"
        )

        if not os.path.exists(result_path) and not os.path.exists(backend_log_path):
            return n

        n += 1

# ============================================================
# SAVE
# ============================================================

def get_next_result_path(run_number):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    return os.path.join(
        RESULTS_DIR,
        f"{RESULT_FILE_PREFIX}{run_number}.json"
    )


def save_results(report, records):
    result_path = get_next_result_path(RUN_NUMBER)

    output = {
        "metadata": {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": os.path.basename(DATASET_FILE),
            "num_questions": report["total_questions"],
        },
        "total_questions": report["total_questions"],
        "passed": report["passed"],
        "failed": report["failed"],
        "accuracy_pct": report["accuracy_pct"],

        "hallucination_tests": report["hallucination_tests"],
        "failed_to_detect": report["failed_to_detect"],
        "hallucination_rate_pct": report["hallucination_rate_pct"],
        "hallucination_failed_ids": report["hallucination_failed_ids"],

        "average_recall_at_k": report["average_recall_at_k"],
        "average_grounding": report["average_grounding"],
        "average_confidence": report["average_confidence"],
        "average_latency_sec": report["average_latency_sec"],
        "total_time_sec": report["total_time_sec"],

        "records": records,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to:\n{result_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    global RUN_NUMBER
    RUN_NUMBER = get_next_run_number()

    print(f"[Evaluation] Run number : {RUN_NUMBER}")
    print("\n=== DCO MIND — 40 QUESTION EVALUATION ===")

    dataset = load_dataset()
    validate_dataset(dataset)

    records = []
    run_started = time.time()

    for question in dataset:
        record = run_question(question)
        records.append(record)
        print_question_result(record)

    total_elapsed = time.time() - run_started

    report = build_report(records, total_elapsed)
    print_full_report(report, records)
    save_results(report, records)


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            _log_file.close()
        except Exception:
            pass











