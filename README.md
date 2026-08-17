# DCO MIND — 3GPP Document Intelligence & RAG System

DCO MIND is a document-grounded Retrieval-Augmented Generation (RAG) system designed for question answering over technical 3GPP specifications. It combines clause-aware document processing, hybrid retrieval, cross-encoder reranking, LLaMA 3.2 generation, citation validation, grounding checks, and hallucination detection to provide reliable answers from technical standards.

The current knowledge base contains:

- 3GPP TS 38.300
- 3GPP TS 38.331

---

## Key Features

- Clause-aware 3GPP document processing
- Hybrid semantic, keyword, and exact-match retrieval
- FAISS vector search
- Sentence-Transformer embeddings
- Cross-encoder reranking
- Exact-match protection for technical terms, clauses, and numerical values
- Adaptive retrieval and query rewriting
- LLaMA 3.2 through Ollama
- Grounded answer generation with source citations
- Citation and grounding validation
- Hallucination / abstention detection
- Conversational memory and follow-up question handling
- RAPTOR-style document summarization
- Flask REST API
- Streaming responses using Server-Sent Events
- React frontend
- Automated evaluation and retrieval/grounding metrics

---

## Architecture

    User Question
          |
          v
    Flask /ask API
          |
          v
    Query Processing
    & Classification
          |
          v
    Hybrid Retrieval
      |       |       |
      |       |       +--> TF-IDF Retrieval
      |       +----------> Exact-Match Retrieval
      +------------------> FAISS Semantic Retrieval
          |
          v
    Cross-Encoder Reranking
          |
          v
    Exact-Match Protection
          |
          v
    Grounded Context
          |
          +----------------------+
          |                      |
          v                      v
      Direct QA            ReAct Fallback
          |                      |
          +----------+-----------+
                     |
                     v
              LLaMA 3.2
              via Ollama
                     |
                     v
        Citation / Grounding Guard
                     |
                     v
        Hallucination Protection
                     |
                     v
              Final Answer
                     |
                     v
                Metrics

---

## Knowledge Base

The current fixed corpus consists of:

- 3GPP TS 38.300
- 3GPP TS 38.331

The extracted corpus files are located under:

    dco_mind/datasets/3gpp/
    ├── 38_300_extracted.txt
    └── 38_331_extracted.txt

The two specifications are processed independently and then combined into a single searchable corpus while retaining the source specification and clause metadata.

Retrieved chunks retain information such as:

    [Source: TS 38.331]
    [3GPP Clause 5.2.2.3.1 | Page 45]

This allows generated answers to be traced back to the relevant 3GPP specification and clause.

---

## RAG Pipeline

### 1. Document Ingestion

PDF documents are processed using PyMuPDF.

When normal PDF extraction does not provide sufficient text, the system can use Tesseract OCR through pytesseract.

Document pages are processed in parallel and extraction results are cached to avoid unnecessary repeated processing.

### 2. Clause-Aware Chunking

The 3GPP documents are processed using clause-aware chunking rather than relying only on generic fixed-size splitting.

The pipeline identifies clause structure, verifies clause titles against the document structure, preserves page information, and attaches source and clause metadata to chunks.

The current RAG configuration uses:

    RAG_CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 200

Summary processing uses a larger chunk size:

    SUMMARY_CHUNK_SIZE = 8000

### 3. Embeddings

The system uses:

    sentence-transformers/all-MiniLM-L6-v2

to generate embeddings for the document chunks.

The embeddings are indexed using FAISS.

### 4. Hybrid Retrieval

The retrieval pipeline combines multiple retrieval strategies:

- FAISS semantic retrieval
- Exact-match retrieval
- TF-IDF keyword retrieval
- MMR fallback

This is particularly useful for technical questions containing clause numbers, timer names, technical identifiers, and numerical values.

### 5. Cross-Encoder Reranking

Retrieved chunks are reranked using:

    cross-encoder/ms-marco-MiniLM-L-6-v2

The reranker improves the ordering of retrieved evidence before it is passed to the language model.

### 6. Exact-Match Protection

After reranking, important exact-match chunks can be restored if they were removed by the reranker.

This helps preserve evidence containing highly specific technical terms, identifiers, clauses, or numerical values.

### 7. Grounded Generation

The retrieved context is passed to LLaMA 3.2 through Ollama.

The model is instructed to:

- Answer only from the retrieved context
- Avoid unsupported assumptions
- Avoid external knowledge
- Answer the user's question directly
- Use retrieved evidence as the factual source
- Provide source information where applicable
- Abstain when the required information is not present

### 8. Validation

The generated response is evaluated using retrieval and answer-quality signals including:

- Recall@K
- Retrieval precision
- Answer grounding
- Semantic similarity
- Token-level F1
- Confidence
- Citation validation
- Hallucination detection

---

## Query Types

The system supports different question types, including:

- Factual QA
- Verification / Yes-No QA
- Multipart QA
- Full-document summarization

The system also supports follow-up questions through conversational context and query rewriting.

---

## Verification Questions

Verification questions such as:

    Does TS 38.331 define a timer named T999?

are handled as direct Yes/No questions when the retrieved evidence supports such a decision.

The generated response should answer the user's question directly rather than simply returning an unrelated document excerpt.

For unsupported questions, the system can return an abstention response such as:

    This information is not present in the document.

---

## Hallucination Protection

DCO MIND uses several mechanisms to reduce unsupported generation:

- Retrieval-grounded prompting
- Explicit no-outside-knowledge instructions
- Citation validation
- Retrieval recall evaluation
- Answer grounding evaluation
- Confidence scoring
- Exact-match protection
- Abstention detection
- Post-generation validation

The goal is to prevent the LLM from presenting unsupported information as though it came from the 3GPP specifications.

---

## Summarization

The system also contains a hierarchical summarization pipeline.

The summarization process combines TF-IDF-based extractive summarization with LLaMA-based reduction and merging.

    Document
       |
       v
    Summary Chunks
       |
       v
    TF-IDF Sentence Selection
       |
       v
    Section Summaries
       |
       v
    LLaMA Reduction
       |
       v
    Final Summary

The configured RAPTOR batch size is:

    RAPTOR_BATCH_CHARS = 15000

---

## Conversational Memory

Session-based conversational memory is supported for follow-up questions.

Previous conversation can help clarify user intent, but it is not treated as independent factual evidence.

Factual claims must still be supported by the retrieved 3GPP context.

---

## Query Rewriting

When additional retrieval is useful, the system can generate alternative search queries while preserving the meaning of the original question.

Rewrite information can be recorded for evaluation, including:

    rewrite_triggered
    rewritten_query
    pre_rewrite_docs
    post_rewrite_docs

This allows retrieval performance before and after rewriting to be compared.

---

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, Flask, Flask-CORS |
| Workflow | LangChain, LangGraph |
| LLM | LLaMA 3.2 |
| LLM Runtime | Ollama |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Reranking | Cross-Encoder |
| Keyword Retrieval | scikit-learn TF-IDF |
| PDF Processing | PyMuPDF |
| OCR | Tesseract, pytesseract |
| Image Processing | Pillow |
| Frontend | React |

---

## API

The backend runs locally at:

    http://127.0.0.1:5000

### GET /health

Returns backend and runtime status.

    GET /health

### POST /ask

Main question-answering endpoint.

Example request:

    {
      "question": "Which RRC state is the UE in when no RRC connection is established?"
    }

Example response structure:

    {
      "answer": "...",
      "metrics": {
        "confidence_score": "...",
        "answer_grounding": "...",
        "recall_at_k": "...",
        "e2e_latency_sec": "..."
      },
      "rewritten_query": null
    }

The exact metric values depend on the query and retrieved context.

### GET /stream

Streaming endpoint:

    GET /stream?request_id=<request_id>

The endpoint provides Server-Sent Events for streamed responses and workflow events.

### POST /evaluate

The project also contains an evaluation endpoint for executing evaluation datasets through the backend.

---

## Installation

### 1. Create a virtual environment

Windows:

    python -m venv venv
    venv\Scripts\activate

Linux/macOS:

    python3 -m venv venv
    source venv/bin/activate

### 2. Install dependencies

    pip install -r requirements.txt

The project requires:

    torch==2.4.0
    transformers==4.41.2
    sentence-transformers==2.7.0
    langchain==0.2.0
    langgraph==0.0.55
    faiss-cpu==1.7.4
    flask==3.0.3
    flask-cors==4.0.1
    pymupdf==1.24.1
    pytesseract==0.3.13
    Pillow==10.3.0
    numpy==1.26.4
    typing-extensions>=4.9.0
    langchain-community==0.2.0
    langchain-huggingface==0.0.3
    scikit-learn
    ollama
    requests

### 3. Install Ollama

Install Ollama separately and make sure it is running.

Pull the configured model:

    ollama pull llama3.2

Verify:

    ollama list

### 4. Install Tesseract OCR

Tesseract is required for the OCR fallback used during PDF processing.

If Tesseract is installed in a non-default location, update the configured Tesseract path in:

    dco_mind/config/settings.py

---

## Running the Backend

From the project root:

    python -m dco_mind.app

The backend will start at:

    http://127.0.0.1:5000

On startup, the application loads the fixed 3GPP corpus and builds the searchable FAISS representation.

---

## Latest Evaluation

The latest evaluation contains:

    40 questions

The hallucination subset contains:

    20 hallucination tests

### Overall Evaluation Results

| Metric | Result |
|---|---:|
| Total Questions | 40 |
| Passed | 36 / 40 |
| Failed | 4 / 40 |
| Accuracy | 90.0% |
| Hallucination Tests | 20 |
| Automatically Detected | 18 / 20 |
| Average Recall@K | 99.0% |
| Average Grounding | 79.7% |
| Average Confidence | 76.2% |
| Evaluation Time | 5187.3s |

### Hallucination Handling

The automated evaluator reported:

    18 / 20 detected
    90.0% hallucination detection rate

One of the reported failures, Q035, was manually reviewed.

The Q035 question is:

    Does TS 38.331 define a timer named T999?

The generated response was:

    No, TS 38.331 does not define a timer named T999.

This response correctly states that T999 is not defined in TS 38.331 and does not hallucinate a T999 timer.

The automated evaluator nevertheless classified Q035 as a failure. This was identified as an evaluation/benchmark-labeling mismatch rather than an incorrect technical answer.

The raw automated evaluation result is retained for transparency.

---
## Evaluation Metrics

### Recall@K

Measures whether relevant supporting information is retrieved in the retrieved context.

### Retrieval Precision

Measures the relevance of retrieved chunks.

### Answer Grounding

Measures how strongly the generated response is supported by the retrieved context.

### Confidence

Combines retrieval and answer relevance signals.

### Answer Correctness

Combines semantic similarity and token-level F1 for evaluating generated answers.

### Hallucination Detection

Measures whether the system correctly abstains when the requested information is not supported by the document.

---

## Screenshots

### Main Question Answering Interface

The main DCO MIND interface showing a 3GPP question, retrieved context, generated answer, and system metrics.

![DCO MIND Question Answering Interface](demo_output_images/sample_output_images/Screenshot%20%282199%29.png)

### Grounded Response for Unsupported Information

Example showing the system responding that information is not present in the 3GPP knowledge base instead of generating an unsupported answer.

![DCO MIND Grounded Response](demo_output_images/sample_output_images/Screenshot%202026-08-15%20220534.png)

### Retrieval and Generation Metrics

Detailed response metrics including retrieval, grounding, confidence, recall, and latency.

![DCO MIND RAG Metrics](demo_output_images/sample_output_images/Screenshot%202026-08-15%20220553.png)

### Evaluation Summary

The 40-question evaluation summary showing overall accuracy, hallucination-test results, recall, grounding, and confidence.

![DCO MIND Evaluation Summary](demo_output_images/summary_metrics/Screenshot%202026-08-17%20222549.png)

## Project Structure

    Mavenir-3GPP-RAG2/
    |
    +-- dco_mind/
    |   |
    |   +-- app.py
    |   |
    |   +-- config/
    |   |   +-- settings.py
    |   |
    |   +-- core/
    |   |   +-- engine.py
    |   |   +-- state.py
    |   |
    |   +-- cognition/
    |   |   +-- memory.py
    |   |   +-- query_brain.py
    |   |
    |   +-- knowledge/
    |   |   +-- corpus_loader.py
    |   |   +-- ingestion.py
    |   |
    |   +-- models/
    |   |   +-- embeddings.py
    |   |   +-- llm.py
    |   |
    |   +-- retrieval/
    |   |   +-- adaptive_search.py
    |   |   +-- reranker.py
    |   |
    |   +-- reasoning/
    |   |   +-- context_builder.py
    |   |
    |   +-- generation/
    |   |   +-- response_generator.py
    |   |
    |   +-- evaluation/
    |   |   +-- metrics.py
    |   |   +-- stability_runner.py
    |   |   +-- results/
    |   |
    |   +-- events/
    |   |   +-- events.py
    |   |
    |   +-- datasets/
    |       +-- 3gpp/
    |           +-- 38_300_extracted.txt
    |           +-- 38_331_extracted.txt
    |
    +-- frontend/
    |
    +-- requirements.txt
    |
    +-- README.md

---

## Limitations

- The current knowledge base is limited to TS 38.300 and TS 38.331.
- Local LLaMA 3.2 inference can result in relatively high response latency compared with hosted inference.
- Complex questions that require evidence across multiple clauses or specifications may still be challenging for the retrieval pipeline.
- Automated evaluation of verification questions can be sensitive to answer-format and labeling criteria, as demonstrated by the Q035 evaluator mismatch.

---

## Future Improvements

- Expand the knowledge base to additional 3GPP specifications.
- Improve retrieval for complex multi-clause questions.
- Optimize local LLM inference latency.
- Expand evaluation coverage across additional 3GPP procedures.
- Improve automated evaluation of nuanced verification questions.
- Extend document-level benchmarking.

---

## Conclusion

DCO MIND provides an end-to-end document-grounded RAG pipeline for technical 3GPP question answering.

The system combines clause-aware ingestion, hybrid retrieval, reranking, grounded LLM generation, citation validation, hallucination protection, conversational context, and automated evaluation.

The latest evaluation consists of 40 questions and achieved:

    36 / 40 passed
    90.0% accuracy

The hallucination subset contains 20 test questions. The automated evaluator detected:

    18 / 20
    90.0% hallucination detection rate

One reported failure, Q035, was manually reviewed. The generated response correctly stated that T999 is not defined in TS 38.331, so this case was identified as an evaluation/benchmark mismatch rather than a hallucinated technical answer.

The raw automated evaluation output is retained for transparency, while the manually reviewed Q035 case is documented as an evaluator-side mismatch.