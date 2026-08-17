import os

from dco_mind.knowledge.ingestion import clause_aware_chunk
from dco_mind.reasoning.context_builder import normalize_text
from dco_mind.models.embeddings import build_faiss_index

_CORPUS_CACHE = {}

SPEC_SOURCES = {
    "TS 38.300": r"dco_mind/datasets/3gpp/38_300_extracted.txt",
    "TS 38.331": r"dco_mind/datasets/3gpp/38_331_extracted.txt",
}


def load_fixed_corpus(force_reload: bool = False):
    """
    Extracts + clause-chunks both 3GPP specs ONCE, tags every chunk
    with its source spec (so identical clause numbers across specs
    don't collide), merges into a single chunk list, and builds one
    combined FAISS index.

    Cached at module level — safe to call this from multiple request
    handlers; only the first call (or a force_reload) does real work.

    Returns:
        all_chunks  — list[str], each chunk prefixed with its
                      source spec tag, e.g.
                      "[Source: TS 38.331]\n[3GPP Clause 5.2.2.3.1
                       | Page 45]\n5.2.2.3.1 Acquisition of MIB..."
        faiss_index — single FAISS index built over all_chunks
    """
    if not force_reload and "chunks" in _CORPUS_CACHE:
        print("[Corpus] ✅ Using cached fixed corpus")
        return _CORPUS_CACHE["chunks"], _CORPUS_CACHE["faiss_index"]

    all_chunks = []

    for source_label, path in SPEC_SOURCES.items():
        if not os.path.exists(path):
            print(f"[Corpus] ⚠️ Missing file for {source_label}: {path} — skipping")
            continue

        print(f"[Corpus] Loading {source_label} from {path}...")

        with open(path, encoding="utf-8") as f:
            text = f.read()

        rag_chunks = clause_aware_chunk(text)
        rag_chunks = [normalize_text(c) for c in rag_chunks]

        tagged_chunks = [
            f"[Source: {source_label}]\n{c}" for c in rag_chunks
        ]

        print(f"[Corpus] {source_label}: {len(tagged_chunks)} chunks")
        all_chunks.extend(tagged_chunks)

    if not all_chunks:
        raise RuntimeError(
            "[Corpus] ❌ No chunks loaded — check SPEC_SOURCES paths "
            "in corpus_loader.py against your actual dataset folder."
        )

    print(f"[Corpus] TOTAL merged chunks: {len(all_chunks)}")

    faiss_index = build_faiss_index(all_chunks, pdf_hash="fixed_3gpp_corpus")

    _CORPUS_CACHE["chunks"] = all_chunks
    _CORPUS_CACHE["faiss_index"] = faiss_index

    return all_chunks, faiss_index