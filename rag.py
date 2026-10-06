"""RAG pipeline over the bank policy document (Requirement 2).

* Document loading: python-docx / docx XML fallback.
* Chunking: structure-aware (splits on numbered Persian headings).
* Embeddings: best-of-class *Persian* model ``csebuetnlp/mMTEB_large_fa``
  (sentence-transformers, local & free) or OpenAI ``text-embedding-3-*``
  (multilingual) — selectable via EMBEDDING_PROVIDER.
* Vector store + retrieval: LangChain InMemoryVectorStore with MMRetrieval
  (vector similarity + BM25 keyword hybrid reranking) → strong Persian recall.
"""

from __future__ import annotations

import logging
import os
import re
import zipfile

import config

log = logging.getLogger("banking_agent.rag")


# ---------------------------------------------------------------------------
# Loading the .docx
# ---------------------------------------------------------------------------
def load_docx_paragraphs(path: str) -> list[str]:
    """Return non-empty paragraph texts from a .docx file."""
    try:
        import docx  # python-docx

        d = docx.Document(path)
        return [p.text.strip() for p in d.paragraphs if p.text.strip()]
    except Exception:
        pass
    # Fallback: raw XML extraction (works without python-docx)
    z = zipfile.ZipFile(path)
    xml = z.read("word/document.xml").decode("utf-8")
    paras = re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)
    out = []
    for p in paras:
        texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S)
        t = "".join(texts).strip()
        if t:
            out.append(t)
    return out


_SECTION_RE = re.compile(r"^\s*(\d+)\.\s+\S")   # top-level: "1. انواع حساب‌ها"
_SUBSEC_RE = re.compile(r"^\s*(\d+\.\d+)\.?\s+\S")  # subsection: "3.4 وام مسکن"


def chunk_policy(paragraphs: list[str]) -> list[str]:
    """Structure-aware chunking keyed on the document's numbered sections.

    Each top-level section (e.g. «۳. شرایط دریافت وام») becomes one or more
    chunks; sub-section headings («3.4 وام مسکن») are preserved inside the
    chunk text so both dense retrieval and BM25 can key on them.
    """
    chunks: list[str] = []
    current: list[str] = []
    limit = config.CHUNK_SIZE

    def flush():
        nonlocal current
        if current:
            text = "\n".join(current).strip()
            if text:
                chunks.append(text)
            current = []

    for para in paragraphs:
        is_top = bool(_SECTION_RE.match(para)) and len(para) < 90
        is_sub = bool(_SUBSEC_RE.match(para)) and len(para) < 90
        joined_len = len("\n".join(current))
        if is_top and joined_len > limit * 0.4:
            flush()
            current.append(f"### {para}")
        elif is_sub and joined_len > limit:
            flush()
            current.append(f"#### {para}")
        else:
            current.append(para if (is_top or is_sub) else para)
            if len("\n".join(current)) > limit + config.CHUNK_OVERLAP:
                # safety valve for very long unstructured runs
                flush()
    flush()
    return chunks


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def build_embeddings():
    """Return a LangChain Embeddings object per configuration."""
    provider = config.EMBEDDING_PROVIDER
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        log.info("Using OpenAI embeddings: %s", config.OPENAI_EMBEDDING_MODEL)
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY or None,
            base_url=config.OPENAI_BASE_URL,
        )

    # default: dedicated Persian SOTA embedding, local & offline
    from langchain_community.embeddings import HuggingFaceEmbeddings

    log.info("Using Persian embeddings: %s", config.PERSIAN_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=config.PERSIAN_EMBEDDING_MODEL)


class MiniHashingEmbeddings:
    """Zero-dependency deterministic embeddings (character n-gram hashing).

    Used only as an emergency fallback so the agent still runs (with reduced
    retrieval quality) if neither sentence-transformers nor OpenAI is usable.
    Implements the LangChain `Embeddings` protocol duck-typed.
    """

    DIM = 512

    def _embed(self, text: str) -> list[float]:
        import hashlib
        import math

        v = [0.0] * self.DIM
        norm_text = re.sub(r"\s+", " ", (text or "").lower().strip())
        tokens = ["$" + w + "$" for w in norm_text.split(" ")]
        grams = []
        for tok in tokens:
            grams.append(tok)
            for n in (2, 3):
                grams.extend(tok[i : i + n] for i in range(len(tok) - n + 1))
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
            idx = h % self.DIM
            sign = 1.0 if (h >> 63) & 1 else -1.0
            v[idx] += sign
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class PolicyRetriever:
    def __init__(self, docs_path: str = config.POLICIES_FILE):
        self.docs_path = docs_path
        self.chunks: list[str] = []
        self.retriever = None
        self.embedder_name = ""
        self._build()

    def _build(self):
        from langchain_core.documents import Document
        from langchain_core.vectorstores import InMemoryVectorStore

        paragraphs = load_docx_paragraphs(self.docs_path)
        self.chunks = chunk_policy(paragraphs)
        docs = [Document(page_text=c, metadata={"source": os.path.basename(self.docs_path)})
                for c in self.chunks]

        try:
            embeddings = build_embeddings()
            self.embedder_name = type(embeddings).__name__
        except Exception as exc:  # no model available offline etc.
            log.warning("Primary embeddings unavailable (%s); using hashing fallback.", exc)
            embeddings = MiniHashingEmbeddings()
            self.embedder_name = "MiniHashingEmbeddings(fallback)"

        store = InMemoryVectorStore(embeddings)
        store.add_documents(docs)

        # Hybrid retrieval: dense vectors + BM25 sparse keyword matching.
        try:
            from langchain_community.retrievers import BM25Retriever
            from langchain.retrievers import EnsembleRetriever

            bm25 = BM25Retriever.from_documents(docs, k=config.RAG_TOP_K)
            self.retriever = EnsembleRetriever(
                retrievers=[store.as_retriever(search_kwargs={"k": config.RAG_TOP_K}), bm25],
                weights=[0.6, 0.4],
            )
            self.mode = "hybrid(vector+BM25)"
        except Exception as exc:
            log.warning("BM25 ensemble unavailable (%s); vector-only retrieval.", exc)
            self.retriever = store.as_retriever(search_kwargs={"k": config.RAG_TOP_K})
            self.mode = "vector"

    def retrieve(self, query: str, k: int | None = None) -> list[str]:
        try:
            docs = self.retriever.invoke(query)
        except Exception as exc:
            log.error("Retrieval failed: %s", exc)
            return []
        if k:
            docs = docs[:k]
        seen, out = set(), []
        for d in docs:
            t = d.page_text.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def context_for(self, query: str, k: int | None = None) -> str:
        passages = self.retrieve(query, k=k)
        if not passages:
            return "(مطالبی از قوانین پیدا نشد)"
        return "\n\n---\n\n".join(passages)


_singleton: PolicyRetriever | None = None


def get_retriever() -> PolicyRetriever:
    global _singleton
    if _singleton is None:
        _singleton = PolicyRetriever()
    return _singleton
