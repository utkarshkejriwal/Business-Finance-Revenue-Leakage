"""Hybrid retrieval (dense + BM25), metadata filtering, reranking and
cross-reference expansion over the file-based vector store."""

import re

import numpy as np
from fastembed.rerank.cross_encoder import TextCrossEncoder
from rank_bm25 import BM25Okapi

from vector_store import embed_query, load_store

RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-12-v2"

DENSE_TOP_K = 30
BM25_TOP_K = 30
RRF_K = 60
RERANK_CANDIDATES = 24
FINAL_TOP_K = 6
FINAL_TOP_K_BROAD = 14   # questions that compare or list across merchants
MAX_EXPANSIONS = 6
# Cross-encoder logits: clearly relevant chunks score > 0, unrelated ones ~ -11.
MIN_CHUNK_SCORE = -6.0

# A schedule-wide reference ("charged at the Schedule A rate card") points at the rate table.
SCHEDULE_REF_TARGET = "A.1"

TOKEN_RE = re.compile(r"[a-z0-9§.]+")
MERCHANT_ID_RE = re.compile(r"\bM0?(\d{1,2})\b", re.IGNORECASE)
BROAD_QUERY_RE = re.compile(
    r"\b(which|all|every|each|compare|comparison|list|across|merchants|highest|lowest|most|least)\b",
    re.IGNORECASE,
)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class ContractRetriever:
    def __init__(self):
        self.chunks, self.vectors = load_store()
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.reranker = TextCrossEncoder(RERANK_MODEL)
        self.by_section = {}
        for index, chunk in enumerate(self.chunks):
            self.by_section.setdefault(chunk["section_id"], []).append(index)
        self.merchants = {
            c["merchant_id"]: c["merchant_name"]
            for c in self.chunks if c["merchant_id"]
        }

    # -- query understanding -------------------------------------------------

    def detect_merchants(self, query):
        """Merchant ids mentioned by id (M01, m1) or by name (full or first word)."""
        found = set()
        for match in MERCHANT_ID_RE.finditer(query):
            merchant_id = f"M{int(match.group(1)):02d}"
            if merchant_id in self.merchants:
                found.add(merchant_id)
        lowered = query.lower()
        for merchant_id, name in self.merchants.items():
            first_word = name.split()[0].lower()
            if name.lower() in lowered or re.search(rf"\b{re.escape(first_word)}\b", lowered):
                found.add(merchant_id)
        return sorted(found)

    def build_filters(self, query):
        merchants = self.detect_merchants(query)
        # Merchant agreements build on Schedule A, so it stays searchable.
        parents = merchants + ["Schedule A"] if merchants else None
        broad = not merchants and bool(BROAD_QUERY_RE.search(query))
        if len(merchants) > 1:
            broad = True
        return {"merchant_ids": merchants, "parent_keys": parents, "broad": broad}

    # -- retrieval -----------------------------------------------------------

    def allowed_indices(self, parent_keys):
        if not parent_keys:
            return np.arange(len(self.chunks))
        return np.array([i for i, c in enumerate(self.chunks) if c["parent_key"] in parent_keys])

    def hybrid_search(self, query, parent_keys=None):
        """Reciprocal-rank fusion of dense and BM25 rankings within the filter."""
        allowed = self.allowed_indices(parent_keys)

        dense_scores = self.vectors[allowed] @ embed_query(query)
        dense_rank = allowed[np.argsort(-dense_scores)[:DENSE_TOP_K]]

        bm25_scores = self.bm25.get_scores(tokenize(query))[allowed]
        bm25_rank = allowed[np.argsort(-bm25_scores)[:BM25_TOP_K]]

        fused = {}
        for ranking in (dense_rank, bm25_rank):
            for position, index in enumerate(ranking):
                fused[int(index)] = fused.get(int(index), 0.0) + 1.0 / (RRF_K + position + 1)
        ordered = sorted(fused, key=fused.get, reverse=True)
        return [(index, fused[index]) for index in ordered[:RERANK_CANDIDATES]]

    def rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        indices = [index for index, _ in candidates]
        scores = list(self.reranker.rerank(query, [self.chunks[i]["text"] for i in indices]))
        ranked = sorted(zip(indices, scores), key=lambda pair: pair[1], reverse=True)
        return [(index, float(score)) for index, score in ranked[:top_k]]

    def expand_references(self, selected, parent_keys):
        """Pull in sections the selected chunks cite (e.g. Schedule A, A.4) and
        annexures that amend a selected section (e.g. M09 Annexure 1 -> M09-§2)."""
        chosen = {index for index, _ in selected}
        chosen_sections = {self.chunks[i]["section_id"] for i in chosen}
        extra = []

        for index in list(chosen):
            for ref in self.chunks[index]["references"]:
                target = SCHEDULE_REF_TARGET if ref == "Schedule A" else ref
                for ref_index in self.by_section.get(target, []):
                    if ref_index not in chosen:
                        chosen.add(ref_index)
                        extra.append(ref_index)

        for index, chunk in enumerate(self.chunks):
            if index in chosen or chunk["section_kind"] != "annexure":
                continue
            if parent_keys and chunk["parent_key"] not in parent_keys:
                continue
            if chosen_sections.intersection(chunk["references"]):
                chosen.add(index)
                extra.append(index)

        return extra[:MAX_EXPANSIONS]

    def retrieve(self, query, filters=None):
        filters = filters or self.build_filters(query)
        top_k = FINAL_TOP_K_BROAD if filters["broad"] else FINAL_TOP_K
        candidates = self.hybrid_search(query, filters["parent_keys"])
        reranked = [(index, score) for index, score in self.rerank(query, candidates, top_k)
                    if score >= MIN_CHUNK_SCORE]
        expanded = self.expand_references(reranked, filters["parent_keys"])

        results = [{**self.chunks[i], "rerank_score": score, "via": "rerank"}
                   for i, score in reranked]
        results += [{**self.chunks[i], "rerank_score": None, "via": "reference"}
                    for i in expanded]
        return results
