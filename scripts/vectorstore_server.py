#!/usr/bin/env python3
"""
Minimal HTTP vector store service for Synndicate RAG retriever.

Endpoints:
- POST /vectors/add
    { "items": [{ "id": str, "embedding": [float], "document": str, "metadata": dict }] }
    -> 200 OK

- POST /vectors/query
    { "embedding": [float], "n_results": int }
    -> { "results": [{ "id": str, "score": float, "document": str, "metadata": dict }] }

Run locally:
    uvicorn scripts.vectorstore_server:app --host 0.0.0.0 --port 8080

This is intended for development/reference. For production, use a proper
vector database and add persistence/auth.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field


class VectorItem(BaseModel):
    id: str
    embedding: List[float]
    document: Optional[str] = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AddVectorsRequest(BaseModel):
    items: List[VectorItem]


class QueryRequest(BaseModel):
    embedding: List[float]
    n_results: int = 10


class QueryResult(BaseModel):
    id: str
    score: float
    document: Optional[str] = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    results: List[QueryResult]


app = FastAPI(title="Synndicate Vector Store", version="0.1.0")

# In-memory store: id -> (embedding np.ndarray, document, metadata)
_STORE: Dict[str, tuple[np.ndarray, str, Dict[str, Any]]] = {}
_DIM: Optional[int] = None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "count": str(len(_STORE))}


@app.post("/vectors/add")
async def add_vectors(req: AddVectorsRequest) -> Dict[str, Any]:
    global _DIM
    added = 0
    for item in req.items:
        emb = np.asarray(item.embedding, dtype=np.float32)
        if emb.ndim != 1:
            continue
        if _DIM is None:
            _DIM = emb.shape[0]
        if emb.shape[0] != _DIM:
            # Skip inconsistent dimension entries
            continue
        _STORE[item.id] = (emb, item.document or "", item.metadata or {})
        added += 1
    return {"added": added, "total": len(_STORE)}


@app.post("/vectors/query", response_model=QueryResponse)
async def query_vectors(req: QueryRequest) -> QueryResponse:
    if not _STORE:
        return QueryResponse(results=[])

    q = np.asarray(req.embedding, dtype=np.float32)
    if q.ndim != 1 or (_DIM is not None and q.shape[0] != _DIM):
        return QueryResponse(results=[])

    # Compute similarities
    scored: List[tuple[str, float]] = []
    for vid, (emb, _doc, _meta) in _STORE.items():
        sim = _cosine_similarity(q, emb)
        scored.append((vid, sim))

    # Top-k
    k = max(1, min(req.n_results, len(scored)))
    topk = sorted(scored, key=lambda t: t[1], reverse=True)[:k]

    results: List[QueryResult] = []
    for vid, score in topk:
        emb, doc, meta = _STORE[vid]
        results.append(QueryResult(id=vid, score=float(score), document=doc, metadata=meta))

    return QueryResponse(results=results)
