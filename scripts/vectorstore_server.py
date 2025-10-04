#!/usr/bin/env python3
"""
Synndicate Vector Store Server
Copyright (c) 2025 Himokai. All Rights Reserved.

Minimal HTTP vector store service for Synndicate RAG retriever.

A lightweight, authenticated vector database with persistence support.
Designed for development and small-scale production deployments.

Endpoints:
- GET /health
    -> {"status": "ok", "count": "N"}

- POST /vectors/add
    { "items": [{ "id": str, "embedding": [float], "document": str, "metadata": dict }] }
    -> {"added": N, "total": N}

- POST /vectors/query
    { "embedding": [float], "n_results": int }
    -> { "results": [{ "id": str, "score": float, "document": str, "metadata": dict }] }

- POST /vectors/delete
    { "ids": [str] }
    -> {"removed": N, "total": N}

Authentication:
- Optional X-API-Key header authentication
- Set SYN_VECTORSTORE_API_KEY environment variable to enable

Persistence:
- Set SYN_VECTORSTORE_PERSIST_PATH to enable JSON file persistence
- Automatically loads on startup, saves on modifications

Run locally:
    # Generate API key (optional)
    export SYN_VECTORSTORE_API_KEY="$(openssl rand -hex 32)"
    export SYN_VECTORSTORE_PERSIST_PATH="$HOME/.synndicate/vectorstore.json"

    # Start server
    uvicorn scripts.vectorstore_server:app --host 0.0.0.0 --port 8080

Platform support:
- Arch Linux (primary development platform)
- Ubuntu/Debian (CI/CD and production)
- macOS and Windows (via Docker recommended)

For production, consider using a proper vector database like:
- Chroma, Pinecone, Weaviate, or Qdrant
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


class VectorItem(BaseModel):
    id: str
    embedding: list[float]
    document: str | None = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AddVectorsRequest(BaseModel):
    items: list[VectorItem]


class QueryRequest(BaseModel):
    embedding: list[float]
    n_results: int = 10


class QueryResult(BaseModel):
    id: str
    score: float
    document: str | None = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    results: list[QueryResult]


class DeleteRequest(BaseModel):
    ids: list[str]


app = FastAPI(title="Synndicate Vector Store", version="0.2.0")

# In-memory store: id -> (embedding np.ndarray, document, metadata)
_STORE: dict[str, tuple[np.ndarray, str, dict[str, Any]]] = {}
_DIM: int | None = None
_PERSIST_PATH: Path | None = None
_API_KEY: str | None = os.getenv("SYN_VECTORSTORE_API_KEY")


def _ensure_auth(x_api_key: str | None) -> None:
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _load_persisted() -> None:
    global _STORE, _DIM
    if not _PERSIST_PATH or not _PERSIST_PATH.exists():
        return
    try:
        raw = json.loads(_PERSIST_PATH.read_text())
        st: dict[str, tuple[np.ndarray, str, dict[str, Any]]] = {}
        for vid, rec in raw.get("items", {}).items():
            emb = np.asarray(rec[0], dtype=np.float32)
            st[vid] = (emb, rec[1], rec[2])
        _STORE = st
        _DIM = raw.get("dim")
    except Exception:
        # best effort
        pass


def _save_persisted() -> None:
    if not _PERSIST_PATH:
        return
    try:
        payload = {
            "dim": _DIM,
            "items": {vid: (emb.tolist(), doc, meta) for vid, (emb, doc, meta) in _STORE.items()},
        }
        _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PERSIST_PATH.write_text(json.dumps(payload))
    except Exception:
        # best effort
        pass


@app.on_event("startup")
async def _startup():
    global _PERSIST_PATH
    path = os.getenv("SYN_VECTORSTORE_PERSIST_PATH")
    if path:
        _PERSIST_PATH = Path(path)
        _load_persisted()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "count": str(len(_STORE))}


@app.post("/vectors/add")
async def add_vectors(
    req: AddVectorsRequest, x_api_key: str | None = Header(default=None)
) -> dict[str, Any]:
    _ensure_auth(x_api_key)
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
    _save_persisted()
    return {"added": added, "total": len(_STORE)}


@app.post("/vectors/query", response_model=QueryResponse)
async def query_vectors(
    req: QueryRequest, x_api_key: str | None = Header(default=None)
) -> QueryResponse:
    _ensure_auth(x_api_key)
    if not _STORE:
        return QueryResponse(results=[])

    q = np.asarray(req.embedding, dtype=np.float32)
    if q.ndim != 1 or (_DIM is not None and q.shape[0] != _DIM):
        return QueryResponse(results=[])

    # Compute similarities
    scored: list[tuple[str, float]] = []
    for vid, (emb, _doc, _meta) in _STORE.items():
        sim = _cosine_similarity(q, emb)
        scored.append((vid, sim))

    # Top-k
    k = max(1, min(req.n_results, len(scored)))
    topk = sorted(scored, key=lambda t: t[1], reverse=True)[:k]

    results: list[QueryResult] = []
    for vid, score in topk:
        emb, doc, meta = _STORE[vid]
        results.append(QueryResult(id=vid, score=float(score), document=doc, metadata=meta))

    return QueryResponse(results=results)


@app.post("/vectors/delete")
async def delete_vectors(
    req: DeleteRequest, x_api_key: str | None = Header(default=None)
) -> dict[str, Any]:
    _ensure_auth(x_api_key)
    removed = 0
    for vid in req.ids:
        if vid in _STORE:
            del _STORE[vid]
            removed += 1
    _save_persisted()
    return {"removed": removed, "total": len(_STORE)}
