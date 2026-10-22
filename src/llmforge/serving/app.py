"""Model serving HTTP API (FR-17 to FR-19)."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from llmforge.context import Forge
from llmforge.data.tokenizer import Tokenizer
from llmforge.serving.service import (
    ModelRouter,
    Principal,
    RateLimiter,
    RetrievalService,
    load_principals,
    log_request,
)
from llmforge.training.embedder import TorchEmbedder


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    dataset_id: str | None = None


class RollbackRequest(BaseModel):
    model_id: str
    reason: str = "rollback requested"


def create_serving_app(
    root: Path | None = None,
    rate_limit_per_minute: int = 60,
    principals: dict[str, Principal] | None = None,
) -> FastAPI:
    key_table = principals if principals is not None else load_principals()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        forge = Forge.create(root=root)
        app.state.forge = forge
        app.state.router = ModelRouter(forge.store, forge.audit)
        app.state.retrieval = RetrievalService(forge.store, forge.audit)
        app.state.embedders: dict[str, TorchEmbedder] = {}
        try:
            yield
        finally:
            forge.close()

    app = FastAPI(
        title="LLM-Forge serving",
        version="0.1.0",
        description="Authenticated inference and permission-aware retrieval.",
        lifespan=lifespan,
    )
    limiter = RateLimiter(rate_limit_per_minute)

    def get_forge() -> Forge:
        return app.state.forge

    def authenticate(x_forge_key: str | None = Header(default=None)) -> Principal:
        if not key_table:
            raise HTTPException(status_code=503, detail="no serving keys configured")
        principal = key_table.get(x_forge_key or "")
        if principal is None:
            raise HTTPException(status_code=401, detail="invalid api key")
        if not limiter.allow(principal.user_id):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate limit exceeded",
                    "limit_per_minute": limiter.limit,
                    "remaining": limiter.remaining(principal.user_id),
                },
            )
        return principal

    def embedder_for(model_path: str) -> TorchEmbedder:
        embedders: dict[str, TorchEmbedder] = app.state.embedders
        if model_path not in embedders:
            forge = get_forge()
            tokenizer_path = Path(model_path).with_suffix(".tokenizer.json")
            if not tokenizer_path.exists():
                candidates = sorted(forge.settings.models_dir.glob("tokenizer-*.json"))
                if not candidates:
                    raise HTTPException(status_code=500, detail="no tokenizer available")
                tokenizer_path = candidates[-1]
            embedders[model_path] = TorchEmbedder(Path(model_path), Tokenizer.load(tokenizer_path))
        return embedders[model_path]

    @app.get("/health")
    def health() -> dict[str, Any]:
        forge = get_forge()
        return {"status": "ok", "models": len(forge.store.list_models())}

    @app.get("/v1/models")
    def list_models(principal: Principal = Depends(authenticate)) -> list[dict[str, Any]]:
        forge = get_forge()
        router: ModelRouter = app.state.router
        return [
            {
                "id": model.id,
                "name": model.name,
                "version": model.version,
                "status": model.status,
                "active": router.active.get(model.name) == model.id,
            }
            for model in forge.store.list_models()
        ]

    @app.post("/v1/search")
    def search(
        request: SearchRequest, principal: Principal = Depends(authenticate)
    ) -> dict[str, Any]:
        started = time.monotonic()
        forge = get_forge()
        router: ModelRouter = app.state.router
        retrieval: RetrievalService = app.state.retrieval
        model = router.resolve("embedding-v1")
        if model is None:
            raise HTTPException(status_code=404, detail="no embedding model is registered")
        dataset_id = request.dataset_id
        if dataset_id is None:
            dataset_id = model.dataset_ids[0] if model.dataset_ids else None
        if dataset_id is None:
            raise HTTPException(status_code=400, detail="no dataset bound to the model")
        try:
            hits = retrieval.search(
                embedder_for(model.path),
                dataset_id,
                request.query,
                principal,
                top_k=request.top_k,
            )
        except HTTPException:
            raise
        except Exception as exc:
            log_request(
                forge.audit, principal, "/v1/search", 500, (time.monotonic() - started) * 1000
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        latency = (time.monotonic() - started) * 1000
        log_request(
            forge.audit,
            principal,
            "/v1/search",
            200,
            latency,
            {"hits": len(hits), "model_id": model.id},
        )
        return {
            "model_id": model.id,
            "hits": [
                {
                    "document_id": hit.document_id,
                    "score": hit.score,
                    "snippet": hit.snippet,
                    "source_id": hit.source_id,
                }
                for hit in hits
            ],
            "latency_ms": round(latency, 3),
        }

    @app.post("/v1/embed")
    def embed(
        request: SearchRequest, principal: Principal = Depends(authenticate)
    ) -> dict[str, Any]:
        started = time.monotonic()
        forge = get_forge()
        router: ModelRouter = app.state.router
        model = router.resolve("embedding-v1")
        if model is None:
            raise HTTPException(status_code=404, detail="no embedding model is registered")
        vectors = embedder_for(model.path).embed([request.query])
        latency = (time.monotonic() - started) * 1000
        log_request(forge.audit, principal, "/v1/embed", 200, latency, {"model_id": model.id})
        return {
            "model_id": model.id,
            "embedding": [round(float(value), 6) for value in vectors[0]],
            "latency_ms": round(latency, 3),
        }

    @app.post("/v1/models/{name}/rollback")
    def rollback(
        name: str, request: RollbackRequest, principal: Principal = Depends(authenticate)
    ) -> dict[str, Any]:
        router: ModelRouter = app.state.router
        if "platform-admins" not in principal.groups:
            raise HTTPException(status_code=403, detail="rollback requires platform-admins")
        try:
            model = router.rollback(name, request.model_id, actor=principal.user_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"model_id": model.id, "name": name, "active": True}

    return app
