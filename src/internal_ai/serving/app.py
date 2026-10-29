"""Model serving HTTP API (FR-17 to FR-19)."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from internal_ai.context import Platform
from internal_ai.data.tokenizer import Tokenizer
from internal_ai.domain import ModelVersion
from internal_ai.serving.service import (
    ModelRouter,
    Principal,
    RateLimiter,
    RetrievalService,
    load_principals,
    log_request,
    out_of_service,
    served_model,
)
from internal_ai.training.embedder import TorchEmbedder

MEMORISATION_SAMPLE = 200


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    dataset_id: str | None = None


class RollbackRequest(BaseModel):
    model_id: str
    reason: str = "rollback requested"


class CitationHit(BaseModel):
    path: str
    line: int
    text: str = ""


class AnswerRequest(BaseModel):
    question: str
    hits: list[CitationHit]


def create_serving_app(
    root: Path | None = None,
    rate_limit_per_minute: int = 60,
    principals: dict[str, Principal] | None = None,
) -> FastAPI:
    key_table = principals if principals is not None else load_principals()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        platform = Platform.create(root=root)
        app.state.platform = platform
        app.state.router = ModelRouter(platform.store, platform.audit)
        app.state.retrieval = RetrievalService(platform.store, platform.audit)
        app.state.embedders: dict[str, TorchEmbedder] = {}
        try:
            yield
        finally:
            platform.close()

    app = FastAPI(
        title="Internal-AI serving",
        version="0.1.0",
        description="Authenticated inference and permission-aware retrieval.",
        lifespan=lifespan,
    )
    limiter = RateLimiter(rate_limit_per_minute)

    def get_platform() -> Platform:
        return app.state.platform

    def authenticate(x_internal_key: str | None = Header(default=None)) -> Principal:
        if not key_table:
            raise HTTPException(status_code=503, detail="no serving keys configured")
        principal = key_table.get(x_internal_key or "")
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

    def in_service_model() -> ModelVersion:
        """The model this request is answered by, or a refusal that says why not."""
        platform = get_platform()
        model = served_model(platform.store, app.state.router)
        if model is not None:
            return model
        reasons = [
            reason
            for item in platform.store.list_models()
            if (reason := out_of_service(platform.store, item)) is not None
        ]
        if reasons:
            raise HTTPException(
                status_code=503,
                detail={"error": "no model is in service", "reasons": reasons},
            )
        raise HTTPException(status_code=404, detail="no embedding model is registered")

    def memorisation_sources(platform: Platform) -> list[str]:
        """Text an answer must not recite: the served model's corpus, sampled evenly."""
        model = served_model(platform.store, app.state.router)
        if model is None or not model.dataset_ids:
            return []
        rows = platform.store.dataset_records(model.dataset_ids[0])
        step = max(len(rows) // MEMORISATION_SAMPLE, 1)
        return [row["payload"] for row in rows[::step][:MEMORISATION_SAMPLE]]

    def embedder_for(model_path: str) -> TorchEmbedder:
        embedders: dict[str, TorchEmbedder] = app.state.embedders
        if model_path not in embedders:
            platform = get_platform()
            tokenizer_path = Path(model_path).with_suffix(".tokenizer.json")
            if not tokenizer_path.exists():
                candidates = sorted(platform.settings.models_dir.glob("tokenizer-*.json"))
                if not candidates:
                    raise HTTPException(status_code=500, detail="no tokenizer available")
                tokenizer_path = candidates[-1]
            embedders[model_path] = TorchEmbedder(Path(model_path), Tokenizer.load(tokenizer_path))
        return embedders[model_path]

    @app.get("/health")
    def health() -> dict[str, Any]:
        platform = get_platform()
        return {"status": "ok", "models": len(platform.store.list_models())}

    @app.get("/v1/models")
    def list_models(principal: Principal = Depends(authenticate)) -> list[dict[str, Any]]:
        platform = get_platform()
        router: ModelRouter = app.state.router
        models = platform.store.list_models()
        served: dict[str, str | None] = {}
        for name in {model.name for model in models}:
            resolved = router.resolve(name)
            served[name] = resolved.id if resolved else None
        return [
            {
                "id": model.id,
                "name": model.name,
                "version": model.version,
                "status": model.status,
                "active": served.get(model.name) == model.id,
            }
            for model in models
        ]

    @app.post("/v1/search")
    def search(
        request: SearchRequest, principal: Principal = Depends(authenticate)
    ) -> dict[str, Any]:
        started = time.monotonic()
        platform = get_platform()
        retrieval: RetrievalService = app.state.retrieval
        model = in_service_model()
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
                platform.audit, principal, "/v1/search", 500, (time.monotonic() - started) * 1000
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        latency = (time.monotonic() - started) * 1000
        log_request(
            platform.audit,
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
        platform = get_platform()
        model = in_service_model()
        vectors = embedder_for(model.path).embed([request.query])
        latency = (time.monotonic() - started) * 1000
        log_request(platform.audit, principal, "/v1/embed", 200, latency, {"model_id": model.id})
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

    @app.post("/v1/answer")
    def answer(
        request: AnswerRequest, principal: Principal = Depends(authenticate)
    ) -> dict[str, Any]:
        """Generate from the on-prem runner. A mixed citation is replaced by the hits."""
        import os

        from internal_ai.llm import LocalRunnerAdapter
        from internal_ai.training.alpha.answer import cited_answer
        from internal_ai.training.alpha.citations import AskHit

        if not os.environ.get("INTERNAL_AI_LOCAL_RUNNER", "").strip():
            raise HTTPException(status_code=503, detail="no on-prem runner is configured")
        if not request.question.strip() or not request.hits:
            raise HTTPException(
                status_code=400, detail="a question and at least one hit are required"
            )
        started = time.monotonic()
        hits = [AskHit(item.path, item.line) for item in request.hits]
        excerpts = [item.text for item in request.hits]
        try:
            result = cited_answer(
                LocalRunnerAdapter(),
                request.question,
                hits,
                excerpts,
                training_texts=memorisation_sources(get_platform()),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        latency = (time.monotonic() - started) * 1000
        log_request(
            get_platform().audit,
            principal,
            "/v1/answer",
            200,
            latency,
            {"hits": len(hits), "citation_passed": result.passed, "model": result.model},
        )
        return {
            "text": result.text,
            "citation_passed": result.passed,
            "model": result.model,
            "mismatches": list(result.mismatches),
            "latency_ms": round(latency, 3),
        }

    return app
