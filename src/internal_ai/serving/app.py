"""Model serving HTTP API (FR-17 to FR-19)."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from internal_ai.context import Platform
from internal_ai.data.tokenizer import Tokenizer
from internal_ai.domain import ModelVersion
from internal_ai.ids import new_id
from internal_ai.serving.guardrails import (
    GUARDRAIL_REFUSE,
    configured_install_policy,
    guardrail_policy,
    validate_output_guardrails,
)
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


class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    top_k: int = 5


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
    max_concurrent = int(os.environ.get("INTERNAL_AI_MAX_CONCURRENT", "5"))
    app.state.concurrency_semaphore = threading.Semaphore(max_concurrent)

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
                status_code=409,
                detail={"error": "no model has a passing frozen report", "reasons": reasons},
            )
        raise HTTPException(status_code=404, detail="no embedding model is registered")

    def served_by(name: str) -> ModelVersion:
        """The model this request is answered by. A named one is never swapped.

        A client that names a model meant that model. Falling back to whichever
        model happens to be in service would answer a question about the wrong
        artifact and look successful doing it.
        """
        platform = get_platform()
        if not name:
            return in_service_model()
        resolved: ModelVersion | None = app.state.router.resolve(name)
        if resolved is not None:
            return resolved
        known = platform.store.list_models(name=name)
        if not known:
            raise HTTPException(status_code=404, detail=f"unknown model: {name}")
        reasons = [
            reason for item in known if (reason := out_of_service(platform.store, item)) is not None
        ]
        raise HTTPException(
            status_code=409,
            detail={
                "error": f"no promoted model named {name} is in service",
                "reasons": reasons or [f"{item.id} is {item.status}" for item in known],
            },
        )

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
            embedders[model_path] = TorchEmbedder(
                Path(model_path),
                Tokenizer.load(tokenizer_path),
                name=f"torch-embedder:{Path(model_path).name}",
            )
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
        """Generate on this machine, or from the runner if one is configured."""
        from internal_ai.serving.trace import FAILURE_CITATION, FAILURE_GUARDRAIL, record_trace
        from internal_ai.training.alpha.answer import cited_answer
        from internal_ai.training.alpha.citations import AskHit
        from internal_ai.training.alpha.local import LocalGenerationUnavailable, answer_adapter

        if not request.question.strip() or not request.hits:
            raise HTTPException(
                status_code=400, detail="a question and at least one hit are required"
            )
        # Enforce load ceiling: acquire concurrency semaphore
        with app.state.concurrency_semaphore:
            started = time.monotonic()
            model = in_service_model()
            hits = [AskHit(item.path, item.line) for item in request.hits]
            excerpts = [item.text for item in request.hits]
            try:
                generator = answer_adapter(get_platform(), model)
            except PermissionError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            try:
                result = cited_answer(
                    generator,
                    request.question,
                    hits,
                    excerpts,
                    training_texts=memorisation_sources(get_platform()),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(
                    status_code=503, detail=f"the local model is not on disk: {exc}"
                ) from exc
            except LocalGenerationUnavailable as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            latency = (time.monotonic() - started) * 1000

            # Enforce latency target from model config if set
            latency_target_ms = model.config.get("latency_target_ms")
            if latency_target_ms is not None and latency > latency_target_ms:
                raise HTTPException(
                    status_code=408,
                    detail={
                        "error": "latency target exceeded",
                        "latency_ms": round(latency, 3),
                        "latency_target_ms": latency_target_ms,
                    },
                )

            guardrail_violations = validate_output_guardrails(result.text)
            try:
                policy = guardrail_policy(
                    model.config, default=configured_install_policy(get_platform().store)
                )
            except ValueError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            details = {
                "hits": len(hits),
                "citation_passed": result.passed,
                "model": result.model,
                "guardrail_policy": policy,
                "guardrail_violations": guardrail_violations,
            }
            citations = [{"path": item.path, "line": item.line} for item in request.hits]
            if policy == GUARDRAIL_REFUSE and guardrail_violations:
                log_request(
                    get_platform().audit,
                    principal,
                    "/v1/answer",
                    422,
                    latency,
                    details,
                    question=request.question,
                )
                record_trace(
                    get_platform().store,
                    get_platform().audit,
                    principal,
                    endpoint="/v1/answer",
                    model_id=model.id,
                    question=request.question,
                    answer=result.text,
                    citations=citations,
                    latency_ms=latency,
                    status_code=422,
                    failure=FAILURE_GUARDRAIL,
                )
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "answer withheld by guardrail policy",
                        "violations": guardrail_violations,
                    },
                )
            log_request(
                get_platform().audit,
                principal,
                "/v1/answer",
                200,
                latency,
                details,
                question=request.question,
            )
            record_trace(
                get_platform().store,
                get_platform().audit,
                principal,
                endpoint="/v1/answer",
                model_id=model.id,
                question=request.question,
                answer=result.text,
                citations=citations,
                latency_ms=latency,
                status_code=200,
                failure="" if result.passed else FAILURE_CITATION,
            )
            return {
                "text": result.text,
                "citation_passed": result.passed,
                "model": result.model,
                "mismatches": list(result.mismatches),
                "latency_ms": round(latency, 3),
                "guardrail_violations": guardrail_violations,
            }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest, principal: Principal = Depends(authenticate)) -> Any:
        """An OpenAI-shaped answer from the same guarded path as `/v1/answer`.

        Retrieval runs under the caller's own ACLs, and the finished answer goes
        through the same citation and guardrail checks. A retrieved record has no
        line of its own, so it is cited as `<record key>:0` — the whole document —
        and an answer that puts any other line number on it is a mismatch, as it
        is everywhere else.

        A guardrail refusal is decided before the first token. A citation failure
        cannot be known until the answer exists, so a streamed reply carries the
        text and then closes with an error frame instead of `[DONE]`, and the
        audit event records the refusal. A client that walks away mid-stream still
        leaves one, logged as 499.
        """
        from internal_ai.serving.trace import FAILURE_CITATION, FAILURE_GUARDRAIL, record_trace
        from internal_ai.training.alpha.answer import cited_answer
        from internal_ai.training.alpha.citations import AskHit
        from internal_ai.training.alpha.local import LocalGenerationUnavailable, answer_adapter

        question = next(
            (
                message.content.strip()
                for message in reversed(request.messages)
                if message.role == "user" and message.content.strip()
            ),
            "",
        )
        if not question:
            raise HTTPException(status_code=400, detail="the last user message is empty")
        if not 1 <= request.top_k <= 20:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

        platform = get_platform()
        model = served_by(request.model)
        dataset_id = model.dataset_ids[0] if model.dataset_ids else ""
        if not dataset_id:
            raise HTTPException(status_code=400, detail="no dataset bound to the model")
        retrieval: RetrievalService = app.state.retrieval
        rows = retrieval.corpus(dataset_id)
        started = time.monotonic()
        with app.state.concurrency_semaphore:
            hits = retrieval.search(
                embedder_for(model.path), dataset_id, question, principal, top_k=request.top_k
            )
            if not hits:
                readable = any(principal.can_read(row["acl"], row["sensitivity"]) for row in rows)
                status = 403 if rows and not readable else 400
                reason = (
                    "no document in this dataset may be read with this key"
                    if status == 403
                    else "nothing in this dataset matched the question"
                )
                log_request(
                    platform.audit,
                    principal,
                    "/v1/chat/completions",
                    status,
                    (time.monotonic() - started) * 1000,
                    {"hits": 0, "model_id": model.id},
                    question=question,
                )
                raise HTTPException(status_code=status, detail=reason)
            citations = [AskHit(hit.document_id, 0) for hit in hits]
            excerpts = [hit.snippet for hit in hits]
            try:
                generator = answer_adapter(platform, model)
            except (PermissionError, ValueError) as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            try:
                result = cited_answer(
                    generator,
                    question,
                    citations,
                    excerpts,
                    training_texts=memorisation_sources(platform),
                )
            except FileNotFoundError as exc:
                raise HTTPException(
                    status_code=503, detail=f"the local model is not on disk: {exc}"
                ) from exc
            except LocalGenerationUnavailable as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

        violations = validate_output_guardrails(result.text)
        try:
            policy = guardrail_policy(
                model.config, default=configured_install_policy(platform.store)
            )
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        cited = [{"path": hit.document_id, "line": 0} for hit in hits]
        latency = (time.monotonic() - started) * 1000

        def event_details(outcome: str) -> dict[str, Any]:
            return {
                "hits": len(hits),
                "citation_passed": result.passed,
                "model": result.model,
                "guardrail_policy": policy,
                "guardrail_violations": violations,
                "stream": request.stream,
                "model_id": model.id,
                "stream_outcome": outcome,
                "completed": outcome == "completed",
            }

        if policy == GUARDRAIL_REFUSE and violations:
            log_request(
                platform.audit,
                principal,
                "/v1/chat/completions",
                422,
                latency,
                event_details("refused"),
                question=question,
            )
            record_trace(
                platform.store,
                platform.audit,
                principal,
                endpoint="/v1/chat/completions",
                model_id=model.id,
                question=question,
                answer=result.text,
                citations=cited,
                latency_ms=latency,
                status_code=422,
                failure=FAILURE_GUARDRAIL,
            )
            raise HTTPException(
                status_code=422,
                detail={"error": "answer withheld by guardrail policy", "violations": violations},
            )

        completion_id = f"chatcmpl-{new_id('chat')}"
        created = int(time.time())
        completion: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model or model.name or model.id,
        }

        if not request.stream:
            log_request(
                platform.audit,
                principal,
                "/v1/chat/completions",
                200,
                latency,
                event_details("completed"),
                question=question,
            )
            record_trace(
                platform.store,
                platform.audit,
                principal,
                endpoint="/v1/chat/completions",
                model_id=model.id,
                question=question,
                answer=result.text,
                citations=cited,
                latency_ms=latency,
                failure="" if result.passed else FAILURE_CITATION,
            )
            prompt_words = len(question.split()) + sum(len(hit.snippet.split()) for hit in hits)
            answer_words = len(result.text.split())
            return {
                **completion,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_words,
                    "completion_tokens": answer_words,
                    "total_tokens": prompt_words + answer_words,
                },
                "usage_note": "whitespace counts; this install reports no billing tokens",
                "citation_passed": result.passed,
                "mismatches": list(result.mismatches),
                "guardrail_violations": violations,
            }

        def frame(payload: dict[str, Any]) -> str:
            return f"data: {json.dumps(payload, sort_keys=True)}\n\n"

        def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
            return frame(
                {
                    **completion,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
            )

        def stream() -> Iterator[str]:
            outcome = "abandoned"
            try:
                yield chunk({"role": "assistant"})
                for match in re.finditer(r"\S+\s*", result.text):
                    yield chunk({"content": match.group(0)})
                if not result.passed:
                    yield frame(
                        {
                            "error": {
                                "message": "answer withheld: its citations do not hold",
                                "type": "citation_failed",
                                "mismatches": list(result.mismatches),
                            }
                        }
                    )
                    outcome = "refused"
                    return
                yield chunk({}, "stop")
                yield "data: [DONE]\n\n"
                outcome = "completed"
            finally:
                # A refusal is this server's decision, not the client's exit, so
                # the two must not share a status: 499 means the reader left.
                elapsed = (time.monotonic() - started) * 1000
                status = 499 if outcome == "abandoned" else 200
                log_request(
                    platform.audit,
                    principal,
                    "/v1/chat/completions",
                    status,
                    elapsed,
                    event_details(outcome),
                    question=question,
                )
                record_trace(
                    platform.store,
                    platform.audit,
                    principal,
                    endpoint="/v1/chat/completions",
                    model_id=model.id,
                    question=question,
                    answer=result.text,
                    citations=cited,
                    latency_ms=elapsed,
                    status_code=status,
                    failure="" if outcome == "completed" else FAILURE_CITATION,
                )

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return app
