"""Long-horizon agentic search over a code and document corpus.

One-shot retrieval answers "which document mentions X". A long-horizon agent
answers "where is X implemented, and what does it do", by planning a query,
reading the results, deciding whether the evidence is sufficient, and refining
the query until it is. This is the retrieval agent of PRD section 7.7.

The planner has two modes. With a model available it asks for the next query.
Without one it uses pseudo-relevance feedback: the terms that distinguish the
best documents become the next query, which is how classic iterative retrieval
works and it needs no external model.

Records are filtered by the caller's document permissions before anything is
read, so the agentic path enforces the same ACLs as the serving API. A
principal who can read nothing gets a denial, never the corpus.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from internal_ai.audit import AuditLog
from internal_ai.budget import BudgetManager
from internal_ai.domain import AuditOutcome
from internal_ai.evals.metrics import bm25_scores, cosine_similarity_matrix, rank_from_scores
from internal_ai.llm import LLMAdapter
from internal_ai.serving.service import Principal

WORD = re.compile(r"[a-z][a-z0-9_]{2,}")

STOPWORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "that",
        "with",
        "this",
        "from",
        "have",
        "has",
        "was",
        "were",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "her",
        "our",
        "out",
        "get",
        "his",
        "how",
        "man",
        "new",
        "now",
        "see",
        "two",
        "way",
        "who",
        "boy",
        "did",
        "its",
        "let",
        "put",
        "say",
        "she",
        "too",
        "use",
        "any",
        "may",
        "find",
        "where",
        "what",
        "when",
        "which",
        "does",
        "done",
        "code",
        "file",
        "files",
        "project",
        "implement",
        "implementation",
        "codebase",
        "repository",
        "source",
        "sources",
        "function",
        "class",
        "module",
    ]
)


@dataclass(slots=True)
class SearchHit:
    document_id: str
    score: float
    snippet: str
    source_id: str = ""
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "score": self.score,
            "snippet": self.snippet,
            "source_id": self.source_id,
            "path": self.path,
        }


@dataclass(slots=True)
class SearchStep:
    index: int
    query: str
    strategy: str
    hits: list[SearchHit] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    decision: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "query": self.query,
            "strategy": self.strategy,
            "hits": [hit.to_dict() for hit in self.hits],
            "evidence": self.evidence,
            "decision": self.decision,
            "note": self.note,
        }


@dataclass(slots=True)
class AgenticSearchResult:
    question: str
    answer: str
    steps: list[SearchStep] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    steps_used: int = 0
    cost_usd: float = 0.0
    sufficient_evidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "steps": [step.to_dict() for step in self.steps],
            "files": list(self.files),
            "citations": list(self.citations),
            "steps_used": self.steps_used,
            "cost_usd": self.cost_usd,
            "sufficient_evidence": self.sufficient_evidence,
        }


def tokenize(text: str) -> list[str]:
    return [word for word in WORD.findall(text.lower()) if word not in STOPWORDS]


def expand_query(question: str) -> str:
    terms = tokenize(question)
    if not terms:
        return question.strip()
    seen: list[str] = []
    for term in terms:
        if term not in seen:
            seen.append(term)
    return " ".join(seen)


def pseudo_relevance_terms(
    query_terms: list[str], documents: list[str], top_n: int = 8
) -> list[str]:
    """Terms that distinguish the best documents, used to expand the next query."""
    if not documents:
        return []
    document_frequency: Counter[str] = Counter()
    tokenised = [tokenize(document) for document in documents]
    for tokens in tokenised:
        document_frequency.update(set(tokens))
    total = len(documents)
    scores: Counter[str] = Counter()
    for tokens in tokenised:
        length = len(tokens) or 1
        counts = Counter(tokens)
        for term, count in counts.items():
            if term in query_terms or term in STOPWORDS:
                continue
            idf = math.log(
                1 + (total - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
            )
            scores[term] += idf * (count / length)
    return [term for term, _ in scores.most_common(top_n)]


def document_frequency(
    rows: list[dict[str, Any]], terms: list[str], sample_limit: int = 6000
) -> tuple[dict[str, float], int]:
    """Share of documents containing each term, estimated from a sample.

    A term that appears everywhere carries no information. Dropping those terms
    from the query is what keeps a search on "mixture of experts" from
    returning every file that mentions the word "work".
    """
    if not terms:
        return {}, 0
    patterns = {term: re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in terms}
    counts = dict.fromkeys(terms, 0)
    step = max(len(rows) // sample_limit, 1) if rows else 1
    sampled = rows[::step][:sample_limit]
    for row in sampled:
        payload = row.get("payload", "")
        for term, pattern in patterns.items():
            if pattern.search(payload):
                counts[term] += 1
    size = max(len(sampled), 1)
    return {term: count / size for term, count in counts.items()}, size


def rare_terms(rows: list[dict[str, Any]], terms: list[str], max_ratio: float = 0.02) -> list[str]:
    shares, _size = document_frequency(rows, terms)
    if not shares:
        return terms
    rare = [term for term, share in shares.items() if share <= max_ratio]
    if not rare:
        rare = [term for term, _ in sorted(shares.items(), key=lambda item: item[1])[:2]]
    return rare


def coordination_boost(
    rows: list[dict[str, Any]], terms: list[str], alpha: float = 0.6
) -> list[float]:
    """Reward documents that match several query terms.

    Plain BM25 lets a long file that repeats one common term outrank a short
    file that matches three rare ones. Term coordination fixes that, which is
    what makes a question like "mixture of experts" land on the router files
    rather than on every file that says "implemented".
    """
    if not terms or not rows:
        return [1.0] * len(rows)
    patterns = {term: re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in terms}
    boosts: list[float] = []
    for row in rows:
        payload = row.get("payload", "")
        matched = sum(1 for pattern in patterns.values() if pattern.search(payload))
        boosts.append(1.0 + alpha * max(matched - 1, 0))
    return boosts


class AgenticSearcher:
    """Runs an iterative retrieve, inspect, refine loop over a corpus."""

    def __init__(
        self,
        store: Any,
        audit: AuditLog,
        budget: BudgetManager,
        embedder: Any | None = None,
        llm: LLMAdapter | None = None,
    ) -> None:
        self.store = store
        self.audit = audit
        self.budget = budget
        self.embedder = embedder
        self.llm = llm
        self._last_evidence: list[dict[str, Any]] = []

    def _corpus(self, dataset_id: str) -> list[dict[str, Any]]:
        return self.store.dataset_records(dataset_id)

    def _retrieve(
        self,
        query: str,
        rows: list[dict[str, Any]],
        top_k: int,
    ) -> list[SearchHit]:
        texts = [row["payload"] for row in rows]
        doc_ids = [row["record_key"] for row in rows]
        if self.embedder is not None:
            query_vector = self.embedder.embed([query])[0]
            doc_vectors = self.embedder.embed(texts)
            scores = list(cosine_similarity_matrix(query_vector.reshape(1, -1), doc_vectors)[0])
        else:
            scores = bm25_scores(query, texts)
            scores = [
                score * boost
                for score, boost in zip(
                    scores, coordination_boost(rows, tokenize(query)), strict=False
                )
            ]
        ranking = rank_from_scores(scores, doc_ids)
        hits: list[SearchHit] = []
        for doc_id in ranking[:top_k]:
            row = next(item for item in rows if item["record_key"] == doc_id)
            hits.append(
                SearchHit(
                    document_id=doc_id,
                    score=round(float(scores[doc_ids.index(doc_id)]), 6),
                    snippet=" ".join(row["payload"].split()[:40]),
                    source_id=row["source_id"],
                    path=str(row.get("meta", {}).get("path", doc_id)),
                )
            )
        return hits

    def _evidence(
        self, hits: list[SearchHit], rows: list[dict[str, Any]], question_terms: list[str]
    ) -> list[dict[str, Any]]:
        """Evidence is a file that contains several of the query terms.

        Matching a single generic term is not evidence. A file is evidence when
        it contains at least two distinct terms, or when it contains every term
        when the query has only one.
        """
        evidence: list[dict[str, Any]] = []
        by_key = {row["record_key"]: row for row in rows}
        terms = [term for term in question_terms if len(term) > 2]
        if not terms:
            return evidence
        patterns = {term: re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in terms}
        minimum = 2 if len(terms) > 1 else 1
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for rank, hit in enumerate(hits):
            row = by_key.get(hit.document_id)
            if row is None:
                continue
            payload = row["payload"]
            matched = [term for term, pattern in patterns.items() if pattern.search(payload)]
            if len(matched) < minimum:
                continue
            lines = payload.splitlines()
            best_line = 0
            best_count = 0
            for number, line in enumerate(lines, start=1):
                hits_in_line = sum(1 for pattern in patterns.values() if pattern.search(line))
                if hits_in_line > best_count:
                    best_count = hits_in_line
                    best_line = number
            if best_line == 0:
                continue
            scored.append(
                (
                    len(matched),
                    rank,
                    {
                        "path": hit.path,
                        "line": best_line,
                        "text": lines[best_line - 1].strip()[:300],
                        "score": hit.score,
                        "title": lines[0].strip()[:160] if lines else "",
                        "rank": rank,
                        "matched_terms": matched,
                    },
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1]))
        evidence = [item[2] for item in scored]
        return evidence

    def _plan_next_query(
        self, question: str, hits: list[SearchHit], rows: list[dict[str, Any]], used: list[str]
    ) -> tuple[str, str]:
        if self.llm is not None:
            prompt = (
                "You plan the next search step for a long-horizon code search. "
                "Given the question and the current best hits, give one refined search "
                'query. Answer with JSON: {"query": "...", "reason": "..."}.'
                f"\nQuestion: {question}\nUsed queries: {used}\n"
                f"Best hits: {[hit.path for hit in hits[:5]]}"
            )
            try:
                completion = self.llm.complete(prompt, system="You plan code search steps.")
                payload = completion.as_json() or {}
                query = str(payload.get("query", "")).strip()
                if query:
                    return query, str(payload.get("reason", "model planned refinement"))
            except Exception:
                pass
        evidence_paths = {item["path"] for item in self._last_evidence}
        top_documents = []
        for hit in hits:
            row = next((item for item in rows if item["record_key"] == hit.document_id), None)
            if row is None:
                continue
            if evidence_paths and hit.path not in evidence_paths:
                continue
            top_documents.append(row["payload"])
            if len(top_documents) >= 3:
                break
        terms = pseudo_relevance_terms(tokenize(question), top_documents)
        if not terms:
            return "", "no expansion terms available"
        informative = rare_terms(rows, terms, max_ratio=0.01)
        chosen = informative[:5] if informative else terms[:3]
        return " ".join(chosen), "pseudo-relevance feedback"

    def search(
        self,
        question: str,
        dataset_id: str,
        principal: Principal,
        max_steps: int = 6,
        top_k: int = 8,
        cost_cap_usd: float = 1.0,
        actor: str = "ai_engineer",
        min_steps: int = 2,
        max_files: int = 12,
    ) -> AgenticSearchResult:
        corpus = self._corpus(dataset_id)
        if not corpus:
            raise ValueError(f"dataset {dataset_id} has no documents")
        rows = [row for row in corpus if principal.can_read(row["acl"], row["sensitivity"])]
        if not rows:
            self.audit.record(
                actor=actor,
                action="agentic_search.denied",
                subject=dataset_id,
                outcome=AuditOutcome.DENIED,
                inputs={"question_hash": hash(question) & 0xFFFFFFFF},
                details={
                    "user": principal.user_id,
                    "visible_documents": 0,
                    "total_documents": len(corpus),
                    "reason": "no record in the dataset is readable by this principal",
                },
            )
            raise PermissionError(
                f"no records in dataset {dataset_id} are readable by {principal.user_id}"
            )
        decision = self.budget.check(actor, task_id=None)
        if not decision.allowed:
            raise PermissionError(f"search blocked: {decision.reason}")

        question_terms = rare_terms(rows, tokenize(question))
        query = " ".join(question_terms) if question_terms else expand_query(question)
        used: list[str] = []
        steps: list[SearchStep] = []
        evidence: list[dict[str, Any]] = []
        files: list[str] = []
        sufficient = False
        step_index = 0
        strategy = "initial"

        while step_index < max_steps:
            hits = self._retrieve(query, rows, top_k)
            step_evidence = self._evidence(hits, rows, question_terms)
            new_files = [item["path"] for item in step_evidence if item["path"] not in files]
            used.append(query)
            step = SearchStep(
                index=step_index + 1,
                query=query,
                strategy=strategy,
                hits=hits,
                evidence=step_evidence,
            )
            if step_evidence:
                step.decision = "evidence_found"
                step.note = (
                    f"{len(step_evidence)} matching lines across "
                    f"{len({item['path'] for item in step_evidence})} files, "
                    f"{len(new_files)} new"
                )
                sufficient = True
            else:
                step.decision = "refine"
                step.note = "no matching lines, broadening the query"
            steps.append(step)
            evidence.extend(step_evidence)
            self._last_evidence = step_evidence
            for item in step_evidence:
                if item["path"] not in files:
                    files.append(item["path"])
            step_index += 1
            if sufficient and step_index >= min_steps and not new_files:
                step.note += "; no new files, stopping"
                break
            if sufficient and step_index >= min_steps and len(files) >= max_files:
                step.note += "; file budget reached"
                break
            next_query, strategy = self._plan_next_query(question, hits, rows, used)
            if not next_query or next_query in used:
                step.decision = "exhausted"
                step.note += "; no new query could be formed"
                break
            query = next_query

        answer = self._answer(question, evidence, files)
        result = AgenticSearchResult(
            question=question,
            answer=answer,
            steps=steps,
            files=files,
            citations=evidence[:12],
            steps_used=len(steps),
            sufficient_evidence=sufficient,
        )
        self.audit.record(
            actor=actor,
            action="agentic_search.completed",
            subject=dataset_id,
            outcome=AuditOutcome.SUCCESS if sufficient else AuditOutcome.FAILURE,
            inputs={"question_hash": hash(question) & 0xFFFFFFFF, "steps": len(steps)},
            details={
                "files_found": len(files),
                "evidence_lines": len(evidence),
                "sufficient": sufficient,
                "queries": used,
            },
        )
        return result

    @staticmethod
    def _answer(question: str, evidence: list[dict[str, Any]], files: list[str]) -> str:
        if not evidence:
            return (
                f"No supporting evidence was found for '{question}' within the step budget. "
                "The corpus or the query wording may need to change."
            )
        best: dict[str, dict[str, Any]] = {}
        for item in evidence:
            path = item["path"]
            current = best.get(path)
            if current is None or int(item.get("rank", 99)) < int(current.get("rank", 99)):
                best[path] = item
        ordered = sorted(best.values(), key=lambda item: (int(item.get("rank", 99)), item["path"]))
        lines = [f"Found {len(evidence)} supporting locations across {len(best)} files:"]
        for item in ordered[:10]:
            title = item.get("title", "")
            suffix = f"  [{title}]" if title and title != item["text"] else ""
            lines.append(f"- {item['path']}:{item['line']} {item['text']}{suffix}")
        return "\n".join(lines)
