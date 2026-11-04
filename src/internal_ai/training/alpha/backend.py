"""Serving backends: the adapter the card names is the adapter that answers.

A promoted alpha adapter is a PEFT directory whose tree hash is recorded on the
model card. Serving it means loading the pinned snapshot and the adapter the
card names, refusing anything whose hash drifted, and saying on the health
surface which hash is answering. The backend here is a reference, not a serving
engine: it loads with transformers and peft, the libraries the promoted adapter
names, and generates greedily, because an answer a run cannot repeat twice is
not evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from internal_ai.context import Platform
from internal_ai.domain import ModelVersion
from internal_ai.ids import tree_hash
from internal_ai.llm import Completion
from internal_ai.training.alpha.local import LocalGenerationUnavailable, base_for
from internal_ai.training.alpha.pin import verify_snapshot
from internal_ai.training.provenance import require_open_weights

LOADED: dict[str, str] = {}


def loaded_adapters() -> dict[str, str]:
    """The models whose adapter this process answered with, and their hashes."""
    return dict(LOADED)


def require_runner_hash(model: ModelVersion, adapter: Any) -> None:
    """A runner may answer only for the adapter the card names, and must say which.

    The runner owns the weights; this process never sees them. The report on its
    health surface is the only evidence that the bytes promotion scored are the
    bytes answering, so a missing report is a refusal rather than a pass.
    """
    expected = str(model.card.get("adapter_sha256") or "")
    if not expected:
        # Nothing was promoted for this model, so there is no adapter hash to hold
        # the runner to; the plain runner path stays as it was.
        return
    reported = adapter.reported_adapter_hash()
    if not reported:
        raise PermissionError(
            f"the runner at {adapter.health_url()} does not report which adapter it loaded; "
            "a process that cannot name its adapter is refused"
        )
    if reported != expected:
        raise PermissionError(
            f"the runner reports adapter {reported[:16]} but the model card names "
            f"{expected[:16]}: a tree_hash mismatch is a different adapter than the one "
            "its eval report scored"
        )


def peft_backend(platform: Platform, model: ModelVersion) -> TransformersPeftBackend:
    """Load the adapter the card names, and refuse one whose tree hash drifted.

    Reading pretrained weights is the act provenance mode governs, so the load
    asks first, exactly as the streamed serving path does.
    """
    candidate = Path(model.path or "")
    require_open_weights(platform, "serving the adapter from the base model")
    if not candidate.is_dir() or not (candidate / "adapter_config.json").exists():
        raise PermissionError(
            f"model {model.id} does not point at an adapter directory "
            f"({candidate or 'no path on the card'})"
        )
    recorded = str(model.card.get("adapter_sha256") or "")
    if not recorded:
        raise PermissionError(f"no adapter hash on the card for model {model.id}")
    actual = tree_hash(candidate)
    if actual != recorded:
        raise PermissionError(
            f"the adapter at {candidate} hashes {actual[:16]} but the model card names "
            f"{recorded[:16]}: a tree_hash mismatch is a different adapter than the one "
            "its eval report scored"
        )
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - the alpha extra is optional
        raise LocalGenerationUnavailable(
            "serving a full-depth adapter needs the libraries that trained it: install "
            "the alpha extra (pip install 'internal-ai[alpha]')"
        ) from exc

    base = base_for(model)
    directory = verify_snapshot(base, platform.settings.root)
    loaded = AutoModelForCausalLM.from_pretrained(directory)
    adapted = PeftModel.from_pretrained(loaded, candidate)
    adapted.eval()
    backend = TransformersPeftBackend(
        model_id=model.id,
        adapted=adapted,
        directory=directory,
        adapter_path=candidate,
        adapter_sha256=recorded,
        model_name=f"{base.repo_id}+{model.name}-v{model.version}",
        max_tokens=400,
    )
    LOADED[model.id] = recorded
    return backend


class TransformersPeftBackend:
    """The pinned base with the runtime LoRA on it, answering on this machine."""

    name = "transformers-peft"

    def __init__(
        self,
        model_id: str = "",
        adapted: Any = None,
        directory: Path | None = None,
        adapter_path: Path | None = None,
        adapter_sha256: str = "",
        model_name: str = "",
        max_tokens: int = 400,
    ) -> None:
        self.model_id = model_id
        self.adapted = adapted
        self.directory = directory
        self.adapter_path = adapter_path
        self.adapter_sha256 = adapter_sha256
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._tokenizer: Any = None

    def complete(  # type: ignore[no-untyped-def]
        self,
        prompt: str,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ):
        if temperature:
            raise ValueError("local generation is greedy; a temperature would make it unrepeatable")
        import torch

        if self._tokenizer is None:
            # transformers' AutoTokenizer handed back empty ids for the pinned
            # WordLevel tokenizer.json; the tokenizers package the snapshot ships
            # encodes it correctly, so the pinned file is read through that.
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:  # pragma: no cover - the alpha extra is optional
                raise LocalGenerationUnavailable(
                    "serving needs the tokenizer the base ships with: install the alpha extra"
                ) from exc
            self._tokenizer = Tokenizer.from_file(str(self.directory / "tokenizer.json"))
        text = f"{system.strip()}\n{prompt}".strip() if system else prompt
        ids = self._tokenizer.encode(text).ids
        input_ids = torch.tensor([ids], dtype=torch.long)
        output = self.adapted.generate(
            input_ids=input_ids,
            max_new_tokens=min(max_tokens or self.max_tokens, self.max_tokens),
            do_sample=False,
            pad_token_id=0,
        )
        produced = output[0, len(ids) :]
        answer = self._tokenizer.decode(produced.tolist(), skip_special_tokens=True).strip()
        return Completion(
            text=answer,
            model=self.model_name,
            input_tokens=len(ids),
            output_tokens=int(produced.shape[0]),
        )

    def health(self) -> dict[str, str]:
        """What this process is running: the backend, and the hash it verified."""
        return {
            "backend": self.name,
            "adapter_sha256": self.adapter_sha256,
        }
