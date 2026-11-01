#!/usr/bin/env python3
"""Run the alpha adaptation on a rented GPU and bring back answers, not weights.

The 1650 cannot hold the pinned base, so this runs on a free Kaggle T4 while
everything that governs it stays on the machine with the audit log: this script
trains and generates, and the only thing it returns is what the model said to
each frozen question plus a record of the bytes it used. Registration, scoring
and gates happen later, at home, against the same frozen set.

It never sees a reference answer. The prompts are the questions alone, so a
generated answer cannot be shaped by the answer it will be judged against.

Both the base and the adapted model answer the same prompts with greedy
decoding, because the comparison is between them and sampling would make the
difference partly luck.

    python kernel.py --data /kaggle/input/internal-ai-alpha-y15 --out /kaggle/working \\
        --base-root /kaggle/tmp/base --download
    python kernel.py --smoke --out /tmp/alpha-smoke --base-root /tmp/alpha-base
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform as host
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# The prompt is not written here: it comes from the package with the pairs, so
# training and generation cannot drift into asking two different questions.
MAX_NEW_TOKENS = 64
TITLE = "internal-ai-alpha-y15"
SNAPSHOT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "model.safetensors.index.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if _sha256(path) != expected:
        raise ValueError(f"{label} is not the recorded bytes: {path.name}")


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def _prepare_environment() -> list[str]:
    """Drop a torchao too old for peft, which otherwise stops LoRA from building.

    Kaggle's image ships torchao 0.10 while peft 0.21 refuses anything below 0.16,
    and peft raises from inside the dispatch path instead of skipping the backend,
    so ``get_peft_model`` cannot run at all. Nothing in this run quantises, so the
    package is not wanted; a run that changes its own environment records it.
    """
    actions: list[str] = []
    try:
        import torchao
        from packaging.version import Version
    except ImportError:
        return actions
    version = getattr(torchao, "__version__", "0")
    try:
        too_old = Version(version) < Version("0.16")
    except ValueError:
        too_old = True
    if too_old:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
            check=False,
            capture_output=True,
            text=True,
        )
        actions.append(f"uninstalled torchao {version}: peft 0.21 needs 0.16 or newer")
        print(actions[-1], flush=True)
    return actions


def _repo_modules(parent: Path) -> dict[str, Any]:
    """The trainer and the pin, imported from the copy the dataset carries.

    The adapter is only worth anything if it was produced by the same training
    step the repository audits, so the copy travels with the data and its hash is
    recorded rather than trusting whatever is installed on the box.
    """
    found = next(
        (
            candidate
            for candidate in (parent, parent / "src")
            if (candidate / "internal_ai" / "training" / "alpha" / "adapt.py").is_file()
        ),
        None,
    )
    if found is None:
        raise FileNotFoundError(f"no internal_ai package under {parent}")
    sys.path.insert(0, str(found))
    from internal_ai.training.alpha.adapt import _load_tokenizer, adapt
    from internal_ai.training.alpha.pairs import prompt_for
    from internal_ai.training.alpha.pin import RECIPE, PinnedBase, WeightShard, verify_snapshot

    return {
        "adapt": adapt,
        "prompt_for": prompt_for,
        "load_tokenizer": _load_tokenizer,
        "PinnedBase": PinnedBase,
        "RECIPE": RECIPE,
        "WeightShard": WeightShard,
        "verify_snapshot": verify_snapshot,
    }


def _base_from(recipe: dict[str, Any], modules: dict[str, Any]) -> Any:
    base = recipe["base"]
    return modules["PinnedBase"](
        repo_id=base["repo_id"],
        revision=base["revision"],
        license=base["license"],
        shards=tuple(
            modules["WeightShard"](shard["name"], shard["sha256"], int(shard["size"]))
            for shard in base["shards"]
        ),
    )


def _ensure_base(base: Any, root: Path, allow_download: bool, modules: dict[str, Any]) -> Path:
    """The pinned snapshot, fetched at the pinned revision if this box lacks it.

    A download is not a shortcut around the pin: the bytes are checked against the
    shard hashes afterwards, so a mirror that served something else is refused
    instead of trained on.
    """
    try:
        return modules["verify_snapshot"](base, root)
    except (FileNotFoundError, ValueError) as exc:
        if not allow_download:
            raise
        print(f"base is not on this box yet ({exc}); fetching the pinned revision", flush=True)
    from huggingface_hub import hf_hub_download

    directory = base.directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    wanted = [shard.name for shard in base.shards] + list(SNAPSHOT_FILES)
    for name in wanted:
        try:
            hf_hub_download(
                repo_id=base.repo_id,
                filename=name,
                revision=base.revision,
                local_dir=str(directory),
            )
        except Exception as exc:
            if name in ("config.json", "tokenizer.json"):
                raise
            print(f"skipped {name}: {type(exc).__name__}", flush=True)
    return modules["verify_snapshot"](base, root)


def _generate(
    prompts: list[str],
    directory: Path,
    adapter: Path | None,
    max_new_tokens: int,
    max_seq_len: int,
    load_tokenizer: Any,
) -> list[str]:
    """Greedy answers from the base, with the adapter when there is one.

    The tokenizer is loaded by the trainer's own loader. Asking transformers to
    infer one from a snapshot that ships a bare `tokenizer.json` silently picked
    a wrapper that encoded the prompt to nothing, and a model asked a zero-token
    question answers nothing for a reason that has nothing to do with the model.
    """
    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = load_tokenizer(directory)
    try:
        model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True, dtype=dtype)
    except TypeError:  # transformers 4 spells the same argument torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            directory, local_files_only=True, torch_dtype=dtype
        )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model = model.to(device).eval()
    answers: list[str] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        padding = {} if tokenizer.pad_token_id is None else {"pad_token_id": tokenizer.pad_token_id}
        with torch.no_grad():
            produced = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                **padding,
            )
        answers.append(
            tokenizer.decode(
                produced[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
        )
        del encoded, produced
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return answers


def run(
    base: Any,
    recipe: dict[str, Any],
    texts: list[str],
    queries: list[dict[str, str]],
    out: Path,
    base_root: Path,
    modules: dict[str, Any],
    allow_download: bool,
    epochs: int,
    seed: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
    adjustments: list[str] | None = None,
) -> dict[str, Any]:
    """Adapt, answer the frozen questions twice, and record how it was done."""
    import torch

    started = time.time()
    max_seq_len = int(recipe.get("max_seq_len") or 64)
    directory = _ensure_base(base, base_root, allow_download, modules)
    adapter = out / "adapter"
    result = modules["adapt"](
        base,
        base_root,
        texts,
        adapter,
        max_seq_len=max_seq_len,
        epochs=epochs,
        # The run at 2e-4 ended with a higher loss than it started (1.406 to
        # 2.327). Halving it keeps the step inside the range this data supports.
        learning_rate=1e-4,
        seed=seed,
    )
    prompts = [modules["prompt_for"](item["query"]) for item in queries]
    load_tokenizer = modules["load_tokenizer"]
    base_answers = _generate(prompts, directory, None, max_new_tokens, max_seq_len, load_tokenizer)
    adapter_answers = _generate(
        prompts, directory, adapter, max_new_tokens, max_seq_len, load_tokenizer
    )
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else host.processor()
    record = {
        "track": "alpha",
        "dataset": {
            "id": recipe.get("dataset_id"),
            "content_hash": recipe.get("dataset_content_hash"),
        },
        "commit": recipe.get("commit"),
        "base": {
            "repo_id": base.repo_id,
            "revision": base.revision,
            "license": base.license,
            "license_class": str(base.license_class),
            "shards_verified": [
                {"name": shard.name, "sha256": shard.sha256} for shard in base.shards
            ],
        },
        "adapter": {
            "backend": result.get("adapter_backend"),
            "sha256": result.get("adapter_sha256"),
            "trainable_parameters": result.get("trainable_parameters"),
            "layers": result.get("layers"),
            "sequences": result.get("sequences"),
            "epochs": result.get("epochs"),
            "loss_first": result.get("loss_first"),
            "loss_last": result.get("loss_last"),
        },
        "generation": {
            "prompt": modules["prompt_for"]("{query}"),
            "max_new_tokens": max_new_tokens,
            "max_seq_len": max_seq_len,
            "decoding": "greedy, one sequence, no sampling",
            "questions": [item["id"] for item in queries],
        },
        "seed": seed,
        "environment": {
            "python": host.python_version(),
            "device": device_name,
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "peft": _version("peft"),
            "kaggle_kernel": host.node(),
            "adjustments": adjustments or [],
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "answers-base.json").write_text(
        json.dumps(base_answers, indent=2) + "\n", encoding="utf-8"
    )
    (out / "answers-adapter.json").write_text(
        json.dumps(adapter_answers, indent=2) + "\n", encoding="utf-8"
    )
    (out / "reproducibility.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def _find_staged(root: Path) -> Path:
    """The staged dataset, wherever this box mounted it.

    Kaggle decides the mount name, and a wrong guess costs a full kernel run to
    discover, so the recipe is looked for instead of the folder being assumed.
    When nothing matches, the error names what was mounted, which is the whole
    diagnosis.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"nothing is mounted at {root}")
    found = [
        path.parent
        for pattern in ("recipe.json", "*/recipe.json", "*/*/recipe.json", "*/*/*/recipe.json")
        for path in sorted(root.glob(pattern))
    ]
    preferred = [path for path in found if path.name == TITLE]
    if preferred or found:
        return (preferred or found)[0]
    mounted = ", ".join(sorted(entry.name for entry in root.iterdir())) or "nothing"
    raise FileNotFoundError(f"no staged dataset under {root}; mounted: {mounted}")


def _staged(data: Path) -> tuple[dict[str, Any], list[str], list[dict[str, str]]]:
    """Read the staged dataset, refusing anything that is not the recorded bytes."""
    recipe = json.loads((data / "recipe.json").read_text(encoding="utf-8"))
    train = data / "train.jsonl"
    _check(train, str(recipe["train_sha256"]), "training text")
    texts = [
        str(json.loads(line)["text"])
        for line in train.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    queries = json.loads((data / "queries.json").read_text(encoding="utf-8"))
    _check(data / "queries.json", str(recipe["queries_sha256"]), "question set")
    return recipe, texts, queries


def smoke(out: Path, base_root: Path) -> dict[str, Any]:
    """The whole path on a two-layer model, so the control flow is proven cheaply."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Qwen3Config, Qwen3ForCausalLM

    modules = _repo_modules(Path(__file__).resolve().parents[1] / "src")
    words = ("the", "ledger", "failed", "nightly", "reconciliation", "which", "[UNK]")
    snapshot = base_root / "Example--tiny" / "rev1"
    snapshot.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(
        WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(snapshot / "tokenizer.json"))
    torch.manual_seed(0)
    Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=len(words),
            max_position_embeddings=64,
        )
    ).save_pretrained(snapshot, safe_serialization=True)
    weights = snapshot / "model.safetensors"
    base = modules["PinnedBase"](
        repo_id="Example/tiny",
        revision="rev1",
        license="Apache-2.0",
        shards=(modules["WeightShard"](weights.name, _sha256(weights), weights.stat().st_size),),
    )
    recipe = {"dataset_id": "smoke", "dataset_content_hash": "smoke", "commit": "smoke"}
    return run(
        base,
        recipe,
        ["the ledger failed", "nightly reconciliation failed", "the nightly ledger failed"],
        [{"id": "rfc0000.txt:1", "query": "which ledger failed"}],
        out,
        base_root,
        modules,
        allow_download=False,
        epochs=1,
        seed=0,
        max_new_tokens=4,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="staged dataset directory")
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--base-root", type=Path, default=Path("/kaggle/tmp/base"))
    parser.add_argument("--sources", type=Path, help="parent of the internal_ai package")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="never fetch the pinned base, even if it is missing",
    )
    parser.add_argument("--smoke", action="store_true", help="run on a tiny local model")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        record = smoke(args.out, args.base_root)
    else:
        if args.data is None:
            args.data = _find_staged(Path("/kaggle/input"))
        recipe, texts, queries = _staged(args.data)
        adjustments = _prepare_environment()
        modules = _repo_modules(args.sources or args.data)
        base = _base_from(recipe, modules)
        record = run(
            base,
            recipe,
            texts,
            queries,
            args.out,
            args.base_root,
            modules,
            allow_download=args.download,
            epochs=args.epochs,
            seed=args.seed,
            adjustments=adjustments,
        )
    print(json.dumps({key: record[key] for key in ("adapter", "environment")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
