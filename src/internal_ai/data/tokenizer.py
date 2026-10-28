"""Byte-pair encoding tokenizer trained from scratch on internal data (FR-9).

No pretrained vocabulary and no external tokenizer library: the vocabulary is
learned from the released corpus only.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PIECE_PATTERN = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


@dataclass(slots=True)
class Tokenizer:
    vocab: dict[str, int]
    merges: list[tuple[str, str]]
    frequencies: Counter[str] = field(default_factory=Counter)
    ranks: dict[tuple[str, str], int] = field(default_factory=dict, repr=False, compare=False)
    merge_order: tuple[tuple[str, str], ...] = field(
        default_factory=tuple, repr=False, compare=False
    )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, max_tokens: int | None = None) -> list[int]:
        unk = self.vocab.get("<unk>", 0)
        tokens: list[int] = []
        for piece in PIECE_PATTERN.findall(text):
            if piece.isspace():
                tokens.extend(self.vocab.get(char, unk) for char in piece)
            else:
                tokens.extend(self.vocab.get(symbol, unk) for symbol in self._encode_word(piece))
            if max_tokens is not None and len(tokens) >= max_tokens:
                return tokens[:max_tokens]
        return tokens

    def _encode_word(self, word: str) -> list[str]:
        """Apply the learned merges to one word, cheapest merge first.

        Walking the whole merge list for every word does not scale to a real
        corpus. Merging the lowest ranked adjacent pair repeatedly is the same
        result and costs the length of the word, not the size of the vocabulary.
        """
        symbols = list(word)
        while len(symbols) > 1:
            best_rank: int | None = None
            best_index = -1
            for index in range(len(symbols) - 1):
                rank = self.ranks.get((symbols[index], symbols[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
            if best_index < 0:
                break
            symbols[best_index : best_index + 2] = [symbols[best_index] + symbols[best_index + 1]]
        return symbols

    def whitespace_ids(self) -> set[int]:
        return {index for token, index in self.vocab.items() if token.isspace()}

    def content_ids(self, ids: list[int]) -> list[int]:
        """Word pieces. Spacing stays in encode so text round-trips; the model trains on words."""
        skip = self.whitespace_ids()
        return [item for item in ids if item not in skip]

    def decode(self, ids: list[int]) -> str:
        inverse = {index: token for token, index in self.vocab.items()}
        hidden = {"<pad>", "<bos>", "<eos>"}
        return "".join(token for item in ids if (token := inverse.get(item, "<unk>")) not in hidden)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Tokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(pair) for pair in payload["merges"]]
        ranks = {pair: rank for rank, pair in enumerate(merges)}
        return cls(
            vocab=dict(payload["vocab"]),
            merges=merges,
            ranks=ranks,
            merge_order=tuple(merges),
        )


def _apply_merge(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and symbols[index] == pair[0] and symbols[index + 1] == pair[1]:
            out.append(pair[0] + pair[1])
            index += 2
        else:
            out.append(symbols[index])
            index += 1
    return out


def _pairs(symbols: tuple[str, ...]) -> list[tuple[str, str]]:
    return [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]


def train_tokenizer(
    texts: list[str],
    vocab_size: int = 8_000,
    min_frequency: int = 2,
    max_documents: int = 20_000,
    max_unique_words: int = 200_000,
) -> Tokenizer:
    """Learn byte-pair merges from the corpus with incremental pair counting."""
    if len(texts) > max_documents:
        step = len(texts) // max_documents
        texts = texts[::step][:max_documents]
    word_counts: Counter[str] = Counter()
    for text in texts:
        word_counts.update(WORD_PATTERN.findall(text))
    if len(word_counts) > max_unique_words:
        word_counts = Counter(dict(word_counts.most_common(max_unique_words)))

    vocab: dict[str, int] = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    char_counts: Counter[str] = Counter()
    for word in word_counts:
        char_counts.update(word)
    for char, _count in sorted(char_counts.items(), key=lambda item: (-item[1], item[0])):
        if char not in vocab:
            vocab[char] = len(vocab)
    for text in texts:
        for char in text:
            if char.isspace() and char not in vocab:
                vocab[char] = len(vocab)

    words: list[tuple[str, ...]] = []
    freqs: list[int] = []
    index_of: dict[tuple[str, ...], int] = {}
    for word, count in word_counts.items():
        symbols = tuple(word)
        if symbols in index_of:
            freqs[index_of[symbols]] += count
            continue
        index_of[symbols] = len(words)
        words.append(symbols)
        freqs.append(count)

    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_words: dict[tuple[str, str], set[int]] = {}
    for word_id, symbols in enumerate(words):
        for pair in _pairs(symbols):
            pair_counts[pair] += freqs[word_id]
            pair_words.setdefault(pair, set()).add(word_id)

    merges: list[tuple[str, str]] = []
    ranks: dict[tuple[str, str], int] = {}
    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        best = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        if pair_counts[best] < min_frequency:
            break
        new_symbol = best[0] + best[1]
        if new_symbol in vocab:
            pair_counts.pop(best, None)
            pair_words.pop(best, None)
            continue
        affected = sorted(pair_words.get(best, set()))
        for word_id in affected:
            old = words[word_id]
            freq = freqs[word_id]
            for pair in _pairs(old):
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                holders = pair_words.get(pair)
                if holders:
                    holders.discard(word_id)
            new = tuple(_apply_merge(list(old), best))
            words[word_id] = new
            for pair in _pairs(new):
                pair_counts[pair] += freq
                pair_words.setdefault(pair, set()).add(word_id)
        vocab[new_symbol] = len(vocab)
        merges.append(best)
        ranks[best] = len(merges) - 1
    return Tokenizer(
        vocab=vocab,
        merges=merges,
        frequencies=word_counts,
        ranks=ranks,
        merge_order=tuple(merges),
    )


def coverage_report(
    tokenizer: Tokenizer,
    texts_by_source: dict[str, list[str]],
    jargon: list[str] | None = None,
    max_documents: int = 200,
) -> dict[str, Any]:
    """Vocabulary coverage of internal jargon and code (FR-9).

    Coverage is measured on a sample per source. Measuring every document of a
    large corpus costs more than it tells us: the statistics are per word, and
    a few hundred documents already characterise the vocabulary.
    """
    per_source: dict[str, Any] = {}
    for source, texts in texts_by_source.items():
        if len(texts) > max_documents:
            step = len(texts) // max_documents
            texts = texts[::step][:max_documents]
        token_counts = 0
        word_count = 0
        fragmented = 0
        for text in texts:
            for word in WORD_PATTERN.findall(text):
                word_count += 1
                ids = tokenizer.encode(word)
                token_counts += len(ids)
                if len(ids) > 1:
                    fragmented += 1
        per_source[source] = {
            "words": word_count,
            "mean_tokens_per_word": round(token_counts / word_count, 4) if word_count else 0.0,
            "fragmented_word_ratio": round(fragmented / word_count, 6) if word_count else 0.0,
        }
    jargon_results: dict[str, Any] = {}
    for word in jargon or []:
        ids = tokenizer.encode(word)
        jargon_results[word] = {"tokens": len(ids), "single_token": len(ids) == 1}
    covered = sum(1 for item in jargon_results.values() if item["single_token"])
    return {
        "vocab_size": tokenizer.vocab_size,
        "merges": len(tokenizer.merges),
        "per_source": per_source,
        "jargon": jargon_results,
        "jargon_single_token_coverage": round(covered / len(jargon_results), 4)
        if jargon_results
        else None,
    }
