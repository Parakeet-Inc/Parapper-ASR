"""Dependency-light inference for SentenceTransformers StaticEmbedding models."""

from __future__ import annotations

import json
import struct
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StaticEmbeddingCoherence:
    piece_sum: float
    piece_mean: float
    vertex_sum: float
    vertex_mean: float
    pieces: int
    vertices: int


class StaticEmbeddingModel:
    """Loads a Unigram-tokenized StaticEmbedding model without Torch."""

    def __init__(self, snapshot: Path) -> None:
        module = snapshot / "0_StaticEmbedding"
        tokenizer = json.loads((module / "tokenizer.json").read_text(encoding="utf-8"))
        model = tokenizer["model"]
        if model["type"] != "Unigram" or model.get("byte_fallback", False):
            raise ValueError("only non-byte-fallback Unigram tokenizers are supported")
        self.unknown_id = int(model["unk_id"])
        self.vocabulary = [piece for piece, _ in model["vocab"]]
        scores = [float(score) for _, score in model["vocab"]]
        self.unknown_score = min(scores) - 10.0
        self.trie: dict[str, dict] = {}
        for token_id, (piece, score) in enumerate(model["vocab"]):
            if token_id <= self.unknown_id or not piece:
                continue
            node = self.trie
            for character in piece:
                node = node.setdefault(character, {})
            node.setdefault("", []).append((token_id, float(score)))

        weights_path = module / "model.safetensors"
        with weights_path.open("rb") as file:
            header_length = struct.unpack("<Q", file.read(8))[0]
            header = json.loads(file.read(header_length))
        tensor = header.get("embedding.weight")
        if tensor is None or tensor["dtype"] != "F32":
            raise ValueError("StaticEmbedding model must contain F32 embedding.weight")
        shape = tuple(int(value) for value in tensor["shape"])
        data_start, data_end = (int(value) for value in tensor["data_offsets"])
        if data_end - data_start != int(np.prod(shape)) * 4:
            raise ValueError("StaticEmbedding tensor byte count does not match its shape")
        self.weights = np.array(
            np.memmap(
                weights_path,
                mode="r",
                dtype="<f4",
                offset=8 + header_length + data_start,
                shape=shape,
            ),
            copy=True,
        )

    @property
    def dimensions(self) -> int:
        return int(self.weights.shape[1])

    @staticmethod
    def normalize(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text).lower()
        return "".join("▁" if character.isspace() else character for character in normalized)

    def tokenize(self, text: str) -> list[tuple[int, int]]:
        """Returns `(token_id, covered_character_count)` without special tokens."""
        normalized = "▁" + self.normalize(text)
        best = [float("-inf")] * (len(normalized) + 1)
        backpointers: list[tuple[int, int, int] | None] = [None] * (
            len(normalized) + 1
        )
        best[0] = 0.0
        for start in range(len(normalized)):
            if not np.isfinite(best[start]):
                continue
            matches: list[tuple[int, int, float]] = []
            node = self.trie
            for end in range(start, len(normalized)):
                child = node.get(normalized[end])
                if child is None:
                    break
                node = child
                matches.extend(
                    (end + 1, token_id, token_score)
                    for token_id, token_score in node.get("", ())
                )
            if not matches:
                matches = [
                    (start + 1, self.unknown_id, self.unknown_score)
                ]
            for end, token_id, token_score in matches:
                score = best[start] + token_score
                if score > best[end]:
                    best[end] = score
                    covered = len(normalized[start:end].replace("▁", ""))
                    if token_id == self.unknown_id:
                        covered = int(normalized[start] != "▁")
                    backpointers[end] = (start, token_id, covered)
        pieces: list[tuple[int, int]] = []
        position = len(normalized)
        while position:
            backpointer = backpointers[position]
            if backpointer is None:
                raise ValueError(f"Unigram tokenizer lost position {position}")
            position, token_id, covered = backpointer
            pieces.append((token_id, covered))
        pieces.reverse()
        return pieces

    def encode_and_coherence(
        self, text: str
    ) -> tuple[np.ndarray, StaticEmbeddingCoherence]:
        pieces = self.tokenize(text)
        token_ids = [token_id for token_id, _ in pieces]
        vectors = np.asarray(self.weights[token_ids], dtype=np.float32)
        sentence = vectors.mean(axis=0)
        sentence_norm = float(np.linalg.norm(sentence))
        if sentence_norm:
            sentence = sentence / sentence_norm
        content_indices = [index for index, (_, covered) in enumerate(pieces) if covered]
        if not content_indices:
            return sentence, StaticEmbeddingCoherence(0.0, 0.0, 0.0, 0.0, 0, 0)
        content = vectors[content_indices]
        norms = np.linalg.norm(content, axis=1)
        cosine = np.divide(
            content @ sentence,
            norms,
            out=np.zeros_like(norms),
            where=norms != 0,
        )
        coverage = np.asarray(
            [pieces[index][1] for index in content_indices], dtype=np.float32
        )
        piece_sum = float(cosine.sum())
        vertex_sum = float(cosine @ coverage)
        return sentence, StaticEmbeddingCoherence(
            piece_sum=piece_sum,
            piece_mean=piece_sum / len(content_indices),
            vertex_sum=vertex_sum,
            vertex_mean=vertex_sum / float(coverage.sum()),
            pieces=len(content_indices),
            vertices=int(coverage.sum()),
        )

    def coherence_batch(
        self, texts: list[str], batch_size: int = 128
    ) -> list[StaticEmbeddingCoherence]:
        """Computes the same coherence values with batched NumPy kernels."""
        results: list[StaticEmbeddingCoherence] = []
        weight_norms = np.linalg.norm(self.weights, axis=1)
        for start in range(0, len(texts), batch_size):
            tokenized = [self.tokenize(text) for text in texts[start : start + batch_size]]
            width = max(len(pieces) for pieces in tokenized)
            token_ids = np.zeros((len(tokenized), width), dtype=np.int64)
            coverage = np.zeros((len(tokenized), width), dtype=np.float32)
            token_mask = np.zeros((len(tokenized), width), dtype=np.float32)
            for row, pieces in enumerate(tokenized):
                length = len(pieces)
                token_ids[row, :length] = [token_id for token_id, _ in pieces]
                coverage[row, :length] = [covered for _, covered in pieces]
                token_mask[row, :length] = 1.0
            vectors = self.weights[token_ids]
            sentence = np.einsum("bld,bl->bd", vectors, token_mask, optimize=True)
            sentence_norms = np.linalg.norm(sentence, axis=1, keepdims=True)
            sentence = np.divide(
                sentence,
                sentence_norms,
                out=np.zeros_like(sentence),
                where=sentence_norms != 0,
            )
            cosine = np.einsum("bld,bd->bl", vectors, sentence, optimize=True)
            norms = weight_norms[token_ids]
            cosine = np.divide(
                cosine,
                norms,
                out=np.zeros_like(cosine),
                where=norms != 0,
            )
            content_mask = coverage > 0
            piece_sums = (cosine * content_mask).sum(axis=1)
            piece_counts = content_mask.sum(axis=1)
            vertex_sums = (cosine * coverage).sum(axis=1)
            vertex_counts = coverage.sum(axis=1)
            for row in range(len(tokenized)):
                results.append(
                    StaticEmbeddingCoherence(
                        piece_sum=float(piece_sums[row]),
                        piece_mean=float(piece_sums[row] / piece_counts[row]),
                        vertex_sum=float(vertex_sums[row]),
                        vertex_mean=float(vertex_sums[row] / vertex_counts[row]),
                        pieces=int(piece_counts[row]),
                        vertices=int(vertex_counts[row]),
                    )
                )
        return results
