from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from static_embedding_numpy import StaticEmbeddingModel  # noqa: E402


class StaticEmbeddingNumpyTests(unittest.TestCase):
    def test_token_coherence_counts_every_covered_vertex(self) -> None:
        model = object.__new__(StaticEmbeddingModel)
        model.unknown_id = 0
        model.unknown_score = -100.0
        model.vocabulary = ["<unk>", "▁", "AB", "C"]
        model.trie = {
            "▁": {"": [(1, 0.0)]},
            "a": {"b": {"": [(2, 0.0)]}},
            "c": {"": [(3, 0.0)]},
        }
        model.weights = np.asarray(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )

        _, score = model.encode_and_coherence("ABC")
        batched = model.coherence_batch(["ABC"])[0]

        self.assertEqual(model.tokenize("ABC"), [(1, 0), (2, 2), (3, 1)])
        self.assertEqual(score.pieces, 2)
        self.assertEqual(score.vertices, 3)
        self.assertAlmostEqual(score.piece_sum, 2**0.5)
        self.assertAlmostEqual(score.piece_mean, 2**-0.5)
        self.assertAlmostEqual(score.vertex_sum, 3 * 2**-0.5, places=6)
        self.assertAlmostEqual(score.vertex_mean, 2**-0.5)
        self.assertEqual((score.pieces, score.vertices), (batched.pieces, batched.vertices))
        for field in ("piece_sum", "piece_mean", "vertex_sum", "vertex_mean"):
            self.assertAlmostEqual(getattr(score, field), getattr(batched, field), places=6)


if __name__ == "__main__":
    unittest.main()
