from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from summarize_reazon_static_coherence import (  # noqa: E402
    select_embedding,
    select_seed_fusion,
)


class ReazonStaticCoherenceTests(unittest.TestCase):
    def test_embedding_and_fusion_selectors_use_the_requested_score(self) -> None:
        candidates = [
            {
                "rank": 1,
                "raw_score": -1.0,
                "token_ids": [1, 2],
                "coherence": {"vertex_mean": 0.1},
            },
            {
                "rank": 2,
                "raw_score": -1.1,
                "token_ids": [3, 4],
                "coherence": {"vertex_mean": 0.9},
            },
        ]

        self.assertIs(select_embedding(candidates, "vertex_mean"), candidates[1])
        self.assertIs(
            select_seed_fusion(candidates, "vertex_mean", 0.0, 0.0),
            candidates[0],
        )
        self.assertIs(
            select_seed_fusion(candidates, "vertex_mean", 0.0, 1.0),
            candidates[1],
        )


if __name__ == "__main__":
    unittest.main()
