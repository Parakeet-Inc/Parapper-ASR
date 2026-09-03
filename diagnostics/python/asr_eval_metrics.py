"""Shared text normalization and alignment metrics for offline ASR evaluation."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence


AlignmentOperation = str


def diagnostic_normalize(text: str) -> str:
    """Apply the repository's diagnostic CER normalization contract."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def semantic_normalize(text: str) -> str:
    """Remove punctuation without concatenating words across its boundary."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    with_boundaries = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(with_boundaries.split())


def reading_normalize(text: str) -> str:
    """Normalize reading text before Kana-CER alignment."""
    return diagnostic_normalize(text)


def align_characters(reference: str, hypothesis: str) -> list[AlignmentOperation]:
    """Return a minimum-edit alignment with a deterministic operation tie break.

    Equal-cost paths prefer correct, substitution, deletion, then insertion.
    """
    reference_length = len(reference)
    hypothesis_length = len(hypothesis)
    costs = [list(range(hypothesis_length + 1))]
    backtrace: list[list[AlignmentOperation | None]] = [
        [None] + ["insertion"] * hypothesis_length
    ]

    for reference_index in range(1, reference_length + 1):
        costs.append([reference_index] + [0] * hypothesis_length)
        backtrace.append(["deletion"] + [None] * hypothesis_length)
        for hypothesis_index in range(1, hypothesis_length + 1):
            if reference[reference_index - 1] == hypothesis[hypothesis_index - 1]:
                candidates = [
                    (
                        costs[reference_index - 1][hypothesis_index - 1],
                        0,
                        "correct",
                    ),
                    (
                        costs[reference_index - 1][hypothesis_index] + 1,
                        2,
                        "deletion",
                    ),
                    (
                        costs[reference_index][hypothesis_index - 1] + 1,
                        3,
                        "insertion",
                    ),
                ]
            else:
                candidates = [
                    (
                        costs[reference_index - 1][hypothesis_index - 1] + 1,
                        1,
                        "substitution",
                    ),
                    (
                        costs[reference_index - 1][hypothesis_index] + 1,
                        2,
                        "deletion",
                    ),
                    (
                        costs[reference_index][hypothesis_index - 1] + 1,
                        3,
                        "insertion",
                    ),
                ]

            selected_cost, _, selected_operation = min(candidates)
            costs[reference_index][hypothesis_index] = selected_cost
            backtrace[reference_index][hypothesis_index] = selected_operation

    operations: list[AlignmentOperation] = []
    reference_index = reference_length
    hypothesis_index = hypothesis_length
    while reference_index > 0 or hypothesis_index > 0:
        operation = backtrace[reference_index][hypothesis_index]
        if operation is None:
            raise RuntimeError("alignment backtrace terminated before the origin")
        operations.append(operation)
        if operation in {"correct", "substitution"}:
            reference_index -= 1
            hypothesis_index -= 1
        elif operation == "deletion":
            reference_index -= 1
        else:
            hypothesis_index -= 1

    operations.reverse()
    return operations


def summarize_alignment(
    operations: Sequence[AlignmentOperation],
) -> dict[str, int]:
    substitutions = operations.count("substitution")
    deletions = operations.count("deletion")
    insertions = operations.count("insertion")

    max_deletion_run = 0
    current_deletion_run = 0
    for operation in operations:
        if operation == "deletion":
            current_deletion_run += 1
            max_deletion_run = max(max_deletion_run, current_deletion_run)
        else:
            current_deletion_run = 0

    leading_deletions = 0
    for operation in operations:
        if operation != "deletion":
            break
        leading_deletions += 1

    trailing_deletions = 0
    for operation in reversed(operations):
        if operation != "deletion":
            break
        trailing_deletions += 1

    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "max_deletion_run": max_deletion_run,
        "leading_deletions": leading_deletions,
        "trailing_deletions": trailing_deletions,
    }
