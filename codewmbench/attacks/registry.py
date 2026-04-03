from __future__ import annotations

from .base import AttackBundle
from .implementations import (
    BlockShuffleAttack,
    BudgetedAdaptiveAttack,
    CommentStripAttack,
    ControlFlowFlattenAttack,
    IdentifierRenameAttack,
    NoiseInsertAttack,
    WhitespaceNormalizeAttack,
)


_ATTACK_INFO: dict[str, dict[str, object]] = {
    "comment_strip": {
        "severity": 0.12,
        "description": "Remove comments and docstrings.",
    },
    "identifier_rename": {
        "severity": 0.2,
        "description": "Rename identifiers while preserving structure.",
    },
    "whitespace_normalize": {
        "severity": 0.08,
        "description": "Whitespace-only normalization.",
    },
    "noise_insert": {
        "severity": 0.18,
        "description": "Add benign comment noise to stress detection.",
    },
    "control_flow_flatten": {
        "severity": 0.42,
        "description": "Flatten redundant control-flow scaffolds while preserving semantics.",
    },
    "block_shuffle": {
        "severity": 0.35,
        "description": "Shuffle code blocks while preserving local semantics.",
    },
    "budgeted_adaptive": {
        "severity": 0.5,
        "description": "Compose multiple transformations under a perturbation budget.",
    },
}


def available_attacks() -> tuple[str, ...]:
    return tuple(_ATTACK_INFO.keys())


def build_attack_bundle(name: str) -> AttackBundle:
    name = name.lower()
    info = _ATTACK_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown attack: {name}")
    if name == "comment_strip":
        attack = CommentStripAttack()
    elif name == "identifier_rename":
        attack = IdentifierRenameAttack()
    elif name == "whitespace_normalize":
        attack = WhitespaceNormalizeAttack()
    elif name == "noise_insert":
        attack = NoiseInsertAttack()
    elif name == "control_flow_flatten":
        attack = ControlFlowFlattenAttack()
    elif name == "block_shuffle":
        attack = BlockShuffleAttack()
    elif name == "budgeted_adaptive":
        attack = BudgetedAdaptiveAttack()
    else:  # pragma: no cover - guarded by info lookup
        raise KeyError(f"unknown attack: {name}")
    return AttackBundle(
        name=name,
        attack=attack,
        severity=float(info.get("severity", 0.0)),
        description=str(info.get("description", "")),
    )

