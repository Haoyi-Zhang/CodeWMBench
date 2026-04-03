from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from ..models import AttackOutcome
from ..utils import normalize_whitespace, stable_hash, strip_comments
from .base import CodeAttack


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_KEYWORDS = {
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "False",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "None",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "True",
    "try",
    "while",
    "with",
    "yield",
}


def _rename_identifiers(source: str, *, seed: int) -> tuple[str, list[str]]:
    names = [name for name in _IDENTIFIER_RE.findall(source) if name not in _KEYWORDS and len(name) > 2]
    mapping: dict[str, str] = {}
    notes: list[str] = []
    for index, name in enumerate(dict.fromkeys(names)):
        token = stable_hash(f"{seed}:{name}")[:6]
        replacement = f"v_{index}_{token}"
        mapping[name] = replacement
        notes.append(f"{name}->{replacement}")
    mutated = source
    for name, replacement in mapping.items():
        mutated = re.sub(rf"\b{re.escape(name)}\b", replacement, mutated)
    return mutated, notes


def _wrap_comment_block(seed: int) -> str:
    token = stable_hash(f"noise:{seed}")[:10]
    return f"# artifact-noise:{token}"


def _shuffle_blocks(source: str, *, seed: int) -> tuple[str, list[str]]:
    blocks = [block for block in source.split("\n\n") if block.strip()]
    if len(blocks) <= 1:
        return source, []
    order = list(range(len(blocks)))
    for idx in range(len(order)):
        swap = (seed + idx * 7) % len(order)
        order[idx], order[swap] = order[swap], order[idx]
    shuffled = [blocks[idx] for idx in order]
    return "\n\n".join(shuffled), [f"order={order}"]


def _is_python_scaffold(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    if not isinstance(node.test, ast.Constant) or node.test.value is not True:
        return False
    if node.orelse:
        return False
    if len(node.body) != 1:
        return False
    child = node.body[0]
    if isinstance(child, ast.Pass):
        return True
    return _is_python_scaffold(child)


class _PythonScaffoldStripper(ast.NodeTransformer):
    @staticmethod
    def _strip_leading_scaffold(body: list[ast.stmt]) -> list[ast.stmt]:
        stripped = list(body)
        while stripped and _is_python_scaffold(stripped[0]):
            stripped = stripped[1:]
        return stripped

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        node.body = self._strip_leading_scaffold(list(node.body))
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.body = self._strip_leading_scaffold(list(node.body))
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.body = self._strip_leading_scaffold(list(node.body))
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        node.body = self._strip_leading_scaffold(list(node.body))
        return node


def _flatten_python_scaffold(source: str) -> tuple[str, list[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []
    stripped_tree = _PythonScaffoldStripper().visit(tree)
    ast.fix_missing_locations(stripped_tree)
    try:
        mutated = ast.unparse(stripped_tree)
    except Exception:  # pragma: no cover - ast.unparse should be available on py311+
        return source, []
    if mutated == source:
        return source, []
    return mutated, ["python_control_flow_flattened"]


def _brace_if_open(line: str, *, style: str) -> bool:
    stripped = line.strip()
    if style == "rust":
        return bool(re.match(r"^if\s+true\s*\{\s*$", stripped, flags=re.IGNORECASE))
    return bool(re.match(r"^if\s*\(\s*true\s*\)\s*\{\s*$", stripped, flags=re.IGNORECASE))


def _flatten_brace_scaffold(source: str, *, style: str) -> tuple[str, list[str]]:
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if "{" not in line:
            continue
        start = index + 1
        if start >= len(lines) or not _brace_if_open(lines[start], style=style):
            continue
        cursor = start
        depth = 0
        while cursor < len(lines):
            current = lines[cursor].strip()
            if _brace_if_open(lines[cursor], style=style):
                depth += 1
                cursor += 1
                continue
            if _BRACE_CLOSE.match(current):
                if depth == 0:
                    break
                depth -= 1
                cursor += 1
                if depth == 0:
                    mutated = "\n".join([*lines[:start], *lines[cursor:]])
                    if mutated != source:
                        return mutated, [f"brace_control_flow_flattened:{style}"]
                    return source, []
                continue
            break
        break
    return source, []


@dataclass(slots=True)
class CommentStripAttack(CodeAttack):
    name: str = "comment_strip"

    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        mutated = strip_comments(source)
        return AttackOutcome(
            attack_name=self.name,
            source=normalize_whitespace(mutated),
            changed=mutated != source,
            notes=("comments_removed",),
            metadata=metadata or {},
        )


@dataclass(slots=True)
class WhitespaceNormalizeAttack(CodeAttack):
    name: str = "whitespace_normalize"

    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        mutated = normalize_whitespace(source)
        return AttackOutcome(
            attack_name=self.name,
            source=mutated,
            changed=mutated != source,
            notes=("whitespace_normalized",),
            metadata=metadata or {},
        )


@dataclass(slots=True)
class IdentifierRenameAttack(CodeAttack):
    name: str = "identifier_rename"

    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        mutated, notes = _rename_identifiers(source, seed=seed)
        return AttackOutcome(
            attack_name=self.name,
            source=mutated,
            changed=mutated != source,
            notes=tuple(notes or ["no_rename"]),
            metadata=metadata or {},
        )


@dataclass(slots=True)
class NoiseInsertAttack(CodeAttack):
    name: str = "noise_insert"

    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        noise = _wrap_comment_block(seed)
        mutated = "\n".join([noise, source, noise])
        return AttackOutcome(
            attack_name=self.name,
            source=mutated,
            changed=mutated != source,
            notes=("noise_comments_added",),
            metadata=metadata or {},
        )


@dataclass(slots=True)
class BlockShuffleAttack(CodeAttack):
    name: str = "block_shuffle"

    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        mutated, notes = _shuffle_blocks(source, seed=seed)
        return AttackOutcome(
            attack_name=self.name,
            source=mutated,
            changed=mutated != source,
            notes=tuple(notes or ["no_shuffle"]),
            metadata=metadata or {},
        )


@dataclass(slots=True)
class ControlFlowFlattenAttack(CodeAttack):
    name: str = "control_flow_flatten"

    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        language = str((context or {}).get("language") or (metadata or {}).get("language") or "").lower()
        if language == "python":
            mutated, notes = _flatten_python_scaffold(source)
        elif language in {"javascript", "java"}:
            mutated, notes = _flatten_brace_scaffold(source, style=language)
        elif language == "rust":
            mutated, notes = _flatten_brace_scaffold(source, style="rust")
        else:
            mutated, notes = _flatten_python_scaffold(source)
            if mutated == source:
                mutated, notes = _flatten_brace_scaffold(source, style="javascript")
            if mutated == source:
                mutated, notes = _flatten_brace_scaffold(source, style="rust")
        return AttackOutcome(
            attack_name=self.name,
            source=mutated,
            changed=mutated != source,
            notes=tuple(notes or ["no_flattening"]),
            metadata=metadata or {},
        )


@dataclass(slots=True)
class BudgetedAdaptiveAttack(CodeAttack):
    name: str = "budgeted_adaptive"
    candidate_order: tuple[str, ...] = (
        "comment_strip",
        "whitespace_normalize",
        "identifier_rename",
        "control_flow_flatten",
        "block_shuffle",
    )

    def _candidate_attack(self, name: str) -> CodeAttack:
        if name == "comment_strip":
            return CommentStripAttack()
        if name == "whitespace_normalize":
            return WhitespaceNormalizeAttack()
        if name == "identifier_rename":
            return IdentifierRenameAttack()
        if name == "block_shuffle":
            return BlockShuffleAttack()
        if name == "control_flow_flatten":
            return ControlFlowFlattenAttack()
        if name == "noise_insert":
            return NoiseInsertAttack()
        raise KeyError(name)

    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        context = dict(context or {})
        config = dict(context.get("config") or {})
        budget = int(config.get("budget", context.get("budget", 3)))
        min_quality = float(config.get("min_quality", context.get("min_quality", 0.0)))
        candidate_order = tuple(config.get("candidate_order") or self.candidate_order)
        detector = context.get("detector")
        quality = context.get("quality")
        validate = context.get("validate")

        current = source
        current_score = float(detector(current)) if callable(detector) else 1.0
        current_quality = float(quality(current)) if callable(quality) else 1.0
        current_semantic = validate(current) if callable(validate) else None
        curve: list[dict[str, Any]] = [
            {
                "budget": 0,
                "detector_score": round(current_score, 4),
                "quality_score": round(current_quality, 4),
                "semantic_preserving": current_semantic,
                "step_name": "start",
            }
        ]
        selected_steps: list[str] = []

        for spent in range(1, budget + 1):
            best_choice: tuple[float, float, str, str, bool | None] | None = None
            for step_name in candidate_order:
                candidate_attack = self._candidate_attack(step_name)
                candidate_outcome = candidate_attack.apply(
                    current,
                    seed=seed + spent,
                    metadata=metadata or {},
                    context=context,
                )
                candidate_source = candidate_outcome.source
                candidate_score = float(detector(candidate_source)) if callable(detector) else current_score
                candidate_quality = float(quality(candidate_source)) if callable(quality) else current_quality
                candidate_semantic = validate(candidate_source) if callable(validate) else None
                if candidate_semantic is False:
                    continue
                if candidate_quality < min_quality:
                    continue
                choice = (candidate_score, -candidate_quality, step_name, candidate_source, candidate_semantic)
                if best_choice is None or choice < best_choice:
                    best_choice = choice
            if best_choice is None:
                break
            candidate_score, neg_quality, step_name, candidate_source, candidate_semantic = best_choice
            candidate_quality = -neg_quality
            if candidate_score >= current_score and candidate_quality <= current_quality:
                break
            current = candidate_source
            current_score = candidate_score
            current_quality = candidate_quality
            current_semantic = candidate_semantic
            selected_steps.append(step_name)
            curve.append(
                {
                    "budget": spent,
                    "detector_score": round(current_score, 4),
                    "quality_score": round(current_quality, 4),
                    "semantic_preserving": current_semantic,
                    "step_name": step_name,
                }
            )

        outcome_metadata = dict(metadata or {})
        outcome_metadata.update(
            {
                "budget": budget,
                "selected_steps": selected_steps,
                "budget_curve": curve,
                "final_detector_score": round(current_score, 4),
                "final_quality_score": round(current_quality, 4),
                "semantic_preserving": current_semantic,
            }
        )
        notes = (f"budget={budget}", f"steps={','.join(selected_steps) or 'none'}")
        return AttackOutcome(
            attack_name=self.name,
            source=current,
            changed=current != source,
            notes=notes,
            metadata=outcome_metadata,
        )
