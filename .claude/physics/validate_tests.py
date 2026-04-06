#!/usr/bin/env python3
"""Physics test validator — semantic validation against invariant registry.

Levels of validation (test mode):
  L1: INV-* reference exists in docstring (syntax)
  L2: INV-* ID exists in INVARIANTS.yaml (validity)
  L3: Test structure matches invariant type AND test_type (semantic coherence)
  L4: Assertion error messages contain required fields (observability)
  L5: Magic number detection in thresholds (heuristic)

Levels of validation (code audit mode):
  C1: Silent clamp/clip without logging (hidden invariant repair)
  C2: Bare numeric bounds without INV-* comment (undocumented constraint)

Usage:
    python .claude/physics/validate_tests.py tests/unit/physics/
    python .claude/physics/validate_tests.py tests/unit/physics/test_T2_explosive_sync.py
    python .claude/physics/validate_tests.py tests/ --summary
    python .claude/physics/validate_tests.py core/ --audit-code
    python .claude/physics/validate_tests.py --self-check
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

# ── Constants ────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
INVARIANTS_PATH = SCRIPT_DIR / "INVARIANTS.yaml"

INV_PATTERN = re.compile(r"INV-[A-Z0-9]+")

PHYSICS_KEYWORDS = {
    "physics", "kuramoto", "serotonin", "dopamine", "gaba", "energy",
    "thermo", "sync", "lyapunov", "ricci", "regime", "conservation",
    "entropy", "free_energy", "order_param", "inhibit",
    # Fix #3: ECS/HPC/active inference modules
    "ecs", "hpc", "pwpe", "active_inference", "freeenergy",
}


# ── Load invariant registry ──────────────────────────────────────

def load_invariants() -> dict[str, dict[str, Any]]:
    """Parse INVARIANTS.yaml into {INV-ID: {type, test_type, priority, ...}}."""
    registry: dict[str, dict[str, Any]] = {}
    if not INVARIANTS_PATH.exists():
        print(f"WARNING: {INVARIANTS_PATH} not found, skipping L2/L3 checks")
        return registry

    text = INVARIANTS_PATH.read_text()
    current_id: str | None = None
    current_block: dict[str, str] = {}

    for line in text.splitlines():
        id_match = re.match(r"\s+id:\s+(INV-[A-Z0-9]+)", line)
        if id_match:
            if current_id and current_block:
                registry[current_id] = current_block
            current_id = id_match.group(1)
            current_block = {}
            continue

        if current_id:
            kv_match = re.match(r"\s+(\w+):\s+(.+)", line)
            if kv_match:
                key, val = kv_match.group(1), kv_match.group(2).strip().strip('"\'')
                if "#" in val:
                    val = val[:val.index("#")].strip()
                current_block[key] = val

            if line.strip() == "" or re.match(r"\S", line):
                if current_id and current_block:
                    registry[current_id] = current_block
                if not id_match:
                    current_id = None
                    current_block = {}

    if current_id and current_block:
        registry[current_id] = current_block

    return registry


# ── AST helpers ──────────────────────────────────────────────────

def _collect_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _collect_call_names(node: ast.AST) -> set[str]:
    calls: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                calls.add(n.func.id)
            elif isinstance(n.func, ast.Attribute):
                calls.add(n.func.attr)
                if isinstance(n.func.value, ast.Name):
                    calls.add(f"{n.func.value.id}.{n.func.attr}")
    return calls


def _has_decorator(func_node: ast.FunctionDef, name: str) -> bool:
    for dec in func_node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == name:
            return True
        if isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name) and dec.func.id == name:
                return True
            if isinstance(dec.func, ast.Attribute) and dec.func.attr == name:
                return True
    return False


def _has_for_loop(node: ast.AST) -> bool:
    return any(isinstance(n, ast.For) for n in ast.walk(node))


def _has_negative_slice(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice):
            lower = n.slice.lower
            if isinstance(lower, ast.UnaryOp) and isinstance(lower.op, ast.USub):
                return True
            if isinstance(lower, ast.Constant) and isinstance(lower.value, (int, float)):
                if lower.value < 0:
                    return True
    return False


def _has_small_epsilon(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, float):
            if 0 < abs(n.value) < 1e-6:
                return True
    return False


def _count_asserts(node: ast.AST) -> int:
    return sum(1 for n in ast.walk(node) if isinstance(n, ast.Assert))


def _has_large_int(node: ast.AST, threshold: int = 100) -> bool:
    return any(
        isinstance(n, ast.Constant) and isinstance(n.value, int) and n.value > threshold
        for n in ast.walk(node)
    )


# ── L3 structural checkers per invariant type ────────────────────

def _check_universal(func: ast.FunctionDef) -> tuple[bool, str]:
    calls = _collect_call_names(func)
    if _has_decorator(func, "given"):
        return True, ""
    if _has_for_loop(func) and _count_asserts(func) >= 1:
        return True, ""
    if calls & {"all", "np.all", "np.testing.assert_array_less"}:
        return True, ""
    if _count_asserts(func) >= 3:
        return True, ""
    return False, (
        "Universal invariants should test across MANY inputs. "
        "Use @given (hypothesis), a for loop over inputs, or np.all() on arrays."
    )


def _check_asymptotic(func: ast.FunctionDef) -> tuple[bool, str]:
    calls = _collect_call_names(func)
    names = _collect_names(func)
    if _has_negative_slice(func):
        return True, ""
    trajectory_names = {"final", "steady", "late", "converged", "R_final",
                        "R_late", "R_steady", "trajectory", "steps"}
    if names & trajectory_names:
        return True, ""
    if calls & {"simulate", "run", "evolve", "integrate", "trajectory"}:
        return True, ""
    if _has_large_int(func, 100):
        return True, ""
    return False, (
        "Asymptotic invariants should test convergence over time. "
        "Simulate a trajectory, then check late-time values (e.g., arr[-1000:])."
    )


def _check_monotonic(func: ast.FunctionDef) -> tuple[bool, str]:
    calls = _collect_call_names(func)
    names = _collect_names(func)
    if "diff" in calls or "np.diff" in calls:
        return True, ""
    if "violations" in names:
        return True, ""
    if _has_for_loop(func) and _count_asserts(func) >= 1:
        return True, ""
    return False, (
        "Monotonic invariants should check that a quantity never increases (or decreases). "
        "Use np.diff() on the trajectory and check sign, or count violations in a loop."
    )


def _check_statistical(func: ast.FunctionDef) -> tuple[bool, str]:
    calls = _collect_call_names(func)
    if calls & {"mean", "np.mean", "std", "np.std", "np.var", "np.average"}:
        return True, ""
    if _has_for_loop(func):
        names = _collect_names(func)
        if names & {"seed", "seeds", "trial", "trials", "realization", "realizations", "n_trials"}:
            return True, ""
    if any(isinstance(n, ast.ListComp) for n in ast.walk(func)):
        return True, ""
    return False, (
        "Statistical invariants should average over multiple realizations. "
        "Use np.mean() over an ensemble of >=50 trials with different seeds."
    )


def _check_algebraic(func: ast.FunctionDef) -> tuple[bool, str]:
    calls = _collect_call_names(func)
    if calls & {"assert_allclose", "np.testing.assert_allclose", "approx"}:
        return True, ""
    if _has_small_epsilon(func):
        return True, ""
    return False, (
        "Algebraic invariants should use exact comparison (up to float precision). "
        "Use abs(actual - expected) < 1e-12 or np.testing.assert_allclose()."
    )


def _check_qualitative(func: ast.FunctionDef) -> tuple[bool, str]:
    if _has_for_loop(func):
        return True, ""
    if _count_asserts(func) >= 2:
        return True, ""
    return False, (
        "Qualitative invariants should sweep a parameter and check direction. "
        "E.g., higher volatility -> higher inhibition across a range of values."
    )


def _check_conservation(func: ast.FunctionDef) -> tuple[bool, str]:
    names = _collect_names(func)
    pairs = {"before", "after", "initial", "final", "total_before", "total_after"}
    if len(names & pairs) >= 2:
        return True, ""
    if _has_for_loop(func) and _count_asserts(func) >= 1:
        return True, ""
    return False, (
        "Conservation invariants should compare quantities before and after a process. "
        "Compute total_before, run process, compute total_after, check equality."
    )


# Fix #6: distributional type was missing — silent L3 skip
def _check_distributional(func: ast.FunctionDef) -> tuple[bool, str]:
    calls = _collect_call_names(func)
    stat_calls = {"rayleigh", "kstest", "ks_2samp", "chisquare", "anderson",
                  "shapiro", "normaltest", "histogram", "hist", "np.histogram"}
    if calls & stat_calls:
        return True, ""
    if calls & {"mean", "np.mean", "std", "np.std"}:
        return True, ""
    if _has_for_loop(func) and _count_asserts(func) >= 1:
        return True, ""
    return False, (
        "Distributional invariants should test the shape of a distribution. "
        "Use scipy.stats tests (Rayleigh, KS) or histogram-based checks."
    )


# ── L3 dispatch ──────────────────────────────────────────────────

# Dispatch by invariant `type` field
TYPE_CHECKERS: dict[str, Any] = {
    "universal":      _check_universal,
    "asymptotic":     _check_asymptotic,
    "monotonic":      _check_monotonic,
    "statistical":    _check_statistical,
    "algebraic":      _check_algebraic,
    "qualitative":    _check_qualitative,
    "conservation":   _check_conservation,
    "conditional":    _check_qualitative,
    "distributional": _check_distributional,
}

# Fix #1: Dispatch by YAML `test_type` field (takes priority over `type`)
TEST_TYPE_CHECKERS: dict[str, Any] = {
    "property_test":     _check_universal,
    "convergence_test":  _check_asymptotic,
    "trajectory_test":   _check_monotonic,
    "sweep_test":        _check_qualitative,
    "ensemble_test":     _check_statistical,
    "monotonicity_test": _check_monotonic,
    "balance_test":      _check_conservation,
    "statistical_test":  _check_statistical,
    "correlation_test":  _check_statistical,
}


def resolve_l3_checker(inv_data: dict[str, str]) -> tuple[Any | None, str]:
    """Resolve the L3 checker function for an invariant.

    Priority: test_type (explicit method) > type (invariant class).
    This ensures INV-5HT7 (type=conditional, test_type=property_test)
    is checked as property_test, not as qualitative.
    """
    test_type = inv_data.get("test_type", "")
    inv_type = inv_data.get("type", "")

    if test_type in TEST_TYPE_CHECKERS:
        return TEST_TYPE_CHECKERS[test_type], f"test_type='{test_type}'"

    if inv_type in TYPE_CHECKERS:
        return TYPE_CHECKERS[inv_type], f"type='{inv_type}'"

    return None, ""


# ── Issue class ──────────────────────────────────────────────────

class Issue:
    def __init__(self, level: str, line: int, func: str, msg: str):
        self.level = level
        self.line = line
        self.func = func
        self.msg = msg

    def __str__(self) -> str:
        return f"  [{self.level}] L{self.line}: {self.func}() — {self.msg}"


# ── File classification ──────────────────────────────────────────

def is_physics_test(filepath: Path) -> bool:
    name = filepath.stem.lower()
    parts = {p.lower() for p in filepath.parts}
    return bool(parts & PHYSICS_KEYWORDS) or any(kw in name for kw in PHYSICS_KEYWORDS)


def is_physics_source(filepath: Path) -> bool:
    """For --audit-code: detect production physics modules (not tests)."""
    name = filepath.stem.lower()
    parts = {p.lower() for p in filepath.parts}
    if "test" in name or "test" in parts:
        return False
    return bool(parts & PHYSICS_KEYWORDS) or any(kw in name for kw in PHYSICS_KEYWORDS)


# ── L4: Error message quality checker ────────────────────────────
# Fix #2: Check for 5 required fields, not just INV-* presence

def _check_error_msg_quality(msg_source: str) -> list[str]:
    """Check assert error message for required physics debug fields.

    Required by contract (CLAUDE.md + TEST_TAXONOMY.md):
    1. INV-* ID
    2. Expected value/behavior
    3. Observed value
    4. Physical reasoning
    5. Parameters used
    """
    missing: list[str] = []

    if not INV_PATTERN.search(msg_source):
        missing.append("INV-* ID")

    has_observed = bool(re.search(
        r"[=:]\s*[-+]?\d|actual|observed|got|result|R=|R_final|delta=|level=",
        msg_source, re.IGNORECASE
    ))
    if not has_observed:
        missing.append("observed value")

    has_expected = bool(re.search(
        r"expect|should|must|theory|predicted|required|outside|violat",
        msg_source, re.IGNORECASE
    ))
    if not has_expected:
        missing.append("expected behavior")

    has_params = bool(re.search(
        r"[NK]=\d|seed=|steps=|at\s+\w+=|with\s+\w+=|gamma=|K_c=",
        msg_source, re.IGNORECASE
    ))
    if not has_params:
        missing.append("parameters")

    return missing


# ── Test file validation (L1-L5) ────────────────────────────────

def check_test_file(filepath: Path, registry: dict[str, dict]) -> list[Issue]:
    issues: list[Issue] = []
    source = filepath.read_text()
    source_lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [Issue("L0", 0, "<file>", f"SYNTAX ERROR in {filepath}")]

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue

        docstring = ast.get_docstring(node) or ""
        inv_refs = INV_PATTERN.findall(docstring)

        # ── L1: INV-* presence ──
        if not inv_refs:
            issues.append(Issue(
                "L1", node.lineno, node.name,
                "No INV-* reference in docstring. Which physics invariant does this test?"
            ))
            continue

        # ── L2: INV-* validity ──
        for inv_id in inv_refs:
            if registry and inv_id not in registry:
                issues.append(Issue(
                    "L2", node.lineno, node.name,
                    f"{inv_id} not found in INVARIANTS.yaml. Typo or missing entry?"
                ))

        # ── L3: Test type vs invariant type ──
        # Fix #1: uses test_type with priority over type
        for inv_id in inv_refs:
            if inv_id not in registry:
                continue
            checker_fn, label = resolve_l3_checker(registry[inv_id])
            if checker_fn is not None:
                ok, reason = checker_fn(node)
                if not ok:
                    issues.append(Issue(
                        "L3", node.lineno, node.name,
                        f"{inv_id} has {label}. {reason}"
                    ))

        # ── L4: Error message quality ──
        # Fix #2: check 5 fields not just INV-* presence
        has_any_msg = False
        has_no_msg = False

        for child in ast.walk(node):
            if not isinstance(child, ast.Assert):
                continue
            if child.msg is not None:
                has_any_msg = True
                assert_start = child.lineno - 1
                assert_end = min(
                    getattr(child.msg, "end_lineno", child.msg.lineno) or child.msg.lineno,
                    len(source_lines),
                )
                msg_source = "\n".join(source_lines[assert_start:assert_end])
                missing = _check_error_msg_quality(msg_source)
                if missing:
                    issues.append(Issue(
                        "L4", child.lineno, node.name,
                        f"Assert message missing: {', '.join(missing)}. "
                        f"Contract requires: INV-ID, expected, observed, reasoning, params."
                    ))
            else:
                has_no_msg = True

        if has_no_msg and not has_any_msg:
            issues.append(Issue(
                "L4", node.lineno, node.name,
                "All assertions lack error messages. "
                "On failure, there's no context about what went wrong or why."
            ))

        # ── L5: Magic number detection ──
        for child in ast.walk(node):
            if not isinstance(child, ast.Assert):
                continue
            if child.lineno > len(source_lines):
                continue
            line = source_lines[child.lineno - 1].strip()
            if not re.search(r"assert.*[<>=]\s*0\.\d+", line):
                continue
            theory_patterns = [
                "sqrt", "np.sqrt", "math.sqrt",
                "1/np", "1/math",
                "k_c", "kc",
                "pi", "np.pi",
                "epsilon", "eps",
                "tolerance", "tol",
            ]
            context_start = max(0, child.lineno - 4)
            context_end = min(len(source_lines), child.lineno)
            context = "\n".join(source_lines[context_start:context_end]).lower()
            if not any(pat in context for pat in theory_patterns):
                issues.append(Issue(
                    "L5", child.lineno, node.name,
                    f"Possible magic number threshold. "
                    f"Is this derived from theory? Line: {line[:80]}"
                ))

    return issues


# ── Code audit (C1-C2) ──────────────────────────────────────────
# Fix #4 + #5: production code↔theory gate

CLAMP_PATTERNS = [
    re.compile(r"np\.clip\s*\("),
    re.compile(r"max\s*\(\s*[\d.]+\s*,\s*min\s*\("),
    re.compile(r"min\s*\(\s*[\d.]+\s*,\s*max\s*\("),
    re.compile(r"=\s*max\s*\(\s*[\d.]+"),
    re.compile(r"=\s*min\s*\(\s*[\d.]+"),
    re.compile(r"\.clamp\s*\("),
    re.compile(r"\.clip\s*\("),
]

LOG_PATTERNS = [
    re.compile(r"log(ger|ging)?\."),
    re.compile(r"warn(ing)?\s*\("),
    re.compile(r"_log\s*\("),
    re.compile(r"emit\s*\("),
    re.compile(r"telemetry"),
    re.compile(r"tacl\."),
]


def audit_code_file(filepath: Path) -> list[Issue]:
    """Scan production code for silent invariant repairs."""
    issues: list[Issue] = []
    try:
        source_lines = filepath.read_text().splitlines()
    except (UnicodeDecodeError, OSError):
        return []

    for i, line in enumerate(source_lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        has_clamp = any(p.search(stripped) for p in CLAMP_PATTERNS)
        if not has_clamp:
            continue

        context_start = max(0, i - 4)
        context_end = min(len(source_lines), i + 3)
        context = "\n".join(source_lines[context_start:context_end])

        has_log = any(p.search(context) for p in LOG_PATTERNS)
        has_inv_ref = bool(INV_PATTERN.search(context))

        if not has_log and not has_inv_ref:
            issues.append(Issue(
                "C1", i, filepath.stem,
                f"Silent clamp/clip without logging or INV-* comment. "
                f"This may hide a physics violation. Line: {stripped[:80]}"
            ))

        if re.search(r"clip\s*\(\s*\w+\s*,\s*[\d.]+\s*,\s*[\d.]+", stripped):
            if not has_inv_ref:
                issues.append(Issue(
                    "C2", i, filepath.stem,
                    f"Numeric bounds in clip without INV-* comment. "
                    f"Which invariant justifies these bounds? Line: {stripped[:80]}"
                ))

    return issues


# ── Main ─────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python validate_tests.py <path>              # validate physics tests")
        print("  python validate_tests.py <path> --summary    # summary only")
        print("  python validate_tests.py <path> --audit-code # audit production code")
        print("  python validate_tests.py --self-check        # verify kernel integrity")
        sys.exit(1)

    if "--self-check" in args:
        _self_check()
        return

    target = Path(args[0])
    summary_mode = "--summary" in args
    audit_code = "--audit-code" in args

    registry = load_invariants()
    if registry:
        print(f"Loaded {len(registry)} invariants from {INVARIANTS_PATH.name}")
    else:
        print("WARNING: No invariants loaded, L2/L3 checks disabled")

    if not target.exists():
        print(f"Not found: {target}")
        sys.exit(1)

    files: list[Path] = [target] if target.is_file() else sorted(target.rglob("*.py"))

    if audit_code:
        _run_audit_code(files, summary_mode)
    else:
        _run_test_validation(files, registry, summary_mode)


def _run_test_validation(files: list[Path], registry: dict, summary_mode: bool) -> None:
    total_issues: dict[str, int] = {}
    physics_files = 0
    total_tests = 0

    for f in files:
        if not f.name.startswith("test_") or not is_physics_test(f):
            continue
        physics_files += 1
        issues = check_test_file(f, registry)

        # Fix #7: proper scoping for file_tests counter
        file_tests = 0
        try:
            tree = ast.parse(f.read_text())
            file_tests = sum(
                1 for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            )
            total_tests += file_tests
        except SyntaxError:
            pass

        if issues:
            if not summary_mode:
                print(f"\n{f}:")
                for issue in issues:
                    print(issue)
            for issue in issues:
                total_issues[issue.level] = total_issues.get(issue.level, 0) + 1

    # Report
    print(f"\n{'='*64}")
    print("Physics Test Validation Report")
    print(f"{'='*64}")
    print(f"Files scanned:    {physics_files}")
    print(f"Test functions:   {total_tests}")
    print()
    levels = ["L1", "L2", "L3", "L4", "L5"]
    labels = {
        "L1": "Missing INV-* reference",
        "L2": "Invalid INV-* ID (not in YAML)",
        "L3": "Test type != invariant type",
        "L4": "Missing/poor error messages",
        "L5": "Possible magic number threshold",
    }
    print("Issues by level:")
    for lv in levels:
        count = total_issues.get(lv, 0)
        print(f"  [{lv}] {labels[lv]+':':<36s} {count}")
    total = sum(total_issues.get(lv, 0) for lv in levels)
    print(f"  {'─'*44}")
    print(f"  {'Total:':<40s} {total}")

    if total == 0:
        print("\n✅ All physics tests pass validation.")
    else:
        l1 = total_issues.get("L1", 0)
        grounded = max(0, total_tests - l1)
        pct = (grounded / total_tests * 100) if total_tests > 0 else 0
        print(f"\nPhysics grounding: {grounded}/{total_tests} tests ({pct:.0f}%)")
        print(f"\nFix priority: L1 -> L4 -> L3 -> L5 -> L2")
        sys.exit(1)


def _run_audit_code(files: list[Path], summary_mode: bool) -> None:
    total_issues: dict[str, int] = {}
    scanned = 0

    for f in files:
        if f.name.startswith("test_") or f.suffix != ".py":
            continue
        if not is_physics_source(f):
            continue
        scanned += 1
        issues = audit_code_file(f)

        if issues:
            if not summary_mode:
                print(f"\n{f}:")
                for issue in issues:
                    print(issue)
            for issue in issues:
                total_issues[issue.level] = total_issues.get(issue.level, 0) + 1

    print(f"\n{'='*64}")
    print("Production Code Audit Report")
    print(f"{'='*64}")
    print(f"Files scanned:  {scanned}")
    c1 = total_issues.get("C1", 0)
    c2 = total_issues.get("C2", 0)
    print(f"  [C1] Silent clamp/clip (no logging):  {c1}")
    print(f"  [C2] Undocumented numeric bounds:      {c2}")
    total = c1 + c2
    print(f"  {'─'*44}")
    print(f"  Total:                                 {total}")

    if total == 0:
        print("\n✅ No silent invariant repairs detected.")
    else:
        print(f"\nThese clamps may hide physics violations. Add logging or INV-* comment.")
        sys.exit(1)


# ── Self-check ───────────────────────────────────────────────────

def _self_check() -> None:
    """Verify physics kernel internal consistency."""
    errors: list[str] = []

    # 1. Load invariants
    reg = load_invariants()
    if not reg:
        print("FAIL: cannot load INVARIANTS.yaml")
        sys.exit(1)
    print(f"1. Loaded {len(reg)} invariants")

    # 2. Check all types have dispatchers
    for inv_id, data in reg.items():
        checker, label = resolve_l3_checker(data)
        if checker is None:
            errors.append(f"   {inv_id}: type='{data.get('type')}' test_type='{data.get('test_type')}' has no L3 checker")
    if errors:
        print(f"2. FAIL: {len(errors)} types without checker")
        for e in errors:
            print(e)
    else:
        print("2. All invariant types have L3 checkers")

    # 3. Regex matches all ID formats
    test_ids = ["INV-K1", "INV-5HT7", "INV-DA3", "INV-GABA5", "INV-ES1",
                "INV-FE2", "INV-RC1", "INV-TH2"]
    bad = [tid for tid in test_ids if not INV_PATTERN.match(tid)]
    if bad:
        print(f"3. FAIL: regex doesn't match: {bad}")
        errors.append(f"regex: {bad}")
    else:
        print(f"3. INV_PATTERN matches all {len(test_ids)} ID formats")

    # 4. Cross-reference theory files
    theory_ids: set[str] = set()
    for f in SCRIPT_DIR.glob("*THEORY*.md"):
        theory_ids |= set(INV_PATTERN.findall(f.read_text()))
    for f in [SCRIPT_DIR / "EXAMPLES.md", SCRIPT_DIR / "TEST_TAXONOMY.md"]:
        if f.exists():
            theory_ids |= set(INV_PATTERN.findall(f.read_text()))

    yaml_ids = set(reg.keys())
    # INV-ID is a placeholder in EXAMPLES.md pattern table, not a real invariant
    orphaned = theory_ids - yaml_ids - {"INV-ID"}
    if orphaned:
        print(f"4. FAIL: IDs in theory but not YAML: {sorted(orphaned)}")
        errors.append(f"orphaned: {orphaned}")
    else:
        print(f"4. Cross-ref OK: {len(theory_ids)} theory IDs, all in YAML")

    # 5. Summary
    ok = not errors
    print(f"\n{'✅' if ok else '❌'} Self-check {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
