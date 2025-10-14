#!/usr/bin/env python3
"""
Lightweight dependency auditor:
- Lists modules under src/ with no inbound imports (heuristic AST scan)
- Lists files under src/ never imported by tests/
- Finds duplicate helper function names across src/viz/

Outputs JSON and CSV to audit/legacy_report.json and audit/legacy_report.csv
"""
from __future__ import annotations

import ast
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
AUDIT = ROOT / "audit"


def find_python_files(base: Path) -> List[Path]:
    return [p for p in base.rglob("*.py") if p.is_file()]


def module_name_from_path(path: Path) -> str:
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        rel = path
    mod = str(rel).replace(os.sep, ".")
    if mod.endswith(".__init__.py"):
        mod = mod[: -len(".__init__.py")]
    elif mod.endswith(".py"):
        mod = mod[: -len(".py")]
    return mod


def parse_imports(py_path: Path) -> Set[str]:
    try:
        code = py_path.read_text(encoding="utf-8")
    except Exception:
        return set()
    try:
        tree = ast.parse(code)
    except Exception:
        return set()
    imps: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imps.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imps.add(node.module)
    return imps


def collect_inbound_graph(files: List[Path]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    outbound: Dict[str, Set[str]] = defaultdict(set)
    inbound: Dict[str, Set[str]] = defaultdict(set)
    for f in files:
        mod = module_name_from_path(f)
        imps = parse_imports(f)
        outbound[mod] |= imps
        for i in imps:
            inbound[i].add(mod)
    return inbound, outbound


def duplicate_helpers_in_viz(viz_dir: Path) -> Dict[str, List[str]]:
    funcs: Dict[str, List[str]] = defaultdict(list)
    for f in viz_dir.glob("*.py"):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                funcs[node.name].append(str(f.relative_to(ROOT)))
    return {name: paths for name, paths in funcs.items() if len(paths) > 1}


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    src_files = find_python_files(SRC)
    test_files = find_python_files(TESTS)

    inbound, outbound = collect_inbound_graph(src_files + test_files)
    src_modules = {module_name_from_path(p) for p in src_files}
    test_imports = set().union(*[parse_imports(p) for p in test_files])

    no_inbound = sorted([m for m in src_modules if len(inbound.get(m, set())) == 0])
    never_imported_by_tests = sorted([m for m in src_modules if m not in test_imports])

    dup_helpers = duplicate_helpers_in_viz(SRC / "viz")

    report = {
        "summary": {
            "python_files_scanned": len(src_files),
            "no_inbound_modules": len(no_inbound),
            "never_imported_by_tests": len(never_imported_by_tests),
            "duplicate_helpers_in_viz": len(dup_helpers),
        },
        "no_inbound_modules": no_inbound,
        "never_imported_by_tests": never_imported_by_tests,
        "duplicate_helpers": dup_helpers,
    }

    (AUDIT / "legacy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # CSV (simple)
    lines = [
        "category,file_or_name,detail",
    ]
    for m in no_inbound:
        lines.append(f"no_inbound,{m},")
    for m in never_imported_by_tests:
        lines.append(f"never_imported_by_tests,{m},")
    for name, paths in dup_helpers.items():
        lines.append(f"duplicate_helper,{name},'{'|'.join(paths)}'")
    (AUDIT / "legacy_report.csv").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", AUDIT / "legacy_report.json")
    print("Wrote:", AUDIT / "legacy_report.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())


