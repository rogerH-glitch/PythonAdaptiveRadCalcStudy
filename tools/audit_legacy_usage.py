from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

LEGACY_PATTERNS = [
    r'\[\s*"xy"\s*\]\s*\[\s*"emitter"\s*\]',
    r'\[\s*"xy"\s*\]\s*\[\s*"receiver"\s*\]',
    r'\[\s*"xz"\s*\]\s*\[\s*"emitter"\s*\]',
    r'\[\s*"xz"\s*\]\s*\[\s*"receiver"\s*\]',
    r'\[\s*"corners3d"\s*\]\s*\[\s*"emitter"\s*\]',
    r'\[\s*"corners3d"\s*\]\s*\[\s*"receiver"\s*\]',
    # positional build_display_geom(args, result)
    r'build_display_geom\s*\(\s*[^,]+,\s*[^,]+\)',
]

SUSPECT_NAME_PATTERNS = [
    r'legacy', r'old', r'original', r'backup', r'copy', r'tmp', r'deprecated'
]

DISPLAY_GEOM_IMPORTS = {
    ("src.viz.display_geom", "build_display_geom"),
    ("src.viz.display_geom", "compute_panel_corners"),
    ("src.viz.display_geom", "PanelPose"),
}

def scan_regex(text: str, patterns: List[str]) -> List[Tuple[str, Tuple[int,int]]]:
    hits = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            hits.append((pat, m.span()))
    return hits

def ast_imports(py_path: Path) -> Set[str]:
    """Return a set of module names imported by this file (AST-based, no exec)."""
    try:
        node = ast.parse(py_path.read_text(encoding="utf-8", errors="ignore"), filename=str(py_path))
    except SyntaxError:
        return set()
    mods: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                if alias.name:
                    mods.add(alias.name)
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                mods.add(n.module)
    return mods

def ast_calls(py_path: Path) -> Set[str]:
    """Rough call name collector (qualnames best-effort)."""
    try:
        node = ast.parse(py_path.read_text(encoding="utf-8", errors="ignore"), filename=str(py_path))
    except SyntaxError:
        return set()
    calls: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            fn = n.func
            if isinstance(fn, ast.Name):
                calls.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                parts = []
                cur = fn
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                calls.add(".".join(reversed(parts)))
    return calls

def looks_suspect_name(path: Path) -> bool:
    lower = str(path.name).lower()
    return any(re.search(p, lower) for p in SUSPECT_NAME_PATTERNS)

def relative_module_name(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root)
    return ".".join(rel.with_suffix("").parts)

def is_python_file(p: Path) -> bool:
    return p.is_file() and p.suffix == ".py"

def collect_files(search_roots: List[Path]) -> List[Path]:
    files: List[Path] = []
    for root in search_roots:
        for p in root.rglob("*.py"):
            # skip venvs and build caches
            parts = {q.lower() for q in p.parts}
            if any(x in parts for x in {".venv", "venv", "__pycache__", ".git"}):
                continue
            files.append(p)
    return files

def build_import_graph(py_files: List[Path], project_root: Path) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}
    for f in py_files:
        mod = relative_module_name(project_root, f)
        imps = ast_imports(f)
        graph[mod] = imps
    return graph

def find_unreferenced_modules(graph: Dict[str, Set[str]], project_root: Path) -> List[str]:
    # modules under 'src.' considered project modules
    project_mods = {m for m in graph.keys() if m.startswith("src.")}
    referenced: Set[str] = set()
    for m, imps in graph.items():
        for imp in imps:
            if imp.startswith("src."):
                referenced.add(imp)
    # Heuristic: unreferenced = project mods that are never imported by others,
    # excluding entry points like src.cli, src.cli_* which may be invoked via __main__.
    unref = []
    for m in sorted(project_mods):
        if m.endswith(".__init__"):
            continue
        if m in {"src.cli", "src.cli_parser", "src.cli_results", "src.cli_cases"}:
            continue
        if m not in referenced:
            unref.append(m)
    return unref

def main():
    ap = argparse.ArgumentParser(description="Audit legacy usage and possible dead modules.")
    ap.add_argument("--paths", nargs="+", default=["src", "tests"], help="Paths to scan")
    ap.add_argument("--write", action="store_true", help="Write JSON/CSV report into ./audit/")
    args = ap.parse_args()

    roots = [Path(p).resolve() for p in args.paths]
    project_root = Path.cwd().resolve()
    py_files = collect_files(roots)

    legacy_hits = []
    imports_hits = []
    suspect_files = []
    call_hits = []

    for f in py_files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        hits = scan_regex(text, LEGACY_PATTERNS)
        for pat, span in hits:
            legacy_hits.append({"file": str(f), "pattern": pat, "pos": span})

        # import usage
        imps = ast_imports(f)
        for mod, name in DISPLAY_GEOM_IMPORTS:
            if mod in imps:
                imports_hits.append({"file": str(f), "import_module": mod})

        # suspicious filename
        if looks_suspect_name(f):
            suspect_files.append(str(f))

        # call names
        calls = ast_calls(f)
        if "build_display_geom" in calls:
            call_hits.append({"file": str(f), "call": "build_display_geom"})

    graph = build_import_graph(py_files, project_root)
    unref = find_unreferenced_modules(graph, project_root)

    report = {
        "summary": {
            "python_files_scanned": len(py_files),
            "legacy_hits": len(legacy_hits),
            "build_display_geom_calls": len(call_hits),
            "suspect_files": len(suspect_files),
            "unreferenced_modules": len(unref),
        },
        "legacy_hits": legacy_hits,
        "build_display_geom_calls": call_hits,
        "suspect_files": suspect_files,
        "unreferenced_modules": unref,
    }

    print(json.dumps(report["summary"], indent=2))

    if args.write:
        out_dir = Path("audit")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "legacy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        # CSV (file, category, detail)
        with (out_dir / "legacy_report.csv").open("w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["category", "file", "detail"])
            for h in legacy_hits:
                w.writerow(["legacy_pattern", h["file"], h["pattern"]])
            for h in call_hits:
                w.writerow(["build_display_geom_call", h["file"], h["call"]])
            for s in suspect_files:
                w.writerow(["suspect_filename", s, "name_pattern"])
            for m in unref:
                w.writerow(["unreferenced_module", m, ""])

        print(f"Wrote audit/legacy_report.json and audit/legacy_report.csv")

if __name__ == "__main__":
    main()
