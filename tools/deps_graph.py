from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from typing import Dict, Set, List, Tuple

def is_code_file(p: Path) -> bool:
    return p.suffix == ".py" and all(s not in {".git", ".venv", "venv", "__pycache__"} for s in p.parts)

def module_name(repo_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(repo_root).with_suffix("")
    return ".".join(rel.parts)

def ast_imports(py_path: Path) -> Set[str]:
    try:
        node = ast.parse(py_path.read_text(encoding="utf-8", errors="ignore"), filename=str(py_path))
    except SyntaxError:
        return set()
    mods: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name:
                    mods.add(a.name)
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
    return mods

def build_graph(repo_root: Path, roots: List[Path]) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}
    files: List[Path] = []
    for root in roots:
        for p in root.rglob("*.py"):
            if is_code_file(p):
                files.append(p)
    for f in files:
        mod = module_name(repo_root, f)
        graph[mod] = ast_imports(f)
    return graph

def filter_project_nodes(graph: Dict[str, Set[str]], prefix: str) -> Dict[str, Set[str]]:
    # keep only project modules and their edges within prefix
    keep = {m for m in graph if m.startswith(prefix)}
    out: Dict[str, Set[str]] = {}
    for m in keep:
        out[m] = {i for i in graph[m] if i.startswith(prefix)}
    return out

def to_dot(graph: Dict[str, Set[str]]) -> str:
    lines = ["digraph G {", '  rankdir=LR;', '  node [shape=box, fontsize=10];']
    nodes = sorted(graph.keys())
    for n in nodes:
        label = n.replace("src.", "")
        lines.append(f'  "{n}" [label="{label}"];')
    for src, imps in graph.items():
        for dst in sorted(imps):
            lines.append(f'  "{src}" -> "{dst}";')
    lines.append("}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Generate a DOT/PNG dependency graph for project modules.")
    ap.add_argument("--prefix", default="src.", help="Module prefix to include (default: src.)")
    ap.add_argument("--roots", nargs="+", default=["src", "tests"], help="Directories to scan")
    ap.add_argument("--outdot", default="audit/deps_graph.dot", help="Path to write DOT")
    ap.add_argument("--png", action="store_true", help="Also render PNG (requires graphviz installed)")
    ap.add_argument("--outpng", default="audit/deps_graph.png", help="Path to write PNG")
    args = ap.parse_args()

    repo_root = Path.cwd().resolve()
    roots = [repo_root / r for r in args.roots]
    graph = build_graph(repo_root, roots)
    proj = filter_project_nodes(graph, args.prefix)

    outdot = Path(args.outdot)
    outdot.parent.mkdir(parents=True, exist_ok=True)
    outdot.write_text(to_dot(proj), encoding="utf-8")
    print(f"Wrote {outdot}")

    if args.png:
        try:
            import shutil, subprocess
            if shutil.which("dot") is None:
                print("Graphviz 'dot' not found on PATH — skipping PNG. Install Graphviz or omit --png.")
            else:
                subprocess.run(["dot", "-Tpng", str(outdot), "-o", args.outpng], check=True)
                print(f"Wrote {args.outpng}")
        except Exception as e:
            print("Could not render PNG with graphviz 'dot'. Skipping PNG.")
            print(f"Reason: {e}")

if __name__ == "__main__":
    main()
