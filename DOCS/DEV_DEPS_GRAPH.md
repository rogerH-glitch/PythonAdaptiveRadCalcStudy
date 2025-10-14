# Dependency Graph (DOT/PNG)

Generate a static module dependency graph for `src.*`.

## Quick start

```powershell
# From repo root
python -m tools.deps_graph --roots src tests --outdot audit/deps_graph.dot --png
```

Outputs:
- `audit/deps_graph.dot` (Graphviz)
- `audit/deps_graph.png` (if Graphviz `dot` is available)

> If PNG generation fails, install Graphviz or open the DOT file in VS Code with a Graphviz extension.
