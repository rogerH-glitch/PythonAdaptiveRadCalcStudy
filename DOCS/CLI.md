# CLI Reference

## Command Line Interface

### Evaluation space
`--eval-mode {center,grid,search}` (default: `grid`)

- **center** – evaluate only at the receiver center.
- **grid** – evaluate a receiver grid (recommended).
- **search** – local-peak search over receiver points.

> If `--eval-mode` is omitted, the tool uses `grid` automatically.

## Quick examples

```powershell
# Basic (eval-mode defaults to grid)
vfcalc --emitter 2 1 --receiver 2 1 --setback 5 `
  --rotate-axis z --angle 20 --rotate-target emitter `
  --plot-both
```

## Plotting
- `--plot` → 2-D PNG (plan, elevation, heatmap)
- `--plot-3d` → 3-D HTML wireframe
- `--plot-both` → both

**Heatmaps always render**: if a field is not present (center/search), a fallback sampler or placeholder is used.

## Notes on pivots
- `angle-pivot toe` keeps the minimum setback by translating the rotated panel along +x so that the near face remains at `x=0` (emitter) or `x=setback` (receiver).

## Validation & common mistakes
- Sizes must be positive; `--setback` must be positive.
- You can provide either `--emitter-offset` **or** `--receiver-offset` (if both are given, the tool will use the receiver offset).
- If a plotting error occurs, the tool now falls back to a blank heatmap and annotates the figure with a helpful note.

### Bash examples

```bash
# Use backslashes to continue long lines in Bash
vfcalc --emitter 2.0 1.5 --receiver 2.0 1.5 \
  --setback 5.0 --receiver-offset 0.3 0.2 \
  --rotate-axis z --angle 20 --rotate-target emitter \
  --method adaptive --plot-both   # eval-mode defaults to grid
```
