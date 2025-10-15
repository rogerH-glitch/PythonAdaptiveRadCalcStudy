# Contributing Guidelines

## Manual Reproduction Steps

When reproducing issues or testing changes manually, **always use the timeout runner** to prevent hanging:

### Windows (PowerShell)
```powershell
# Basic usage
.\scripts\run_case_with_timeout.ps1 -Timeout 5.0 -Payload '{"method":"adaptive","emitter":[5.1,2.1],"receiver":[5.1,2.1],"setback":3.0,"eval_mode":"grid"}'

# With custom timeout
.\scripts\run_case_with_timeout.ps1 -Timeout 10.0 -Payload '{"method":"adaptive","emitter":[5.1,2.1],"receiver":[5.1,2.1],"setback":3.0,"eval_mode":"grid","rc_grid_n":31}'
```

### Unix/Linux/macOS
```bash
# Basic usage
./scripts/run_case_with_timeout.sh --timeout 5.0 --payload '{"method":"adaptive","emitter":[5.1,2.1],"receiver":[5.1,2.1],"setback":3.0,"eval_mode":"grid"}'

# With custom timeout
./scripts/run_case_with_timeout.sh --timeout 10.0 --payload '{"method":"adaptive","emitter":[5.1,2.1],"receiver":[5.1,2.1],"setback":3.0,"eval_mode":"grid","rc_grid_n":31}'
```

### Exit Codes
- `0`: Success
- `124`: Timeout (process was terminated)
- `1`: Error (invalid arguments, missing dependencies, etc.)

## Why Use the Timeout Runner?

The timeout runner prevents:
- **Silent hangs** during grid evaluation
- **Infinite loops** in peak search algorithms
- **Resource exhaustion** from runaway processes
- **CI/CD pipeline timeouts** in automated testing

## Direct Python Usage

If you need to run the timeout runner directly:

```bash
python eng/run_with_timeout.py --timeout 5.0 --payload '{"method":"adaptive",...}'
```

## Testing Guidelines

1. **Always use timeout runners** for manual reproduction
2. **Test both zero and non-zero angles** to verify geometry handling
3. **Verify true-scale rendering** in all plot outputs
4. **Check that heatmap stars align** with field peaks
5. **Ensure setback preservation** under yaw rotation

## Plotting Principles

- **Never massage visuals** - render computed geometry/fields as-is
- **Use true-scale aspect ratios** for accurate representation
- **Maintain faithful peak marking** in heatmaps
- **Preserve geometric relationships** in all views
