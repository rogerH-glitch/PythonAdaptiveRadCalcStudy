#!/usr/bin/env python3
"""
Main entry point for the radiation view factor validation tool.

This script provides the command-line interface for calculating
view factors between rectangular surfaces using multiple methods.
"""

from src.cli import main  # single source of truth


def eval_case(method: str = "adaptive", eval_mode: str = "grid", geom=None, plot: bool = False):
    """
    Lightweight helper for tests to evaluate a case and return dense grid field.

    Returns a dict with keys:
      - vf_peak: float
      - vf_peak_yz: (float, float)
      - vf_field: np.ndarray (ny, nz)
      - grid_y: np.ndarray (ny,)
      - grid_z: np.ndarray (nz,)
    """
    # Build a minimal args-like object for src.cli.run_calculation
    class Args:
        pass
    args = Args()
    # Default geometry if not provided via geom dict
    em = (5.1, 2.1)
    rc = (5.1, 2.1)
    setback = 3.0
    angle = 0.0
    rotate_axis = "z"
    rotate_target = "emitter"
    angle_pivot = "toe"
    if isinstance(geom, dict):
        em = (float(geom.get("W_emitter", em[0])), float(geom.get("H_emitter", em[1])))
        rc = (float(geom.get("W_receiver", rc[0])), float(geom.get("H_receiver", rc[1])))
        setback = float(geom.get("setback", setback))
        angle = float(geom.get("angle", angle))
        rotate_axis = str(geom.get("rotate_axis", rotate_axis))
        rotate_target = str(geom.get("rotate_target", rotate_target))
        angle_pivot = str(geom.get("angle_pivot", angle_pivot))

    args.method = method
    args.emitter = em
    args.receiver = rc
    args.setback = setback
    args.angle = angle
    args.rotate_axis = rotate_axis
    args.rotate_target = rotate_target
    args.angle_pivot = angle_pivot
    args.align_centres = False
    args.receiver_offset = (0.0, 0.0)
    args.emitter_offset = (0.0, 0.0)
    args.plot = bool(plot)
    args.plot_3d = False
    args.plot_both = False
    args.outdir = "results"
    args.version = False
    # rc/eval mode mirrors
    args.eval_mode = eval_mode
    args.rc_mode = eval_mode
    # grid/search controls
    args.rc_grid_n = 41
    args.rc_search_rel_tol = 3e-3
    args.rc_search_max_iters = 50
    args.rc_search_multistart = 4
    args.rc_search_time_limit_s = 2.0
    args.rc_bounds = "auto"

    from src.cli import run_calculation
    from src.util.plot_payload import attach_grid_field
    from src.util.grid_tap import drain as _drain_grid
    import numpy as _np

    result = run_calculation(args)

    # Attach captured field from tap, if any
    try:
        tapped = _drain_grid()
        if tapped is not None:
            Y, Z, F = tapped
            attach_grid_field(result, Y, Z, F)
    except Exception:
        pass

    # Build return dict with explicit keys expected by tests/spec
    out = {
        "vf_peak": float(result.get("vf", result.get("vf_peak", 0.0))),
        "vf_peak_yz": (
            float(result.get("x_peak", result.get("y_peak", 0.0))),
            float(result.get("y_peak", result.get("z_peak", 0.0))),
        ),
    }

    F = result.get("F", None)
    Y = result.get("Y", None)
    Z = result.get("Z", None)
    if F is None or Y is None or Z is None:
        # Build a dense grid using the evaluator if not present
        try:
            from src.peak_locator import create_vf_evaluator
            from src.eval.watchdog import check_time as _check_time, Timeout as _Timeout
            import time as _time
            geom_cfg = {
                "receiver_offset": (0.0, 0.0),
                "rotate_axis": rotate_axis,
                "rotate_target": rotate_target,
                "angle_deg": angle,
            }
            # Fast-path point evaluator for grid mode (no peak search)
            fast_params = {
                'max_cells': 1500,
                'max_iters': 40,
                'rel_tol': 1e-2,
                'abs_tol': 1e-4,
                'peak_search': False,  # Disable peak search
                'multistart': 1,       # Single evaluation per point
            }
            vf_eval = create_vf_evaluator(method, em[0], em[1], rc[0], rc[1], setback, angle, geom_cfg, **fast_params)
            import numpy as _np
            from src.eval.cancel import CancelToken as _CancelToken
            ny = 41
            nz = 41
            gy = _np.linspace(-rc[0]/2.0, rc[0]/2.0, ny)
            gz = _np.linspace(-rc[1]/2.0, rc[1]/2.0, nz)
            Yg, Zg = _np.meshgrid(gy, gz, indexing='xy')
            Fg = _np.empty_like(Yg, dtype=float)
            start_ts = _time.time()
            every_n = 25
            time_limit_s = 3.0
            token = _CancelToken(timeout_s=time_limit_s)
            try:
                for j in range(Zg.shape[0]):
                    if token.expired():
                        print("[warn] grid timeout -> partial return")
                        Fg[j:, :] = float('nan')
                        break
                    for i in range(Yg.shape[1]):
                        v, _m = vf_eval(float(Yg[0, i]), float(Zg[j, 0]))
                        Fg[j, i] = float(v)
                        _check_time(start_ts, time_limit_s, every_n, j*Yg.shape[1] + i)
            except _Timeout as _e:
                Fg[j:, i:] = float('nan')
                print(f"[warn] grid evaluation timed out: {_e}")
            F, Y, Z = Fg, Yg, Zg
        except Exception:
            F, Y, Z = None, None, None

    if F is not None and Y is not None and Z is not None:
        out["vf_field"] = F
        # Derive 1-D axes from meshgrid (xy indexing implies Z.shape==(nz,ny))
        try:
            gy = _np.asarray(Y[0, :], float)
            gz = _np.asarray(Z[:, 0], float)
        except Exception:
            try:
                gy = _np.unique(_np.asarray(Y, float))
                gz = _np.unique(_np.asarray(Z, float))
            except Exception:
                gy, gz = None, None
        if gy is not None and gz is not None:
            out["grid_y"] = gy
            out["grid_z"] = gz
    # Mirror for compatibility with existing plotting/tests
    out.setdefault("F", out.get("vf_field"))
    out.setdefault("Y", Y)
    out.setdefault("Z", Z)
    return out

if __name__ == "__main__":
    raise SystemExit(main())