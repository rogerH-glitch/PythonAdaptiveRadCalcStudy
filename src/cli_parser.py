"""
Command-line argument parsing for the radiation view factor validation tool.

This module handles argument parsing, validation, and normalization
following the Single Responsibility Principle.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Sequence
import logging

# Public constant so tests and other modules can assert the default.
DEFAULT_EVAL_MODE = "grid"

# -----------------------------------------------------------------------------
# Parser that silently de-dupes duplicate option strings (e.g., --log-level)
# This prevents "conflicting option string" even if helpers add the same arg.
# -----------------------------------------------------------------------------
class NoDupArgumentParser(argparse.ArgumentParser):
    def add_argument(self, *names, **kwargs):
        existing = set()
        for act in self._actions:
            for s in getattr(act, "option_strings", []):
                existing.add(s)
        if any(name in existing for name in names):
            # Return the first existing action that matches any provided name
            for act in self._actions:
                if any(name in getattr(act, "option_strings", []) for name in names):
                    return act
        return super().add_argument(*names, **kwargs)


def _add_arg_unique(parser: argparse.ArgumentParser, *names, **kwargs):
    """Add argument only if not already present (prevents conflicts)."""
    for action in parser._actions:
        if any(name in getattr(action, 'option_strings', []) for name in names):
            return action
    return parser.add_argument(*names, **kwargs)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = NoDupArgumentParser(
        description="Local Python tool for radiation view factor validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  # 2-D plot only (single line)
  python main.py --method adaptive --emitter 5 2 --receiver 5 2 --setback 3 --eval-mode grid --plot

  # Multi-line (POSIX shells use backslash \ )
  python main.py \
    --method adaptive \
    --emitter 5 2 --receiver 5 2 --setback 3 \
    --eval-mode grid --plot

  # Multi-line (Windows PowerShell uses the backtick ` )
  python main.py `
    --method adaptive `
    --emitter 5 2 --receiver 5 2 --setback 3 `
    --eval-mode grid --plot

Default assumptions:
  - Surfaces face each other, centres aligned
  - Receiver dimensions default to emitter dimensions
  - Parallel orientation (angle = 0°)
        """
    )
    
    # Version flag
    parser.add_argument(
        '--version',
        action='store_true',
        help='Print version and exit'
    )
    
    # Core calculation method (required unless using --cases)
    parser.add_argument(
        '--method',
        choices=['analytical', 'fixedgrid', 'adaptive', 'montecarlo'],
        help='Calculation method to use (required unless using --cases)'
    )
    
    # Geometry parameters - emitter is required unless using --cases
    parser.add_argument(
        '--emitter',
        nargs=2,
        type=float,
        metavar=('W', 'H'),
        help='Emitter dimensions (width height) in metres'
    )
    parser.add_argument(
        '--receiver',
        nargs=2,
        type=float,
        metavar=('W', 'H'),
        help='Receiver dimensions (width height) in metres (default: same as emitter)'
    )
    parser.add_argument(
        '--setback',
        type=float,
        metavar='S',
        help='Setback distance in metres'
    )
    parser.add_argument(
        '--angle',
        type=float,
        default=0.0,
        metavar='DEG',
        help="Rotation angle in degrees (default: 0). Positive angles follow the right-hand rule: CCW when looking along the +axis."
    )
    parser.add_argument(
        '--rotate-target',
        choices=('emitter', 'receiver'),
        default='emitter',
        help='Which panel to rotate (currently only emitter rotation affects geometry)'
    )
    parser.add_argument(
        '--rotate-axis',
        choices=('z', 'y'),
        default='z',
        help='Rotation axis: \'z\' (yaw; x–y top view) or \'y\' (pitch; x–z side view)'
    )
    parser.add_argument(
        '--angle-pivot',
        choices=('toe', 'center'),
        default='toe',
        help='Rotation pivot: \'toe\' (edge pivot; preserves min setback at that edge) or \'center\' (rotate about centre)'
    )
    parser.add_argument(
        '--receiver-offset',
        nargs=2,
        type=float,
        metavar=('DY', 'DZ'),
        help='Centre-to-centre offset (receiver - emitter) in (y,z). Example: --receiver-offset 0.6 0.4'
    )
    parser.add_argument(
        '--emitter-offset',
        nargs=2,
        type=float,
        metavar=('DY', 'DZ'),
        help='Alternative offset entry: emitter centre shift. We convert it so that (receiver - emitter)=(dy,dz).'
    )
    parser.add_argument(
        '--align-centres',
        action='store_true',
        help='Force centres aligned in y,z (dy=dz=0) after rotation; overrides explicit offsets.'
    )
    
    # Test cases and output options
    parser.add_argument(
        '--cases',
        type=str,
        metavar='PATH',
        help='YAML test suite file path (optional)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate 2-D combined geometry+heatmap'
    )
    parser.add_argument(
        '--plot-3d',
        dest="plot_3d",
        action='store_true',
        help='Generate 3-D interactive HTML only'
    )
    parser.add_argument(
        '--plot-both',
        dest="plot_both",
        action='store_true',
        help='Generate both 2-D combined PNG and 3-D HTML'
    )
    # Default to "results" so tests that inspect parser defaults pass
    _add_arg_unique(
        parser,
        '--outdir',
        type=str,
        default='results',
        metavar='PATH',
        help='Directory to write outputs (CSV/plots). Stored under \'results/\' if relative.'
    )
    _add_arg_unique(
        parser,
        '--test-run',
        action='store_true',
        help='Route outputs into \'results/test_results/...\'. Automatically enabled under pytest.'
    )
    _add_arg_unique(
        parser,
        '--log-level',
        type=str,
        default='INFO',
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
        help='Logging level (default: INFO).'
    )
    
    # Analytical method tuning parameters
    analytical_group = parser.add_argument_group('Analytical method tuning')
    analytical_group.add_argument(
        '--analytical-nx',
        type=int,
        default=220,
        help='Emitter grid Nx for analytical approx (default: 220)'
    )
    analytical_group.add_argument(
        '--analytical-ny',
        type=int,
        default=220,
        help='Emitter grid Ny for analytical approx (default: 220)'
    )
    
    # Adaptive method tuning parameters
    adaptive_group = parser.add_argument_group('Adaptive method tuning')
    adaptive_group.add_argument(
        '--rel-tol',
        type=float,
        default=3e-3,
        help='Relative tolerance (default: 3e-3)'
    )
    adaptive_group.add_argument(
        '--abs-tol',
        type=float,
        default=1e-6,
        help='Absolute tolerance (default: 1e-6)'
    )
    adaptive_group.add_argument(
        '--max-depth',
        type=int,
        default=12,
        help='Maximum recursion depth (default: 12)'
    )
    adaptive_group.add_argument(
        '--min-cell-area-frac',
        type=float,
        default=1e-8,
        help='Minimum cell area fraction (default: 1e-8)'
    )
    adaptive_group.add_argument(
        '--max-cells',
        type=int,
        default=150000,
        help='Maximum number of cells (default: 150000)'
    )
    adaptive_group.add_argument(
        '--time-limit-s',
        type=float,
        default=60.0,
        help='Time limit in seconds (default: 60)'
    )
    adaptive_group.add_argument(
        '--init-grid',
        type=str,
        default='4x4',
        help='Initial grid size (default: 4x4)'
    )
    
    # Fixed grid method tuning parameters
    fixed_group = parser.add_argument_group('Fixed grid method tuning')
    fixed_group.add_argument(
        '--grid-nx',
        type=int,
        default=160,
        help='Grid points in x-direction (default: 160)'
    )
    fixed_group.add_argument(
        '--grid-ny',
        type=int,
        default=160,
        help='Grid points in y-direction (default: 160)'
    )
    fixed_group.add_argument(
        '--quadrature',
        choices=['centroid', '2x2'],
        default='centroid',
        help='Quadrature method (default: centroid)'
    )
    
    # Monte Carlo method tuning parameters
    mc_group = parser.add_argument_group('Monte Carlo method tuning')
    mc_group.add_argument(
        '--samples',
        type=int,
        default=300000,
        help='Number of samples (default: 300000)'
    )
    mc_group.add_argument(
        '--target-rel-ci',
        type=float,
        default=0.02,
        help='Target relative confidence interval (default: 0.02)'
    )
    mc_group.add_argument(
        '--max-iters',
        type=int,
        default=60,
        help='Maximum iterations (default: 60)'
    )
    mc_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    # Peak locator options
    peak_group = parser.add_argument_group('Peak locator options')
    # New clearer name
    peak_group.add_argument(
        '--eval-mode',
        choices=['center', 'grid', 'search'],
        default=DEFAULT_EVAL_MODE,
        help=f'Receiver evaluation mode: center (single point), grid (max over uniform grid), search (peak search) (default: {DEFAULT_EVAL_MODE})'
    )
    peak_group.add_argument(
        '--rc-mode',
        choices=['center', 'grid', 'search'],
        default='center',
        help='Receiver peak search mode (default: center)'
    )
    peak_group.add_argument(
        '--min-cells',
        type=int,
        default=16,
        help='Minimum number of cells before convergence (default: 16)'
    )
    peak_group.add_argument(
        '--rc-grid-n',
        type=int,
        default=21,
        help='Grid resolution for coarse sampling (default: 21)'
    )
    peak_group.add_argument(
        '--heatmap-n',
        type=int,
        default=None,
        help='Heatmap grid resolution (default: use rc-grid-n)'
    )
    peak_group.add_argument(
        '--heatmap-interp',
        choices=['nearest', 'bilinear', 'bicubic'],
        default='bilinear',
        help='Heatmap interpolation method (default: bilinear)'
    )
    peak_group.add_argument(
        '--debug-plots',
        action='store_true',
        help='Enable verbose plot debug logs (markers and 3D vertices). Off by default.'
    )
    peak_group.add_argument(
        '--heatmap-marker',
        choices=['adaptive', 'grid', 'both'],
        default='both',
        help='Heatmap marker mode: adaptive peak, grid argmax, or both'
    )
    peak_group.add_argument(
        '--rc-search-rel-tol',
        type=float,
        default=3e-3,
        help='Target relative improvement tolerance (default: 3e-3)'
    )
    peak_group.add_argument(
        '--rc-search-max-iters',
        type=int,
        default=200,
        help='Max local-optimizer iterations (default: 200)'
    )
    peak_group.add_argument(
        '--rc-search-multistart',
        type=int,
        default=8,
        help='Number of multi-start seeds (default: 8)'
    )
    peak_group.add_argument(
        '--rc-search-time-limit-s',
        type=float,
        default=10.0,
        help='Wall clock cap for search phase (default: 10.0)'
    )
    peak_group.add_argument(
        '--rc-bounds',
        choices=['auto', 'explicit'],
        default='auto',
        help='Bounds mode (default: auto)'
    )
    
    # General options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--log-level',
        choices=('WARNING', 'INFO', 'DEBUG'),
        default='WARNING',
        help='Logging level for this tool (third-party libs stay at WARNING unless DEBUG).'
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate parsed command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Raises:
        ValueError: If arguments are invalid or missing
    """
    # If using test cases file, emitter/setback not required
    if args.cases:
        if not os.path.isfile(args.cases):
            raise ValueError(f"Test cases file not found: {args.cases}")
        return
    
    # For direct geometry specification, method, emitter and setback are required
    if not args.method:
        raise ValueError("--method is required when not using --cases")
    
    if not args.emitter:
        raise ValueError("--emitter is required when not using --cases")
    
    if args.setback is None:
        raise ValueError("--setback is required when not using --cases")
    
    # Validate emitter dimensions
    emitter_width, emitter_height = args.emitter
    if emitter_width <= 0 or emitter_height <= 0:
        raise ValueError(f"Emitter dimensions must be positive, got {emitter_width} × {emitter_height}")
    
    # Validate receiver dimensions if provided
    if args.receiver:
        receiver_width, receiver_height = args.receiver
        if receiver_width <= 0 or receiver_height <= 0:
            raise ValueError(f"Receiver dimensions must be positive, got {receiver_width} × {receiver_height}")
    
    # Validate setback distance
    if args.setback <= 0:
        raise ValueError(f"Setback distance must be positive, got {args.setback}")
    
    # Validate tuning parameters
    if args.rel_tol <= 0:
        raise ValueError(f"Relative tolerance must be positive, got {args.rel_tol}")
    
    if args.abs_tol <= 0:
        raise ValueError(f"Absolute tolerance must be positive, got {args.abs_tol}")
    
    if args.max_depth < 1:
        raise ValueError(f"Maximum depth must be at least 1, got {args.max_depth}")
    
    if args.max_cells < 1:
        raise ValueError(f"Maximum cells must be at least 1, got {args.max_cells}")
    
    if args.time_limit_s <= 0:
        raise ValueError(f"Time limit must be positive, got {args.time_limit_s}")
    
    if args.grid_nx < 1 or args.grid_ny < 1:
        raise ValueError(f"Grid dimensions must be at least 1, got {args.grid_nx} × {args.grid_ny}")
    
    if args.samples < 1:
        raise ValueError(f"Number of samples must be at least 1, got {args.samples}")
    
    if not (0 < args.target_rel_ci < 1):
        raise ValueError(f"Target relative CI must be between 0 and 1, got {args.target_rel_ci}")
    
    if args.max_iters < 1:
        raise ValueError(f"Maximum iterations must be at least 1, got {args.max_iters}")


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Normalise inexpensive options, but DO NOT rewrite args.outdir.
    We keep the user-provided outdir verbatim; writers will resolve it.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Normalized arguments with defaults applied
    """
    logger = logging.getLogger(__name__)
    
    # If receiver dimensions not provided, use emitter dimensions
    if args.emitter and not args.receiver:
        args.receiver = args.emitter.copy()
        logger.debug(f"Receiver dimensions defaulted to emitter: {args.receiver}")
    
    # Set default outdir if not provided, but don't mutate the user's choice
    if not getattr(args, "outdir", None):
        args.outdir = "results"
    # Remember what the user typed for nice printing
    if not hasattr(args, "_outdir_user"):
        args._outdir_user = args.outdir
    
    # Parse init-grid format (e.g., "4x4" -> (4, 4))
    if 'x' in args.init_grid.lower():
        try:
            nx, ny = map(int, args.init_grid.lower().split('x'))
            args.init_grid_nx = nx
            args.init_grid_ny = ny
        except ValueError:
            raise ValueError(f"Invalid init-grid format: {args.init_grid}. Expected format: NxM")
    else:
        # Assume square grid if single number
        try:
            n = int(args.init_grid)
            args.init_grid_nx = n
            args.init_grid_ny = n
        except ValueError:
            raise ValueError(f"Invalid init-grid format: {args.init_grid}. Expected format: N or NxM")
    
    # decide a single internal plot mode
    if getattr(args, "plot_both", False):
        args._plot_mode = "both"
    elif getattr(args, "plot_3d", False):
        args._plot_mode = "3d"
    elif getattr(args, "plot", False):
        args._plot_mode = "2d"
    else:
        args._plot_mode = "none"
    
    return args


def map_eval_mode_args(args: argparse.Namespace) -> None:
    """Map deprecated --rc-mode to --eval-mode and keep both in sync.

    - If eval_mode not provided, take rc_mode (falls back to its default)
    - If both provided, prefer eval_mode but mirror into rc_mode for legacy code
    """
    eval_mode = getattr(args, 'eval_mode', None)
    rc_mode = getattr(args, 'rc_mode', None)

    # Check if user explicitly provided --rc-mode by checking if it differs from default
    rc_mode_explicit = rc_mode is not None and rc_mode != 'center'
    
    # If eval_mode is the default and rc_mode was explicitly provided, use rc_mode
    if eval_mode == DEFAULT_EVAL_MODE and rc_mode_explicit:
        setattr(args, 'eval_mode', rc_mode)
        # Emit deprecation note similar to CLI entrypoint behaviour
        print("[deprecation] --rc-mode is deprecated; use --eval-mode.", file=sys.stderr)
    
    # Always mirror eval_mode back to rc_mode for downstream compatibility
    setattr(args, 'rc_mode', getattr(args, 'eval_mode'))


def build_parser():
    """
    Return the project CLI parser.
    Exposes DEFAULT_EVAL_MODE ('grid') for --eval-mode when not specified.
    """
    return create_parser()


def parse_args(argv: Optional[Sequence[str]] = None):
    """
    Parse args and perform basic validation, raising SystemExit with
    clear messages for common mistakes.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    def _dim_ok(dest: str, label: str):
        """
        Validate a two-number flag like --emitter W H or --receiver W H if provided.
        Treat absence as OK (defaults handled downstream).
        """
        vals = getattr(args, dest.replace("-", "_"), None)
        if vals is None:
            return
        # e.g. vals == [W, H] from nargs=2
        if isinstance(vals, (list, tuple)) and len(vals) == 2:
            w, h = vals
            try:
                w = float(w)
                h = float(h)
            except (TypeError, ValueError):
                parser.error(f"--{dest} expects two numbers (W H)")
            if w <= 0 or h <= 0:
                parser.error(f"--{dest} values must be > 0 (got {w}, {h})")
        # else: ignore (another component may parse/normalize)

    # Validate sizes
    _dim_ok("emitter", "emitter")
    _dim_ok("receiver", "receiver")

    # Treat missing/None setback as default 1.0 for validation purposes
    _sb = getattr(args, "setback", None)
    try:
        sb_val = 1.0 if _sb is None else float(_sb)
    except (TypeError, ValueError):
        parser.error("--setback must be a positive number")
    if sb_val <= 0:
        parser.error("--setback must be > 0")

    # Validate offsets: if both emitter-offset and receiver-offset are given, allow but warn via help text
    # (no hard error; downstream determines precedence). If you'd prefer hard error, uncomment below:
    # if args.emitter_offset and args.receiver_offset:
    #     parser.error("Specify only one of --emitter-offset or --receiver-offset.")

    return args
