"""Test plotting flag detection and path logic."""
from types import SimpleNamespace
from src.cli import _want_plots


def test_want_plots_flags():
    """Test that _want_plots correctly detects plotting flags."""
    # Test individual flags
    args_2d = SimpleNamespace(plot=True, plot_3d=False, plot_both=False)
    assert _want_plots(args_2d) == (True, False)
    
    args_3d = SimpleNamespace(plot=False, plot_3d=True, plot_both=False)
    assert _want_plots(args_3d) == (False, True)
    
    # Test plot_both flag
    args_both = SimpleNamespace(plot=False, plot_3d=False, plot_both=True)
    assert _want_plots(args_both) == (True, True)
    
    # Test no plotting
    args_none = SimpleNamespace(plot=False, plot_3d=False, plot_both=False)
    assert _want_plots(args_none) == (False, False)
    
    # Test plot_both overrides individual flags
    args_override = SimpleNamespace(plot=True, plot_3d=True, plot_both=True)
    assert _want_plots(args_override) == (True, True)
