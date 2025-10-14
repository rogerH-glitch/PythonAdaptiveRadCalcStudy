"""Test that plotting functions never trigger numpy truth-value errors."""
import numpy as np
from src.viz.plots import plot_geometry_and_heatmap


def test_plot_geometry_and_heatmap_never_truth_tests_arrays(tmp_path):
    """Test that plot_geometry_and_heatmap never triggers numpy truth-value errors."""
    result = {
        "We": 5.1, "He": 2.1, "Wr": 5.1, "Hr": 2.1,
        "emitter_center": (0.0, 0.0, 0.0),
        "receiver_center": (0.0, 0.6, 0.4),
        "Y": np.linspace(-2.55, 2.55, 21).reshape(1, -1).repeat(21, 0),
        "Z": np.linspace(-1.05, 1.05, 21).reshape(-1, 1).repeat(21, 1),
        "F": np.ones((21, 21)) * 0.1,
        "y_peak": 0.0, "z_peak": 0.0
    }
    out = tmp_path / "ok.png"
    # Must not raise "truth value of an array" error
    plot_geometry_and_heatmap(result=result, eval_mode="grid", method="adaptive",
                              setback=3.0, out_png=str(out))
    assert out.exists()


def test_plot_geometry_and_heatmap_with_boolean_arrays(tmp_path):
    """Test that plotting works even with boolean arrays that would cause truth-value errors."""
    result = {
        "We": 5.1, "He": 2.1, "Wr": 5.1, "Hr": 2.1,
        "emitter_center": (0.0, 0.0, 0.0),
        "receiver_center": (0.0, 0.6, 0.4),
        "Y": np.array([True, False, True]).reshape(1, -1).repeat(3, 0),
        "Z": np.array([False, True, False]).reshape(-1, 1).repeat(3, 1),
        "F": np.array([0.1, 0.2, 0.3]).reshape(-1, 1).repeat(3, 1),
        "y_peak": 0.0, "z_peak": 0.0
    }
    out = tmp_path / "boolean_arrays.png"
    # Must not raise "truth value of an array" error even with boolean arrays
    plot_geometry_and_heatmap(result=result, eval_mode="grid", method="adaptive",
                              setback=3.0, out_png=str(out))
    assert out.exists()


def test_plot_geometry_and_heatmap_with_empty_arrays(tmp_path):
    """Test that plotting handles empty arrays gracefully."""
    result = {
        "We": 5.1, "He": 2.1, "Wr": 5.1, "Hr": 2.1,
        "emitter_center": (0.0, 0.0, 0.0),
        "receiver_center": (0.0, 0.6, 0.4),
        "Y": np.array([]),
        "Z": np.array([]),
        "F": np.array([]),
        "y_peak": 0.0, "z_peak": 0.0
    }
    out = tmp_path / "empty_arrays.png"
    # Must not raise "truth value of an array" error and should show "No field data available"
    plot_geometry_and_heatmap(result=result, eval_mode="grid", method="adaptive",
                              setback=3.0, out_png=str(out))
    assert out.exists()


def test_plot_geometry_and_heatmap_with_mismatched_shapes(tmp_path):
    """Test that plotting handles mismatched array shapes gracefully."""
    result = {
        "We": 5.1, "He": 2.1, "Wr": 5.1, "Hr": 2.1,
        "emitter_center": (0.0, 0.0, 0.0),
        "receiver_center": (0.0, 0.6, 0.4),
        "Y": np.zeros((5, 5)),
        "Z": np.ones((4, 5)),  # Different shape
        "F": np.zeros((5, 5)),
        "y_peak": 0.0, "z_peak": 0.0
    }
    out = tmp_path / "mismatched_shapes.png"
    # Must not raise "truth value of an array" error and should show "No field data available"
    plot_geometry_and_heatmap(result=result, eval_mode="grid", method="adaptive",
                              setback=3.0, out_png=str(out))
    assert out.exists()
