"""Test that heatmap_core functions are array-safe."""
import numpy as np
import matplotlib.pyplot as plt
from src.viz.heatmap_core import draw_heatmap, render_receiver_heatmap


def test_draw_heatmap_never_truth_tests_arrays():
    """Test that draw_heatmap never triggers numpy truth-value errors."""
    fig, ax = plt.subplots()
    
    # Test with valid arrays
    Y = np.linspace(-2.55, 2.55, 21).reshape(1, -1).repeat(21, 0)
    Z = np.linspace(-1.05, 1.05, 21).reshape(-1, 1).repeat(21, 1)
    F = np.ones((21, 21)) * 0.1
    
    # Must not raise "truth value of an array" error
    draw_heatmap(ax, Y, Z, F, ypk=0.0, zpk=0.0)
    
    plt.close(fig)


def test_draw_heatmap_with_boolean_arrays():
    """Test that draw_heatmap works even with boolean arrays that would cause truth-value errors."""
    fig, ax = plt.subplots()
    
    # Create boolean arrays that would cause ValueError if used in boolean context
    Y = np.array([True, False, True]).reshape(1, -1).repeat(3, 0)
    Z = np.array([False, True, False]).reshape(-1, 1).repeat(3, 1)
    F = np.array([0.1, 0.2, 0.3]).reshape(-1, 1).repeat(3, 1)
    
    # Must not raise "truth value of an array" error even with boolean arrays
    draw_heatmap(ax, Y, Z, F, ypk=0.0, zpk=0.0)
    
    plt.close(fig)


def test_draw_heatmap_with_empty_arrays():
    """Test that draw_heatmap handles empty arrays gracefully."""
    fig, ax = plt.subplots()
    
    Y = np.array([])
    Z = np.array([])
    F = np.array([])
    
    # Must not raise "truth value of an array" error and should show "No field data available"
    draw_heatmap(ax, Y, Z, F, ypk=0.0, zpk=0.0)
    
    plt.close(fig)


def test_draw_heatmap_with_mismatched_shapes():
    """Test that draw_heatmap handles mismatched array shapes gracefully."""
    fig, ax = plt.subplots()
    
    Y = np.zeros((5, 5))
    Z = np.ones((4, 5))  # Different shape
    F = np.zeros((5, 5))
    
    # Must not raise "truth value of an array" error and should show "No field data available"
    draw_heatmap(ax, Y, Z, F, ypk=0.0, zpk=0.0)
    
    plt.close(fig)


def test_render_receiver_heatmap_with_boolean_arrays():
    """Test that render_receiver_heatmap works with boolean arrays."""
    fig, ax = plt.subplots()
    
    # Create boolean arrays
    Y = np.array([True, False, True]).reshape(1, -1).repeat(3, 0)
    Z = np.array([False, True, False]).reshape(-1, 1).repeat(3, 1)
    F = np.array([0.1, 0.2, 0.3]).reshape(-1, 1).repeat(3, 1)
    
    # Must not raise "truth value of an array" error
    render_receiver_heatmap(ax, Y, Z, F, ypk=0.0, zpk=0.0)
    
    plt.close(fig)
