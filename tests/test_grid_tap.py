import numpy as np
from src.util.grid_tap import capture, drain

def test_capture_and_drain():
    # Drain any prior captured data from other tests
    _ = drain()
    Y, Z = np.zeros((3,4)), np.zeros((3,4))
    F = np.ones((3,4))
    
    # Initially no data
    assert drain() is None
    
    # Capture data
    capture(Y, Z, F)
    
    # Drain should return the captured data
    result = drain()
    assert result is not None
    Y_out, Z_out, F_out = result
    assert Y_out is Y and Z_out is Z and F_out is F
    
    # After drain, should be empty again
    assert drain() is None

def test_capture_overwrites():
    Y1, Z1 = np.zeros((2,3)), np.zeros((2,3))
    F1 = np.ones((2,3))
    Y2, Z2 = np.ones((4,5)), np.ones((4,5))
    F2 = np.zeros((4,5))
    
    # Capture first set
    capture(Y1, Z1, F1)
    
    # Capture second set (should overwrite)
    capture(Y2, Z2, F2)
    
    # Drain should return the second set
    result = drain()
    assert result is not None
    Y_out, Z_out, F_out = result
    assert Y_out is Y2 and Z_out is Z2 and F_out is F2
