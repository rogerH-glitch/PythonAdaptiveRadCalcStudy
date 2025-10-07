import numpy as np
from unittest.mock import patch, MagicMock
from src.cli import main_with_args
from src.util.grid_tap import capture, drain
from src.util.plot_payload import attach_grid_field

def test_auto_attach_grid_data():
    """Test that grid data is automatically attached when available."""
    # Create some test grid data
    Y, Z = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    F = np.ones_like(Y) * 0.5
    
    # Test the attach function directly first
    result = {"method": "analytical", "vf": 0.5}
    attach_grid_field(result, Y, Z, F)
    assert "grid_data" in result
    assert "Y" in result
    assert "Z" in result
    assert "F" in result
    assert np.array_equal(result["Y"], Y)
    
    # Test the grid tap capture and drain
    capture(Y, Z, F)
    tapped = drain()
    assert tapped is not None
    Y_out, Z_out, F_out = tapped
    assert np.array_equal(Y_out, Y)
    assert np.array_equal(Z_out, Z)
    assert np.array_equal(F_out, F)
    
    # After drain, should be empty
    assert drain() is None

def test_no_attach_for_center_mode():
    """Test that grid data is not attached for center mode."""
    args = MagicMock()
    args.method = "analytical"
    args.emitter = (5.0, 2.0)
    args.receiver = (5.0, 2.0)
    args.setback = 1.0
    args.eval_mode = "center"
    args.rc_mode = "center"
    args.plot = False
    args.outdir = "test_results"
    args._outdir_user = "test_results"
    args.verbose = False
    args.log_level = "WARNING"
    
    # Create some test grid data
    Y, Z = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    F = np.ones_like(Y) * 0.5
    
    # Capture the grid data
    capture(Y, Z, F)
    
    # Mock the calculation
    with patch('src.cli.run_calculation') as mock_calc:
        mock_result = {"method": "analytical", "vf": 0.5, "x_peak": 0.0, "y_peak": 0.0}
        mock_calc.return_value = mock_result
        
        # Mock the results functions
        with patch('src.cli.print_results'), patch('src.cli.save_and_report_csv'):
            # Run the main function
            main_with_args(args)
            
            # Check that the grid data was NOT attached for center mode
            assert "grid_data" not in mock_result
            assert "Y" not in mock_result
            assert "Z" not in mock_result
            assert "F" not in mock_result
            
            # Check that the grid tap was NOT drained
            assert drain() is not None  # Should still have the captured data
