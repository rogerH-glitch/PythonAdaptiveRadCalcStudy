# scripts/test_timeout_wrapper.py
import json
import subprocess
import sys

def test_timeout_wrapper():
    """Test the timeout wrapper with a simple case."""
    payload = {
        "method": "adaptive",
        "emitter": [5.1, 2.1],
        "receiver": [5.1, 2.1],
        "setback": 3.0,
        "eval_mode": "grid",
        "rc_mode": "grid",
        "rc_grid_n": 15,
        "rc_search_time_limit_s": 1.5,
        "rc_search_rel_tol": 3e-3,
        "rc_search_max_iters": 50,
        "rc_search_multistart": 1,
        "rc_bounds": "auto",
        "plot": False,
        "plot_3d": False,
        "plot_both": False,
        "outdir": "results",
        "version": False
    }
    
    # Test the timeout runner directly
    cmd = [sys.executable, "eng/run_with_timeout.py", "--timeout", "3.0", "--payload", json.dumps(payload)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        print(f"Exit code: {result.returncode}")
        print(f"Output: {result.stdout.strip()}")
        if result.stderr:
            print(f"Error: {result.stderr.strip()}")
        
        # Parse the JSON output
        if result.returncode == 0:
            output_data = json.loads(result.stdout)
            print(f"Success: {output_data.get('ok', False)}")
            if 'res' in output_data:
                res = output_data['res']
                vf_field = res.get('vf_field') or res.get('F')
                print(f"Has vf_field: {vf_field is not None}")
                if vf_field is not None:
                    print(f"Field shape: {vf_field.shape if hasattr(vf_field, 'shape') else 'unknown'}")
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Test timed out - this shouldn't happen")
        return False
    except Exception as e:
        print(f"Error running timeout wrapper: {e}")
        return False

if __name__ == "__main__":
    success = test_timeout_wrapper()
    sys.exit(0 if success else 1)
