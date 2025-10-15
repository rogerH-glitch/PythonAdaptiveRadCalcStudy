# scripts/test_3d_mesh.py
from src.viz.display_geom import build_display_geom
from src.viz.plot3d import plot_geometry_3d
import numpy as np

def test_3d_mesh_rendering():
    """Test that 3D panels render as Mesh3d (two triangles) without bow-ties."""
    # Create a simple geometry
    result = {
        "We": 5.1, "He": 2.1, "Wr": 5.1, "Hr": 2.1,
        "setback": 3.0, "angle": 30.0, "rotate_axis": "z",
        "rotate_target": "emitter", "angle_pivot": "toe"
    }
    
    # Generate 3D plot
    fig = plot_geometry_3d(result, "test_3d_mesh.html", return_fig=True)
    
    # Check that we have Mesh3d traces
    traces = fig.data
    assert len(traces) == 2, f"Expected 2 traces, got {len(traces)}"
    
    # Check that both traces are Mesh3d
    for trace in traces:
        assert trace.type == "mesh3d", f"Expected mesh3d trace, got {trace.type}"
        assert hasattr(trace, 'i') and hasattr(trace, 'j') and hasattr(trace, 'k'), "Mesh3d should have face indices"
        # Check that we have two triangles (faces)
        assert len(trace.i) == 2, f"Expected 2 triangles, got {len(trace.i)}"
        assert len(trace.j) == 2, f"Expected 2 triangles, got {len(trace.j)}"
        assert len(trace.k) == 2, f"Expected 2 triangles, got {len(trace.k)}"
    
    # Check colors
    emitter_trace = traces[0] if "Emitter" in traces[0].name else traces[1]
    receiver_trace = traces[1] if "Receiver" in traces[1].name else traces[0]
    
    assert emitter_trace.color == "red", f"Emitter should be red, got {emitter_trace.color}"
    assert receiver_trace.color == "black", f"Receiver should be black, got {receiver_trace.color}"
    
    print("3D mesh rendering test passed")
    print(f"Emitter: {emitter_trace.name}, color: {emitter_trace.color}")
    print(f"Receiver: {receiver_trace.name}, color: {receiver_trace.color}")

if __name__ == "__main__":
    test_3d_mesh_rendering()
