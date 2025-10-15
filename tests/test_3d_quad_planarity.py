"""
Test 3D quad planarity and triangle rendering at high yaw angles.
This test is marked as xfail because the current implementation has bow-tie issues at high yaw.
"""
import pytest
import numpy as np
from src.viz.display_geom import build_display_geom, _order_quad
from src.viz.plot3d import plot_geometry_3d
import plotly.graph_objects as go


# @pytest.mark.xfail(reason="3D panels appear as bow-ties at high yaw angles - corners not properly ordered or triangles incorrectly formed")  # FIXED
def test_3d_quad_planarity_high_yaw():
    """Test that 3D quads are coplanar and render as proper rectangles at high yaw angles."""
    # Create geometry with high yaw angle (89°)
    g = build_display_geom(width=5.1, height=2.1, setback=1.0, angle=89.0, pitch=0.0,
                          angle_pivot="toe", rotate_target="emitter", dy=0.25)
    
    result = {
        "We": 5.1, "He": 2.1, "Wr": 5.1, "Hr": 2.1,
        "setback": 1.0, "angle": 89.0, "rotate_axis": "z",
        "rotate_target": "emitter", "angle_pivot": "toe"
    }
    
    # Get the 3D plot
    fig = plot_geometry_3d(result=result, out_html="test_3d_planarity.html", return_fig=True)
    
    # Find Mesh3d traces
    mesh_traces = [t for t in fig.data if isinstance(t, go.Mesh3d)]
    assert len(mesh_traces) >= 2, f"Expected at least 2 Mesh3d traces, got {len(mesh_traces)}"
    
    for trace in mesh_traces:
        # Get vertices and faces
        x, y, z = trace.x, trace.y, trace.z
        i, j, k = trace.i, trace.j, trace.k
        
        # Check that we have exactly 4 vertices (quad)
        assert len(x) == 4, f"Expected 4 vertices, got {len(x)}"
        
        # Check that we have exactly 2 triangles (faces)
        assert len(i) == 2, f"Expected 2 triangles, got {len(i)}"
        
        # Verify triangle indices are valid
        for idx in i + j + k:
            assert 0 <= idx < 4, f"Invalid triangle index {idx}"
        
        # Check that triangles are (0,1,2) and (0,2,3) or equivalent
        faces = list(zip(i, j, k))
        expected_faces = [(0, 1, 2), (0, 2, 3)]
        
        # Check if faces match expected pattern (allowing for different ordering)
        face_matches = 0
        for face in faces:
            for expected in expected_faces:
                if set(face) == set(expected):
                    face_matches += 1
                    break
        
        assert face_matches == 2, f"Expected faces (0,1,2) and (0,2,3), got {faces}"
        
        # Check planarity - fit a plane to the 4 corners
        corners = np.column_stack([x, y, z])
        assert corners.shape == (4, 3), f"Expected corners shape (4,3), got {corners.shape}"
        
        # Use SVD to find the best-fit plane
        centroid = np.mean(corners, axis=0)
        centered = corners - centroid
        U, S, Vt = np.linalg.svd(centered)
        
        # The normal is the last row of Vt (smallest singular value)
        normal = Vt[-1]
        
        # Check that all points are close to the plane
        distances = np.abs(np.dot(centered, normal))
        max_distance = np.max(distances)
        
        # For a proper rectangle, all points should be on the same plane
        assert max_distance < 1e-6, f"Quad is not planar, max distance from plane: {max_distance}"
        
        # Check that the quad has consistent winding (no bow-tie)
        # Compute cross product of adjacent edges
        edges = np.diff(corners, axis=0)
        # Add the closing edge
        edges = np.vstack([edges, corners[0] - corners[-1]])
        cross_products = []
        for i in range(4):
            edge1 = edges[i]
            edge2 = edges[(i + 1) % 4]
            cross = np.cross(edge1, edge2)
            cross_products.append(cross)
        
        # All cross products should point in the same direction (same sign of dot product with normal)
        dot_products = [np.dot(cp, normal) for cp in cross_products]
        signs = [int(np.sign(dp)) for dp in dot_products]
        
        # All signs should be the same (consistent winding)
        assert len(set(signs)) == 1, f"Inconsistent winding detected, signs: {signs}"


@pytest.mark.xfail(reason="3D quad ordering may not produce proper perimeter order")
def test_3d_quad_ordering():
    """Test that _order_quad returns corners in proper perimeter order."""
    # Create a simple quad and test ordering
    corners = np.array([
        [0, 0, 0],  # corner 0
        [1, 0, 0],  # corner 1  
        [1, 1, 0],  # corner 2
        [0, 1, 0]   # corner 3
    ])
    
    ordered = _order_quad(corners)
    
    # Check that ordered corners form a proper perimeter
    # Compute edge vectors
    edges = np.diff(ordered, axis=0)
    edges = np.vstack([edges, ordered[0] - ordered[-1]])  # Close the loop
    
    # Check that adjacent edges are perpendicular (for a rectangle)
    for i in range(4):
        edge1 = edges[i]
        edge2 = edges[(i + 1) % 4]
        dot_product = np.dot(edge1, edge2)
        assert abs(dot_product) < 1e-6, f"Adjacent edges not perpendicular at corner {i}, dot product: {dot_product}"
    
    # Check that the quad is closed (first and last points are the same)
    assert np.allclose(ordered[0], ordered[-1]), "Quad is not closed"
