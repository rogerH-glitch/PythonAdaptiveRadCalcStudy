# PLOTTING PRINCIPLE: never massage visuals. Render computed geometry/fields as-is.

from __future__ import annotations

import numpy as np
from .display_geom import build_display_geom, _order_quad


def _title_from_result(result: dict) -> str:
    ang = float(result.get("angle", 0.0))
    ax  = str(result.get("rotate_axis", "z"))
    tgt = str(result.get("rotate_target", "emitter"))
    piv = str(result.get("angle_pivot", "toe"))
    rc  = result.get("receiver_center", (0.0, 0.0, 0.0))
    off = f" | Offset (dy,dz)=({rc[1]:.3f},{rc[2]:.3f}) m"
    rot = "" if abs(ang) < 1e-12 else f" | {'Yaw' if ax=='z' else 'Pitch'} {ang:.0f}° (pivot={piv}, target={tgt})"
    return "Emitter / Receiver – 3D" + off + rot


def plot_geometry_3d(result: dict, out_html: str, *, return_fig: bool=False, debug_plots: bool=False):
    import plotly.graph_objects as go
    
    # Create a mock args object for build_display_geom
    class MockArgs:
        def __init__(self, result):
            self.emitter = (result.get("We", 5.0), result.get("He", 2.0))
            self.receiver = (result.get("Wr", 5.0), result.get("Hr", 2.0))
            self.setback = result.get("setback", 3.0)
            self.rotate_axis = result.get("rotate_axis", "z")
            self.angle = result.get("angle", 0.0)
            self.angle_pivot = result.get("angle_pivot", "toe")
            self.rotate_target = result.get("rotate_target", "emitter")
    
    args = MockArgs(result)
    display_geom = build_display_geom(args, result)
    
    # Create mesh traces using display geometry
    traces = []


    # Get and order corners for proper rectangle rendering
    emitter_corners = display_geom.get("corners3d", {}).get("emitter")
    receiver_corners = display_geom.get("corners3d", {}).get("receiver")
    if debug_plots:
        try:
            import numpy as _np
            _em = _np.asarray(display_geom.get("corners3d", {}).get("emitter"), float)
            _rc = _np.asarray(display_geom.get("corners3d", {}).get("receiver"), float)
            if _em.size:
                print("[p0-3d] emitter_corners=", _em.round(6).tolist())
            if _rc.size:
                print("[p0-3d] receiver_corners=", _rc.round(6).tolist())
        except Exception:
            pass
    
    def mesh_from_quad(q, name, color):
        """Convert ordered quad corners to Mesh3d with two triangles."""
        x, y, z = q[:, 0], q[:, 1], q[:, 2]
        # faces: (0,1,2) and (0,2,3) - two triangles forming the quad
        
        # Debug instrumentation for 3D quad planarity
        if debug_plots:
            # Check planarity using SVD
            centroid = np.mean(q, axis=0)
            centered = q - centroid
            U, S, Vt = np.linalg.svd(centered)
            normal = Vt[-1]  # Last row of Vt (smallest singular value)
            distances = np.abs(np.dot(centered, normal))
            max_distance = np.max(distances)
            is_planar = max_distance < 1e-6
            
            print(f"[debug-3d] {name} quad planarity check:")
            print(f"[debug-3d]   corners: {q.round(6).tolist()}")
            print(f"[debug-3d]   plane normal: {normal.round(6)}")
            print(f"[debug-3d]   max distance from plane: {max_distance:.2e}")
            print(f"[debug-3d]   is planar: {is_planar}")
            print(f"[debug-3d]   triangle indices: (0,1,2) and (0,2,3)")
            
            if not is_planar:
                print(f"[debug-3d]   WARNING: {name} quad is not planar (expected for rectangles)")
        
        return go.Mesh3d(x=x, y=y, z=z, 
                        i=[0, 0], j=[1, 2], k=[2, 3],
                        color=color, opacity=0.9, name=name, 
                        flatshading=True, showscale=False)
    
    if emitter_corners is not None:
        # Convert list to numpy array and take first 4 points (remove duplicates)
        em_array = np.array(emitter_corners[:4])  # shape (4,3), values unmodified
        em = _order_quad(em_array)
        try:
            print("[f3-3d] emitter_ordered=", np.asarray(em).round(6).tolist())
        except Exception:
            pass
        traces.append(mesh_from_quad(em, "Emitter", "red"))
        # --- G4: thin wireframe outline (closed loop) ---
        def _edge_loop(q):
            q = np.asarray(q, float)
            loop = np.vstack([q, q[0:1]])
            return loop[:, 0], loop[:, 1], loop[:, 2]
        ex, ey, ez = _edge_loop(em)
        traces.append(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                                   line=dict(width=2), name="Emitter edge", showlegend=False))

    if receiver_corners is not None:
        # Convert list to numpy array and take first 4 points (remove duplicates)
        rc_array = np.array(receiver_corners[:4])  # shape (4,3)
        rc = _order_quad(rc_array)
        try:
            print("[f3-3d] receiver_ordered=", np.asarray(rc).round(6).tolist())
        except Exception:
            pass
        traces.append(mesh_from_quad(rc, "Receiver", "black"))
        rx, ry, rz = _edge_loop(rc)
        traces.append(go.Scatter3d(x=rx, y=ry, z=rz, mode="lines",
                                   line=dict(width=2), name="Receiver edge", showlegend=False))
    
    # Create figure with title
    fig = go.Figure(data=traces)
    fig.update_layout(title=_title_from_result(result),
                      scene_aspectmode="data",
                      legend=dict(orientation="h", y=1.02, x=0.0))
    
    fig.write_html(out_html, include_plotlyjs="cdn")
    if return_fig:
        return fig
