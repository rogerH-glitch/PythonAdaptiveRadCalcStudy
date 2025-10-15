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


def plot_geometry_3d(result: dict, out_html: str, *, return_fig: bool=False):
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
    
    # Create wireframe traces using display geometry
    traces = []

    def _maybe_to_xyz_lists(corners):
        """
        Accept np.ndarray shape (N,3) or list of (x,y,z) and return 3 lists (x,y,z).
        Returns (None, None, None) if corners is falsy or empty.
        """
        if corners is None:
            return None, None, None
        arr = np.asarray(corners)
        if arr.size == 0 or arr.ndim != 2 or arr.shape[1] != 3:
            return None, None, None
        x, y, z = arr[:, 0].tolist(), arr[:, 1].tolist(), arr[:, 2].tolist()
        x.append(x[0]); y.append(y[0]); z.append(z[0])
        return x, y, z

    # Get and order corners for proper rectangle rendering
    emitter_corners = display_geom.get("corners3d", {}).get("emitter")
    receiver_corners = display_geom.get("corners3d", {}).get("receiver")
    
    if emitter_corners is not None:
        # Convert list to numpy array and take first 4 points (remove duplicates)
        em_array = np.array(emitter_corners[:4])  # shape (4,3), values unmodified
        em = _order_quad(em_array)
        xE, yE, zE = _maybe_to_xyz_lists(em)
        if xE is not None:
            traces.append(go.Scatter3d(x=xE, y=yE, z=zE, mode="lines",
                                       line=dict(width=6, color="red"), name="Emitter"))

    if receiver_corners is not None:
        # Convert list to numpy array and take first 4 points (remove duplicates)
        rc_array = np.array(receiver_corners[:4])  # shape (4,3)
        rc = _order_quad(rc_array)
        xR, yR, zR = _maybe_to_xyz_lists(rc)
        if xR is not None:
            traces.append(go.Scatter3d(x=xR, y=yR, z=zR, mode="lines",
                                       line=dict(width=6, color="black"), name="Receiver"))
    
    # Create figure with title
    fig = go.Figure(data=traces)
    fig.update_layout(title=_title_from_result(result),
                      scene_aspectmode="data",
                      legend=dict(orientation="h", y=1.02, x=0.0))
    
    fig.write_html(out_html, include_plotlyjs="cdn")
    if return_fig:
        return fig
