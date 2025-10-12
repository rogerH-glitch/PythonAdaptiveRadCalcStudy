from __future__ import annotations

import numpy as np


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
    
    # Extract geometry parameters
    We = float(result.get("We", 5.0))
    He = float(result.get("He", 2.0))
    Wr = float(result.get("Wr", 5.0))
    Hr = float(result.get("Hr", 2.0))
    setback = float(result.get("setback", 3.0))
    
    # Get centers and offsets
    em_center = result.get("emitter_center", (0.0, 0.0, 0.0))
    rc_center = result.get("receiver_center", (setback, 0.0, 0.0))
    
    # Create wireframe traces
    traces = []
    
    # Emitter trace (red)
    xE, yE, zE = _rect_wire_points(em_center, We, He)
    traces.append(go.Scatter3d(x=xE, y=yE, z=zE, mode="lines",
                               line=dict(width=6, color="red"), name="Emitter"))
    
    # Receiver trace (black)
    xR, yR, zR = _rect_wire_points(rc_center, Wr, Hr)
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


def _rect_wire_points(center_xyz, w, h, R=np.eye(3)):
    """Return X,Y,Z arrays for a rectangle wireframe lying in local Y–Z at x=0, rotated by R, translated to center."""
    c = np.array([[0, -w/2, -h/2],
                  [0,  w/2, -h/2],
                  [0,  w/2,  h/2],
                  [0, -w/2,  h/2],
                  [0, -w/2, -h/2]], dtype=float)
    pts = (np.asarray(R, float) @ c.T).T + np.asarray(center_xyz, float)
    return pts[:, 0], pts[:, 1], pts[:, 2]


def geometry_3d_html(*, emitter_center, receiver_center, We, He, Wr, Hr,
                     R_emitter=None, R_receiver=None, out_html="geometry_3d.html",
                     include_plotlyjs="cdn") -> str:
    """
    Save an interactive 3-D HTML showing emitter (red) and receiver (black) wireframes.
    Returns the output path. If plotly is missing, raises ImportError.
    - include_plotlyjs: 'cdn' (smaller file) or 'inline' (offline, bigger file).
    """
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise ImportError("Plotly is required for 3-D output. Try: pip install plotly") from e

    if R_emitter is None:
        R_emitter = np.eye(3)
    if R_receiver is None:
        R_receiver = np.eye(3)

    xE, yE, zE = _rect_wire_points(emitter_center, We, He, R_emitter)
    xR, yR, zR = _rect_wire_points(receiver_center, Wr, Hr, R_receiver)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xE, y=yE, z=zE, mode="lines",
                               line=dict(width=6, color="red"), name="Emitter"))
    fig.add_trace(go.Scatter3d(x=xR, y=yR, z=zR, mode="lines",
                               line=dict(width=6, color="black"), name="Receiver"))
    fig.update_layout(scene_aspectmode="data",
                      legend=dict(orientation="h", y=1.02, x=0.0))
    fig.write_html(out_html, include_plotlyjs=include_plotlyjs)
    return out_html


