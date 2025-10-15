from src.viz.display_geom import build_display_geom
import numpy as np

def debug_xz_geometry():
    """Debug the X-Z geometry to understand the spans."""
    g = build_display_geom(width=5.1, height=2.1, setback=3.0,
                          angle=30.0, pitch=0.0, angle_pivot="toe", rotate_target="emitter",
                          x_offset=0.0, y_offset=0.25)
    em = g["emitter"]["corners"]
    rc = g["receiver"]["corners"]
    
    print("Emitter corners:")
    print(em)
    print(f"Emitter X range: {np.min(em[:, 0]):.3f} to {np.max(em[:, 0]):.3f}")
    print(f"Emitter Z range: {np.min(em[:, 2]):.3f} to {np.max(em[:, 2]):.3f}")
    print(f"Emitter X span: {np.ptp(em[:, 0]):.3f}")
    print(f"Emitter Z span: {np.ptp(em[:, 2]):.3f}")
    
    print("\nReceiver corners:")
    print(rc)
    print(f"Receiver X range: {np.min(rc[:, 0]):.3f} to {np.max(rc[:, 0]):.3f}")
    print(f"Receiver Z range: {np.min(rc[:, 2]):.3f} to {np.max(rc[:, 2]):.3f}")
    print(f"Receiver X span: {np.ptp(rc[:, 0]):.3f}")
    print(f"Receiver Z span: {np.ptp(rc[:, 2]):.3f}")

if __name__ == "__main__":
    debug_xz_geometry()
