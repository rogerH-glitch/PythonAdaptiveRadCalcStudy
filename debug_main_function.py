import numpy as np
from src.viz.display_geom import build_display_geom

def support_gap(a, b, n):
    """Compute the minimum signed distance between polygons along normal n."""
    n = n/np.linalg.norm(n)
    return np.min(b @ n) - np.min(a @ n)

# Test case: yaw=30°, toe pivot, target=receiver
print("=== Debug: Main function execution ===")
g = build_display_geom(width=5.1, height=2.1, setback=3.0,
                      angle=30.0, pitch=0.0, angle_pivot="toe", target="receiver",
                      x_offset=0.0, y_offset=0.25)

em = g["emitter"]["corners"]
rc = g["receiver"]["corners"]

print(f"Emitter corners:\n{em}")
print(f"Receiver corners:\n{rc}")

yaw = np.deg2rad(30.0)
n_rc = np.array([-np.cos(yaw), -np.sin(yaw), 0.0])

gap_rc = support_gap(rc, em, n_rc)
print(f"Gap along receiver normal: {gap_rc}")

# Check if the geometry makes sense
print(f"\n=== Geometry analysis ===")
print(f"Emitter x range: {np.min(em[:, 0]):.3f} to {np.max(em[:, 0]):.3f}")
print(f"Receiver x range: {np.min(rc[:, 0]):.3f} to {np.max(rc[:, 0]):.3f}")
print(f"Emitter y range: {np.min(em[:, 1]):.3f} to {np.max(em[:, 1]):.3f}")
print(f"Receiver y range: {np.min(rc[:, 1]):.3f} to {np.max(rc[:, 1]):.3f}")

# Check if the receiver is actually rotated
print(f"\n=== Rotation analysis ===")
# The receiver should be rotated, so its corners should not be axis-aligned
rc_x_variation = np.max(rc[:, 0]) - np.min(rc[:, 0])
rc_y_variation = np.max(rc[:, 1]) - np.min(rc[:, 1])
print(f"Receiver x variation: {rc_x_variation:.6f}")
print(f"Receiver y variation: {rc_y_variation:.6f}")

if rc_x_variation < 1e-6:
    print("Receiver is NOT rotated (x coordinates are constant)")
else:
    print("Receiver IS rotated (x coordinates vary)")

if rc_y_variation < 1e-6:
    print("Receiver is NOT rotated (y coordinates are constant)")
else:
    print("Receiver IS rotated (y coordinates vary)")
