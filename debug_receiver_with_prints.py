import numpy as np
from src.viz.display_geom import build_display_geom, _place_with_setback

def support_gap(a, b, n):
    """Compute the minimum signed distance between polygons along normal n."""
    n = n/np.linalg.norm(n)
    return np.min(b @ n) - np.min(a @ n)

# Test case: yaw=30°, toe pivot, target=receiver
print("=== Debug: Receiver target case with detailed analysis ===")
g = build_display_geom(width=5.1, height=2.1, setback=3.0,
                      angle=30.0, pitch=0.0, angle_pivot="toe", target="receiver",
                      x_offset=0.0, y_offset=0.25)

em = g["emitter"]["corners"]
rc = g["receiver"]["corners"]

print(f"Emitter corners:\n{em}")
print(f"Receiver corners:\n{rc}")

yaw = np.deg2rad(30.0)
n_rc = np.array([-np.cos(yaw), -np.sin(yaw), 0.0])

print(f"Receiver normal: {n_rc}")

gap_rc = support_gap(rc, em, n_rc)
print(f"Gap along receiver normal: {gap_rc}")

# Test the _place_with_setback function manually
print(f"\n=== Testing _place_with_setback manually ===")
em_new, rc_new = _place_with_setback(em, rc, n_rc, 3.0, "toe")

print(f"After _place_with_setback:")
print(f"Emitter corners:\n{em_new}")
print(f"Receiver corners:\n{rc_new}")

gap_after = support_gap(rc_new, em_new, n_rc)
print(f"Gap along receiver normal after: {gap_after}")

# Check if the function actually changed anything
em_changed = not np.allclose(em, em_new)
rc_changed = not np.allclose(rc, rc_new)
print(f"Emitter changed: {em_changed}")
print(f"Receiver changed: {rc_changed}")

# Check what the function is doing internally
n = n_rc/np.linalg.norm(n_rc)
em_min = np.min(em @ n)
em_max = np.max(em @ n)
rc_min = np.min(rc @ n)
rc_max = np.max(rc @ n)

print(f"\nDebug _place_with_setback internals:")
print(f"n (normalized): {n}")
print(f"em_min: {em_min}")
print(f"em_max: {em_max}")
print(f"rc_min: {rc_min}")
print(f"rc_max: {rc_max}")
print(f"delta = (em_min + setback) - rc_min = ({em_min} + 3.0) - {rc_min} = {em_min + 3.0 - rc_min}")
