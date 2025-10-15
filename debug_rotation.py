import numpy as np
from src.viz.display_geom import build_display_geom, _apply_about_pivot, _toe_pivot_point

def support_gap(a, b, n):
    """Compute the minimum signed distance between polygons along normal n."""
    n = n/np.linalg.norm(n)
    return np.min(b @ n) - np.min(a @ n)

# Test case: yaw=30°, toe pivot, target=receiver
print("=== Debug: Rotation analysis ===")

# Get the configuration
W = 5.1
H = 2.1
yaw = np.deg2rad(30.0)
pitch = 0.0
setback = 3.0
base = np.array([0.0, 0.25, 0.0])
pivot_mode = "toe"
target = "receiver"

# Compute pivots
emitter_pivot_world = _toe_pivot_point("emitter", W, H, setback, base, yaw, pitch)
receiver_pivot_world = _toe_pivot_point("receiver", W, H, setback, base, yaw, pitch)

print(f"Emitter pivot: {emitter_pivot_world}")
print(f"Receiver pivot: {receiver_pivot_world}")

# Create local panels
emitter_local = np.array([
    [0.0,  0.0,  0.0,  0.0],
    [-W/2, W/2,  W/2, -W/2],
    [-H/2, -H/2, H/2,  H/2],
], dtype=float)

receiver_local = emitter_local.copy()

# Start from world positions before rotation
E0 = emitter_local + base.reshape(3, 1)
R0 = receiver_local + (base + np.array([setback, 0.0, 0.0])).reshape(3, 1)

print(f"E0 (emitter before rotation):\n{E0}")
print(f"R0 (receiver before rotation):\n{R0}")

# Rotation matrix
R = np.array([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw), np.cos(yaw), 0],
    [0, 0, 1]
])

print(f"Rotation matrix:\n{R}")

# Apply rotation
R1 = _apply_about_pivot(R0, R, receiver_pivot_world)

print(f"R1 (receiver after rotation):\n{R1}")

# Check if rotation actually happened
r0_x_variation = np.max(R0[0, :]) - np.min(R0[0, :])
r1_x_variation = np.max(R1[0, :]) - np.min(R1[0, :])

print(f"R0 x variation: {r0_x_variation:.6f}")
print(f"R1 x variation: {r1_x_variation:.6f}")

if r1_x_variation > r0_x_variation + 1e-6:
    print("Rotation DID happen")
else:
    print("Rotation did NOT happen")
