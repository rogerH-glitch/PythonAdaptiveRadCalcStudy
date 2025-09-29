# viewfactor_demo_plot.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

# ---- Imports from your modules ----
from viewfactor_1ai_general import viewfactor_1ai_rect_to_rect
from viewfactor_1ai_obstructed_general import viewfactor_1ai_rects_with_occluders
from occluder_params import build_occluders_from_specs

# ---------------- Basic calculator helpers ----------------
SIGMA = 5.670374419e-8  # W/m²-K⁴

def radiant_flux(F12, T_fire, T_surf=300.0, emissivity=1.0):
    """Return q'' [kW/m²] on target from emitter."""
    q = F12 * emissivity * SIGMA * (T_fire**4 - T_surf**4)  # W/m²
    return q / 1000.0  # convert to kW/m²


# ---------------- Basic geometry helpers ----------------
def rectangle_patch_centers(origin, u_vec, v_vec, nu, nv):
    du = u_vec / nu
    dv = v_vec / nv
    dA = np.linalg.norm(np.cross(u_vec, v_vec)) / (nu * nv)
    n = np.cross(u_vec, v_vec)
    n = n / np.linalg.norm(n)
    P = []
    for i in range(nu):
        for j in range(nv):
            p = origin + (i + 0.5) * du + (j + 0.5) * dv
            P.append(p)
    return np.vstack(P), dA, n

def rect_corners(origin, u_vec, v_vec):
    o = origin
    a = origin + u_vec
    b = origin + u_vec + v_vec
    c = origin + v_vec
    return np.array([o, a, b, c, o])

def oriented_normals_for_display(rect1, rect2):
    o1, u1, v1 = rect1
    o2, u2, v2 = rect2
    n1 = np.cross(u1, v1); n1 /= np.linalg.norm(n1)
    n2 = np.cross(u2, v2); n2 /= np.linalg.norm(n2)
    c1 = o1 + 0.5*(u1 + v1)
    c2 = o2 + 0.5*(u2 + v2)
    if np.dot(n1, c2 - c1) < 0: n1 = -n1
    if np.dot(n2, c1 - c2) < 0: n2 = -n2
    return n1, n2

def draw_normal_vec(ax, origin, u_vec, v_vec, n_vec, scale=0.4):
    c = origin + 0.5*(u_vec + v_vec)
    p2 = c + scale * (n_vec / np.linalg.norm(n_vec))
    ax.plot([c[0], p2[0]], [c[1], p2[1]], [c[2], p2[2]], c='k', linewidth=2)
    ax.text(p2[0], p2[1], p2[2], "n", color='k')

# -------- Non-blocking keep-alive utilities ----------
def nonblocking_keepalive(fig):
    plt.show(block=False); plt.pause(0.01)
    # Keep alive until window closed
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)

# -------- General occluder visual (rays clipped at first hit) --------
def _segment_intersection_with_plane(p1, p2, plane_point, plane_normal):
    d = p2 - p1
    n = plane_normal / np.linalg.norm(plane_normal)
    denom = np.dot(d, n)
    if abs(denom) < 1e-12:
        return (False, None, None)
    t = np.dot(plane_point - p1, n) / denom
    if t < 0.0 or t > 1.0:
        return (False, None, None)
    return (True, p1 + t*d, t)

def demo_plot_with_occluder_general(rect1, rect2, occluder_list,
                                    nu=12, nv=12, n_rays=60, seed=42, log_scale=True,
                                    show_F=None, ax=None):
    """
    Draw source/target and sample rays.
    - Visible rays: colored by kernel (thicker = stronger).
    - Blocked rays: dashed gray, clipped at the first occluder face.
    Works for general occluders (rect faces or arbitrary polygons).
    """
    rng = np.random.default_rng(seed)

    P1, _, n1_raw = rectangle_patch_centers(*rect1, nu, nv)
    P2, _, n2_raw = rectangle_patch_centers(*rect2, nu, nv)

    c1 = P1.mean(axis=0); c2 = P2.mean(axis=0)
    n1 = n1_raw.copy(); n2 = n2_raw.copy()
    if np.dot(n2, c1 - c2) < 0: n2 = -n2
    if np.dot(n1, c2 - c1) < 0: n1 = -n1

    idx1 = rng.choice(P1.shape[0], size=min(n_rays, P1.shape[0]), replace=False)
    idx2 = rng.choice(P2.shape[0], size=min(n_rays, P2.shape[0]), replace=False)

    R1 = rect_corners(*rect1)
    R2 = rect_corners(*rect2)

    local_fig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        local_fig = True

    # Frames and normals
    ax.plot(R1[:,0], R1[:,1], R1[:,2], linewidth=2, color='red', label="Source (rect1)")
    ax.plot(R2[:,0], R2[:,1], R2[:,2], linewidth=2, color='blue', label="Target (rect2)")
    n1_disp, n2_disp = oriented_normals_for_display(rect1, rect2)
    draw_normal_vec(ax, *rect1, n1_disp, scale=0.4)
    draw_normal_vec(ax, *rect2, n2_disp, scale=0.4)

    c1d = rect1[0] + 0.5*(rect1[1] + rect1[2])
    c2d = rect2[0] + 0.5*(rect2[1] + rect2[2])
    ax.text(c1d[0], c1d[1], c1d[2], "Source", color='red')
    ax.text(c2d[0], c2d[1], c2d[2], "Target", color='blue')

    # --- General projection helpers (same logic as solver) ---
    from shapely.geometry import Polygon, Point as ShPoint
    from shapely.ops import unary_union
    from occluder_projection import (
        _rect_corners as _rect_corners_full, _unit, target_plane_frame, world_to_plane_2d,
        project_poly_from_point_to_plane
    )
    # target plane 2D frame
    o2, u2, v2 = rect2
    n2_math = _unit(np.cross(u2, v2))
    o_plane, t1p, t2p, n_plane = target_plane_frame(o2, n2_math)

    # pre-build face data for ray/plane hits
    def face_data_from_rect(o,u,v):
        face3 = _rect_corners_full(o,u,v)
        n = _unit(np.cross(u,v))
        return dict(kind="rect", corners3=face3, plane_point=o, plane_normal=n, o=o, u=u, v=v)
    def face_data_from_poly(poly3):
        p0, p1_, p2_ = poly3[0], poly3[1], poly3[2]
        n = _unit(np.cross(p1_-p0, p2_-p0))
        return dict(kind="poly", corners3=poly3, plane_point=p0, plane_normal=n)

    faces = []
    for kind, data in occluder_list:
        if kind == "rect":
            o,u,v = data
            faces.append(face_data_from_rect(o,u,v))
        elif kind == "poly":
            faces.append(face_data_from_poly(data))

    # compute K + visibility + clipping for shown rays
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    cmap = plt.colormaps['viridis']
    Ks, pairs, clipped = [], [], []

    for i in range(len(idx1)):
        p1 = P1[idx1[i]]
        p2 = P2[idx2[i]]

        r_vec = p2 - p1; r = np.linalg.norm(r_vec)
        if r == 0:
            K = 0.0
        else:
            r_hat = r_vec / r
            cos1 = np.dot(n1, r_hat)
            cos2 = -np.dot(n2, r_hat)
            K = (cos1 * cos2) / (np.pi * r*r) if (cos1>0 and cos2>0) else 0.0

        # perspective projection union (like solver)
        occ_proj_2d = []
        for fd in faces:
            proj3 = project_poly_from_point_to_plane(p1, o_plane, n_plane, fd["corners3"])
            proj2 = world_to_plane_2d(proj3, o_plane, t1p, t2p)
            occ_proj_2d.append(Polygon(proj2[:-1]))
        occ_union = unary_union(occ_proj_2d) if occ_proj_2d else None

        uv = world_to_plane_2d(p2[None,:], o_plane, t1p, t2p)[0]
        is_blocked = (occ_union is not None) and occ_union.contains(ShPoint(uv[0], uv[1]))

        # If blocked, clip at nearest occluder face intersection (if any)
        Pend = p2
        if is_blocked:
            d = p2 - p1
            tmin = 1e9; hit_any = False
            for fd in faces:
                hit, Pint, t = _segment_intersection_with_plane(p1, p2, fd["plane_point"], fd["plane_normal"])
                if not hit:
                    continue
                # check Pint inside face polygon
                if fd["kind"] == "rect":
                    # local uv by Pint ≈ o + a*u + b*v
                    M = np.column_stack((fd["u"], fd["v"]))  # 3x2
                    uv2, *_ = np.linalg.lstsq(M, (Pint - fd["o"]), rcond=None)
                    a, b = uv2[0], uv2[1]
                    inside = (0 <= a <= 1) and (0 <= b <= 1)
                else:
                    # general polygon: project to a basis on face plane
                    nface = fd["plane_normal"]
                    # build simple in-plane axes
                    xaxis = np.cross(nface, np.array([1.0,0.0,0.0]))
                    if np.linalg.norm(xaxis) < 1e-12:
                        xaxis = np.cross(nface, np.array([0.0,1.0,0.0]))
                    xaxis = xaxis/np.linalg.norm(xaxis)
                    yaxis = np.cross(nface, xaxis); yaxis = yaxis/np.linalg.norm(yaxis)
                    def to_face_uv(Q):
                        return ((Q - fd["plane_point"]) @ xaxis, (Q - fd["plane_point"]) @ yaxis)
                    poly_uv = Polygon([to_face_uv(Q) for Q in fd["corners3"][:-1]])
                    inside = poly_uv.contains(ShPoint(*to_face_uv(Pint)))
                if inside and 0.0 <= t < tmin:
                    tmin = t; Pend = Pint; hit_any = True
            is_blocked = is_blocked and hit_any

        Ks.append(K); pairs.append((p1, p2)); clipped.append((p1, Pend, is_blocked))

    Ks = np.array(Ks)
    Kmin = np.min(Ks[Ks>0]) if np.any(Ks>0) else 0.0
    Kmax = np.max(Ks) if np.max(Ks)>0 else 1.0
    if log_scale and Kmax > 0 and Kmin > 0:
        Kplot = np.log10(np.clip(Ks, Kmin, None))
        norm = Normalize(vmin=np.log10(Kmin), vmax=np.log10(Kmax))
        cbar_label = "log10 Kernel K"
    else:
        Kplot = Ks
        norm = Normalize(vmin=0.0, vmax=Kmax)
        cbar_label = "Kernel K = (cosθ1 cosθ2)/(π r^2)"

    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])

    # draw rays
    for (p1, p2), (q1, qend, blocked), kval, kval_plot in zip(pairs, clipped, Ks, Kplot):
        if blocked:
            ax.plot([q1[0], qend[0]], [q1[1], qend[1]], [q1[2], qend[2]],
                    linewidth=0.5*1.0, c='gray', linestyle='--')
        else:
            color = cmap(norm(kval_plot))
            # Half line width
            lw = 0.5*(0.5 + 2.5 * (0 if Kmax==0 else (kval / Kmax)))
            ax.plot([q1[0], qend[0]], [q1[1], qend[1]], [q1[2], qend[2]],
                    linewidth=0.5*lw, c=color)

    # --- draw only occluders that actually block at least one ray ---
    from shapely.geometry import Point as ShPoint
    from shapely.ops import unary_union

    drawn = False
    for kind, data in occluder_list:
        occ_hits = 0
        if kind == "rect":
            o, u, v = data
            Rc = rect_corners(o, u, v)
            # Check if any displayed ray endpoint is behind this occluder
            for (p1, p2) in pairs:
                # simple check: ray midpoint
                mid = 0.5 * (p1 + p2)
                if np.dot(np.cross(u, v), mid - o) > 0:  # crude facing test
                    occ_hits += 1
            if occ_hits > 0:
                ax.plot(Rc[:, 0], Rc[:, 1], Rc[:, 2],
                        color='green', linewidth=2, linestyle='--', label="Occluder")
                drawn = True

        elif kind == "poly":
            poly = np.array(data)
            for (p1, p2) in pairs:
                mid = 0.5 * (p1 + p2)
                # crude test: midpoint near plane of polygon
                if np.abs(np.dot(np.cross(poly[1] - poly[0], poly[2] - poly[0]),
                                 mid - poly[0])) < 0.2:
                    occ_hits += 1
            if occ_hits > 0:
                ax.plot(poly[:, 0], poly[:, 1], poly[:, 2],
                        color='green', linewidth=2, linestyle='--', label="Occluder")
                drawn = True

    if drawn:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

    cbar = plt.colorbar(sm, ax=ax, pad=0.1, fraction=0.03)
    cbar.set_label(cbar_label)

    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

    title = 'General occluders: kernel-weighted rays'
    if show_F is not None:
        title += f'\nF1→2 ≈ {show_F:.4f}'
    ax.set_title(title)

    if local_fig:
        plt.show(block=False); plt.pause(0.01)
        return fig

# -------- Side-by-side: Unobstructed vs Obstructed ----------
def show_unob_vs_obstr(rect1, rect2, occluder_list,
                       F_unobs, F_obstr,
                       q_unobs=None, q_obstr=None,
                       nu=12, nv=12, n_rays=50, seed=7, log_scale=True):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5),
                            subplot_kw={'projection': '3d'})

    demo_plot_with_occluder_general(rect1, rect2, [],
                                    nu=nu, nv=nv, n_rays=n_rays, seed=seed,
                                    log_scale=log_scale, show_F=F_unobs, ax=axs[0])
    t1 = f'Unobstructed (F≈{F_unobs:.4f})'
    if q_unobs is not None:
        t1 += f'\nq≈{q_unobs:.1f} kW/m²'
    axs[0].set_title(t1)

    demo_plot_with_occluder_general(rect1, rect2, occluder_list,
                                    nu=nu, nv=nv, n_rays=n_rays, seed=seed,
                                    log_scale=log_scale, show_F=F_obstr, ax=axs[1])
    t2 = f'Obstructed (F≈{F_obstr:.4f})'
    if q_obstr is not None:
        t2 += f'\nq≈{q_obstr:.1f} kW/m²'
    axs[1].set_title(t2)

    plt.tight_layout()
    return fig


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    # --- Base geometry: two 1×1 squares, 1 m apart (Z) ---
    rect1 = (np.array([0.0, 0.0, 0.0]),
             np.array([1.0, 0.0, 0.0]),
             np.array([0.0, 1.0, 0.0]))
    rect2 = (np.array([0.0, 0.0, 1.0]),
             np.array([1.0, 0.0, 0.0]),
             np.array([0.0, 1.0, 0.0]))

    # --- Unobstructed view factor (general 1AI) ---
    F_unobs_general = viewfactor_1ai_rect_to_rect(rect1, rect2, n_src=14, n_tgt=24)
    print(f"Unobstructed (general) F1->2 ≈ {F_unobs_general:.4f}")

    # --- Define occluders via simple parameter specs ---
    emitter_origin = rect1[0] + 0.5*(rect1[1] + rect1[2])
    emitter_normal = np.cross(rect1[1], rect1[2]); emitter_normal /= np.linalg.norm(emitter_normal)

    occluder_specs = [
        # Parallel slab at 0.50 m, 0.60×0.60 m, 0.1 m thick
        dict(shape="rect", orientation="parallel",
             setback=0.50, width=0.60, height=0.60, thickness=0.1),

        # Perpendicular fence at 0.40 m, 0.80×0.60 m
        dict(shape="rect", orientation="perpendicular",
             setback=0.40, width=0.80, height=0.60),

        # Angled sheet +25° at 0.55 m, 0.70×0.50 m
        dict(shape="rect", orientation="angled", angle_deg=25.0,
             setback=0.55, width=0.70, height=0.50),

        # Circular disk near center, radius 0.15 m (@ 0.48 m)
        dict(shape="disk", setback=0.48, radius=0.15, n_sides=48),
    ]
    occluder_list = build_occluders_from_specs(emitter_origin, emitter_normal, occluder_specs)

    # --- Obstructed view factor (general projection) ---
    F_obstr_general = viewfactor_1ai_rects_with_occluders(rect1, rect2, occluder_list,
                                                          n_src=14, n_tgt=24)
    print(f"Obstructed (general projection) F1->2 ≈ {F_obstr_general:.4f}")

    # --- Calculate heat fluxes ---
    q_obs = radiant_flux(F_obstr_general, T_fire=1000.0, T_surf=300.0, emissivity=0.95)
    print(f"Obstructed heat flux q'' ≈ {q_obs:.2f} kW/m²")

    q_unobs = radiant_flux(F_unobs_general, T_fire=1000.0, T_surf=300.0, emissivity=0.95)
    print(f"Unobstructed heat flux q'' ≈ {q_unobs:.2f} kW/m²")

    # --- Side-by-side window: Unobstructed vs Obstructed (with F labels) ---
    fig = show_unob_vs_obstr(rect1, rect2, occluder_list,
                             F_unobs_general, F_obstr_general,
                             q_unobs=q_unobs, q_obstr=q_obs,
                             nu=12, nv=12, n_rays=50, seed=7, log_scale=True)

    nonblocking_keepalive(fig)


