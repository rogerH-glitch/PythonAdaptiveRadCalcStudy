from src.viz.display_geom import build_display_geom


def _mk_args(**kw):
    # Minimal args/result pair mimicking caller contract
    class NS:
        pass
    args = NS()
    result = NS()
    for k, v in kw.items():
        setattr(args, k, v)
    # Common defaults
    for k, v in dict(
        emitter_w=1.0, emitter_h=1.0,
        receiver_w=1.0, receiver_h=1.0,
        setback=2.0, angle=0.0,
        rotate_axis="z", rotate_target="emitter",
        receiver_offset=(0.0, 0.0),
    ).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args, result


def test_legacy_xy_xz_and_corners3d_shapes_exist():
    args, result = _mk_args()
    geom = build_display_geom(args, result)

    # Legacy nested keys present
    assert "xy" in geom and "xz" in geom and "corners3d" in geom
    assert "emitter" in geom["xy"] and "receiver" in geom["xy"]
    assert "emitter" in geom["xz"] and "receiver" in geom["xz"]
    assert "emitter" in geom["corners3d"] and "receiver" in geom["corners3d"]

    # XY shape: exactly two endpoints [(x0,y0),(x1,y1)]
    e_xy = geom["xy"]["emitter"]
    r_xy = geom["xy"]["receiver"]
    assert isinstance(e_xy, list) and len(e_xy) == 2 and len(e_xy[0]) == 2
    assert isinstance(r_xy, list) and len(r_xy) == 2 and len(r_xy[0]) == 2

    # XZ dict has aliases: x, zmin/zmax, z0/z1
    e_xz = geom["xz"]["emitter"]
    for key in ("x", "zmin", "zmax", "z0", "z1"):
        assert key in e_xz, f"missing key {key} in emitter XZ"
    # Alias equality
    assert e_xz["z0"] == e_xz["zmin"] and e_xz["z1"] == e_xz["zmax"]

    # 3D corners: 8 points (legacy expectation)
    e_c = geom["corners3d"]["emitter"]
    r_c = geom["corners3d"]["receiver"]
    assert isinstance(e_c, list) and len(e_c) == 8 and len(e_c[0]) == 3
    assert isinstance(r_c, list) and len(r_c) == 8 and len(r_c[0]) == 3
