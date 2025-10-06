from types import SimpleNamespace


def _ns(**kw): return SimpleNamespace(**kw)


def test_resolve_offsets_emitter_only():
    # Simulate args with only emitter-offset supplied (the user's case)
    from src.cli import _resolve_offsets
    args = _ns(receiver_offset=None, emitter_offset=(0.6, 0.4), align_centres=False)
    dy, dz = _resolve_offsets(args)
    # receiver - emitter should be the negative of emitter offset
    assert abs(dy - (-0.6)) < 1e-12
    assert abs(dz - (-0.4)) < 1e-12


def test_resolve_offsets_receiver_only():
    from src.cli import _resolve_offsets
    args = _ns(receiver_offset=(0.2, -0.1), emitter_offset=None, align_centres=False)
    dy, dz = _resolve_offsets(args)
    assert dy == 0.2 and dz == -0.1


def test_resolve_offsets_align_centres():
    from src.cli import _resolve_offsets
    args = _ns(receiver_offset=(1, 1), emitter_offset=(2, 2), align_centres=True)
    dy, dz = _resolve_offsets(args)
    assert dy == 0.0 and dz == 0.0
