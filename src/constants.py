import warnings

warnings.warn(
    "Module 'src.constants' is deprecated; migrate to local module constants within call sites or central config. This shim will be removed in v1.2.",
    DeprecationWarning,
    stacklevel=2,
)
EPS = 1e-12

STATUS_CONVERGED = "converged"
STATUS_REACHED_LIMITS = "reached_limits"
STATUS_FAILED = "failed"
