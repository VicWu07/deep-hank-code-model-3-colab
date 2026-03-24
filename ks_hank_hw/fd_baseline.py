from __future__ import annotations

from .fd_benchmark import solve_fd_steady_anchor, solve_fd_state, solve_fd_z_slices


def solve_fd_baseline(*args, **kwargs):
    """
    Backward-compatible alias.
    For the MRS-adapted Deep HANK codebase, baseline FD means steady anchor.
    """
    return solve_fd_steady_anchor(*args, **kwargs)


__all__ = [
    "solve_fd_baseline",
    "solve_fd_state",
    "solve_fd_steady_anchor",
    "solve_fd_z_slices",
]
