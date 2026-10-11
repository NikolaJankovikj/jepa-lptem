"""The exponent-ceiling references and reading on cases with a known answer."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from exponent_ceiling import median_exponents, oracle_forecasts, reachable

T, CTX = 32, 8


def power_law_trajectories():
    """Reaction and static windows on exact power laws, two window starts each."""
    rows = [(p, s) for p in ("reaction", "static") for s in (0, 36) for _ in range(4)]
    process = np.array([p for p, _ in rows])
    start = np.array([s for _, s in rows])
    times = 10.0 + 0.5 * (start[:, None] + 2 * np.arange(T))
    alpha = np.where(process == "reaction", 0.5, 0.0)[:, None]
    train = np.tile([True, True, False, False], len(rows) // 4)
    truth = 4.0 * times**alpha
    return {"truth": truth, "times": times, "process": process, "start": start, "train": train}


def test_process_and_start_oracle_recovers_the_law_exactly():
    traj = power_law_trajectories()
    forecast = oracle_forecasts(traj["truth"], traj, CTX)["process_start_given"]
    exps = median_exponents(forecast, traj, CTX)
    assert np.isclose(exps["reaction"], 0.5) and np.isclose(exps["static"], 0.0)


def test_reachable_needs_both_growth_processes_within_the_tolerance():
    def block(d, r):
        e = {"diffusion": d, "reaction": r}
        return {"context_8": {"sd_0.8": {"ridge": e, "knn": e}}}

    assert reachable(block(0.30, 0.45))["reachable"]
    assert not reachable(block(0.30, 0.35))["reachable"]
