"""Mode forecasts recover an exact law when the context identifies the branch."""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from exponent_ceiling import median_exponents
from mop_ceiling import build_m1, free_modes, labelled_modes, nearest_train

T, CTX, PER_BRANCH = 32, 8, 60


def power_law_trajectories():
    """Reaction and static windows on exact power laws, two window starts each,
    enough per branch that a window's 25 nearest train contexts share its branch."""
    rows = [
        (p, s)
        for p in ("reaction", "static")
        for s in (0, 36)
        for _ in range(PER_BRANCH)
    ]
    process = np.array([p for p, _ in rows])
    start = np.array([s for _, s in rows])
    times = 10.0 + 0.5 * (start[:, None] + 2 * np.arange(T))
    alpha = np.where(process == "reaction", 0.5, 0.0)[:, None]
    r0 = np.linspace(3.0, 5.0, len(rows))[:, None]
    train = np.tile([True, False], len(rows) // 2)
    truth = r0 * times**alpha
    return {
        "truth": truth,
        "times": times,
        "process": process,
        "start": start,
        "train": train,
    }


def test_noise_free_modes_recover_the_law_exactly():
    traj = power_law_trajectories()
    nearest = nearest_train(
        traj["truth"][:, :CTX] / traj["truth"][:, CTX - 1 : CTX], traj["train"]
    )
    labelled, accuracy = labelled_modes(traj["truth"], traj, nearest, CTX)
    with warnings.catch_warnings():  # 4 distinct futures for 6 clusters
        warnings.simplefilter("ignore")
        last = traj["truth"][:, CTX - 1 : CTX]
        free, _ = free_modes(last, np.log(traj["truth"][:, CTX:] / last), traj, nearest)
    assert accuracy["process"] == 1.0  # static starts are identical, so branch is not
    for forecast in (labelled, free["free_top"]):
        exps = median_exponents(forecast, traj, CTX)
        assert np.isclose(exps["reaction"], 0.5) and np.isclose(
            exps["static"], 0.0, atol=1e-9
        )


def test_build_m1_needs_both_growth_laws_and_a_flat_static():
    def result(d, r, s):
        return {
            "sd_0.8": {
                "exponent": {"free_top": {"diffusion": d, "reaction": r, "static": s}}
            }
        }

    assert build_m1(result(0.30, 0.45, 0.05))
    assert not build_m1(result(0.30, 0.38, 0.05))
    assert not build_m1(result(0.30, 0.45, 0.15))
