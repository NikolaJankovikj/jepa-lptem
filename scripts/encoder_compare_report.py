"""Step 15 M4: frozen V-JEPA 2 vs LoRA vs CNN, scored with clip-bootstrap CIs.

Reads each run's ``preds.npz`` (per-clip observables) against the tubelet ground
truth, on the test clips, physics stratum. Every run is resampled with the same
clip draws, so model differences are paired. Detection and MSD come from each
run's ``metrics.json`` (no CI: the predicted positions were not saved).

The mean-radius reading sd is ``snr_split.reading_noise``'s sd-equivalent
(MAE / 0.80 on frames with a particle), split at beam flux 21.2, as Step 10's RQ3.

    uv run python scripts/encoder_compare_report.py --runs experiments/encoder-compare \
        --tubelets data/tubelets/dynamics-v1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gate2_followup import BOOT_SEED, N_BOOT
from snr_split import JUDGED_VARIABLE, SPLITS, reading_noise
from train_encoder_compare import Windowing
from train_probes import OBS_NAMES, r_squared

MODELS = ("frozen", "cnn", "lora")
FRACS = (100, 30, 10)
NAMES = tuple(f"{m}-{f}" for f in FRACS for m in MODELS)
MSD_CEILING_TOL = 0.10
THESIS_SD = 0.6
RADIUS = OBS_NAMES.index("mean_radius")
COUNT = OBS_NAMES.index("count")
R2S = tuple(f"r2_{n}" for n in OBS_NAMES)


@dataclass
class TestSet:
    """Ground truth of the physics test clips, at the latent frames Gate 1 scores."""

    rows: np.ndarray
    truth: np.ndarray
    frames: np.ndarray
    dose_low: np.ndarray

    def pooled(self, preds: dict, rows) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.concatenate([preds[r][self.frames] for r in rows]),
            np.concatenate([self.truth[r][self.frames] for r in rows]),
        )


def frame_stats(pred: np.ndarray, truth: np.ndarray) -> dict:
    """R² per observable and the mean-radius reading sd of pooled ``(n, 3)`` frames."""
    out = {s: r_squared(pred[:, j], truth[:, j]) for j, s in enumerate(R2S)}
    noise = reading_noise(pred[:, RADIUS], truth[:, RADIUS], truth[:, COUNT])
    out["reading_sd"] = noise["sd_equivalent"]
    return out


def run_stats(test: TestSet, preds: dict, rows) -> dict:
    """Statistics of one run on one resample ``rows`` of the test clips."""
    out = frame_stats(*test.pooled(preds, rows))
    low = test.dose_low[rows]
    for name, sub in (("low", rows[low]), ("high", rows[~low])):
        out[f"reading_sd_{name}"] = frame_stats(*test.pooled(preds, sub))["reading_sd"]
    out["low_dose_penalty"] = out["reading_sd_low"] - out["reading_sd_high"]
    return out


def bootstrap(test: TestSet, runs: dict) -> dict:
    """Per run: (point statistics, paired clip-bootstrap draws of each)."""
    rng = np.random.default_rng(BOOT_SEED)
    picks = [
        test.rows[rng.integers(0, len(test.rows), len(test.rows))]
        for _ in range(N_BOOT)
    ]
    out = {}
    for name, preds in runs.items():
        draws = [run_stats(test, preds, p) for p in picks]
        out[name] = (
            run_stats(test, preds, test.rows),
            {s: np.array([d[s] for d in draws]) for s in draws[0]},
        )
    return out


def ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def diff(boot: dict, a: str, b: str, stat: str) -> dict:
    """``a - b`` on one statistic, paired over the same clip draws."""
    d = boot[a][1][stat] - boot[b][1][stat]
    return {
        "stat": f"{a} - {b} {stat}",
        "value": boot[a][0][stat] - boot[b][0][stat],
        "ci": ci(d),
    }


def comparison(name: str, ds: list[dict], *, strict: bool) -> dict:
    """A prediction that some differences are positive (``strict``) or not negative;
    ``resolved`` when every difference's CI excludes 0."""
    return {
        "prediction": name,
        "right": all(d["value"] > 0 if strict else d["value"] >= 0 for d in ds),
        "resolved": all(d["ci"][0] > 0 or d["ci"][1] < 0 for d in ds),
        "differences": ds,
    }


# Predictions 1-3 of PLAN Step 15, fixed before any M3 number:
# (name, [(a, b, statistic) whose a - b is predicted >= 0, or > 0 when strict], strict)
R2_PREDICTIONS = (
    ("1 C >= F on count at 100 %", [("cnn-100", "frozen-100", "r2_count")], False),
    (
        "2 F > C at 10 % on count and mean radius",
        [("frozen-10", "cnn-10", s) for s in R2S[:2]],
        True,
    ),
    (
        "3 L >= F and L >= C on every R2 at 100 %",
        [("lora-100", m, s) for m in ("frozen-100", "cnn-100") for s in R2S],
        False,
    ),
)


def r2_predictions(boot: dict) -> list[dict]:
    return [
        comparison(name, [diff(boot, *t) for t in terms], strict=strict)
        for name, terms, strict in R2_PREDICTIONS
    ]


def reading_prediction(boot: dict) -> dict:
    """Prediction 4: L reads radius better than F, but by less than half, and its
    low-dose penalty shrinks but stays."""
    lora, frozen = boot["lora-100"][0], boot["frozen-100"][0]
    return {
        "prediction": "4 L's reading sd below F's by less than half; penalty shrinks, stays",
        "right": frozen["reading_sd"] / 2 < lora["reading_sd"] < frozen["reading_sd"]
        and 0 < lora["low_dose_penalty"] < frozen["low_dose_penalty"],
        "differences": [
            diff(boot, "lora-100", "frozen-100", "reading_sd"),
            diff(boot, "lora-100", "frozen-100", "low_dose_penalty"),
        ],
        "lora_penalty_ci": ci(boot["lora-100"][1]["low_dose_penalty"]),
    }


def msd_prediction(det: dict) -> dict:
    ratios = {k: v["msd_ratio"] / v["msd_ceiling"] for k, v in det.items()}
    return {
        "prediction": "5 MSD/4Dt within 10 % of the ceiling for all",
        "right": all(abs(r - 1) <= MSD_CEILING_TOL for r in ratios.values()),
        "ratio_to_ceiling": ratios,
    }


def detection(run_dir: Path) -> dict:
    t = json.loads((run_dir / "metrics.json").read_text())["test"]
    p, r = t["det_precision"], t["det_recall"]
    return {
        "det_precision": p,
        "det_recall": r,
        "det_f1": 2 * p * r / (p + r),
        "msd_ratio": t["msd_ratio_pooled"],
        "msd_ceiling": t["msd_ratio_pooled_gt"],
    }


def load(runs_dir: Path, tubelets: Path, win: Windowing) -> tuple[TestSet, dict]:
    """The physics test set and every run's per-clip predictions, by tubelet row."""
    meta = [json.loads(line) for line in (tubelets / "meta.jsonl").open()]
    row_of = {m["clip"]: i for i, m in enumerate(meta)}
    tests = set()
    runs = {}
    for name in NAMES:
        split = json.loads((runs_dir / name / "metrics.json").read_text())["split"]
        tests.add(tuple(sorted(split["test"])))
        z = np.load(runs_dir / name / "preds.npz")
        runs[name] = {
            row_of[str(c)]: a for c, a in zip(z["clips"], z["attentive"], strict=True)
        }
    if len(tests) != 1:
        msg = "runs differ in their test clips"
        raise SystemExit(msg)
    rows = [row_of[c] for c in tests.pop() if meta[row_of[c]]["stratum"] == "physics"]
    truth = np.load(tubelets / "gt.npz")["attentive"]
    dose = np.array([m[JUDGED_VARIABLE] < SPLITS[JUDGED_VARIABLE] for m in meta])
    test = TestSet(np.array(rows), truth, win.frame_index(truth.shape[1]), dose)
    return test, runs


def run_table(boot: dict, det: dict) -> dict:
    """Every statistic of every run, with its CI, plus detection and MSD."""
    return {
        n: {s: {"value": v, "ci": ci(boot[n][1][s])} for s, v in boot[n][0].items()}
        | det[n]
        for n in NAMES
    }


def report(test: TestSet, runs: dict, runs_dir: Path) -> dict:
    boot = bootstrap(test, runs)
    det = {n: detection(runs_dir / n) for n in NAMES}
    lora_sd = boot["lora-100"][0]["reading_sd"]
    return {
        "test_clips_physics": len(test.rows),
        "n_boot": N_BOOT,
        "runs": run_table(boot, det),
        "predictions": [
            *r2_predictions(boot),
            reading_prediction(boot),
            msd_prediction(det),
        ],
        "thesis_link": {
            "lora_100_reading_sd": lora_sd,
            "rerun_longer_context": lora_sd <= THESIS_SD,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--tubelets", required=True, type=Path)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=36)
    args = ap.parse_args()
    test, runs = load(args.runs, args.tubelets, Windowing(args.window, args.stride))
    out = report(test, runs, args.runs)
    (args.runs / "metrics.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
