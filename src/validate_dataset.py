from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from roboops.data import (
    ACTION_KEY,
    AGENT_STATE_KEY,
    ENVIRONMENT_STATE_KEY,
    EPISODE_KEY,
    REQUIRED_KEYS,
    load_validation_data,
)

DEFAULT_DATA = Path("data/processed/pusht_keypoints.npz")
EXPECTED_SHAPES = {
    AGENT_STATE_KEY: (2,),
    ENVIRONMENT_STATE_KEY: (16,),
    ACTION_KEY: (2,),
}


@dataclass(frozen=True)
class Check:
    label: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run beginner-friendly dataset quality checks.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    return parser.parse_args()


def _row_count(value: np.ndarray) -> int:
    return int(value.shape[0]) if value.ndim > 0 else 0


def validate(values: dict[str, np.ndarray]) -> list[Check]:
    missing = [key for key in REQUIRED_KEYS if key not in values]
    checks = [
        Check(
            "required state / action fields exist",
            not missing,
            "all required fields found" if not missing else f"missing: {', '.join(missing)}",
        )
    ]
    if missing:
        checks.extend(
            [
                Check("dataset is not empty", False, "cannot inspect rows"),
                Check("state / action lengths match", False, "cannot compare fields"),
                Check("state / action shapes match schema", False, "cannot inspect shapes"),
                Check("no NaN / Inf values", False, "cannot inspect numeric values"),
                Check("state / action range", False, "cannot inspect numeric values"),
                Check("episode length", False, "cannot inspect episodes"),
            ]
        )
        return checks

    required = {key: np.asarray(values[key]) for key in REQUIRED_KEYS}
    counts = {key: _row_count(value) for key, value in required.items()}
    frame_count = counts[ACTION_KEY]
    checks.append(Check("dataset is not empty", frame_count > 0, f"{frame_count} frames"))
    same_length = len(set(counts.values())) == 1
    checks.append(
        Check(
            "state / action lengths match",
            same_length,
            ", ".join(f"{key}={count}" for key, count in counts.items()),
        )
    )

    shapes_ok = all(
        required[key].ndim == 2 and required[key].shape[1:] == expected
        for key, expected in EXPECTED_SHAPES.items()
    ) and required[EPISODE_KEY].ndim == 1
    shape_detail = ", ".join(f"{key}={required[key].shape}" for key in REQUIRED_KEYS)
    checks.append(Check("state / action shapes match schema", shapes_ok, shape_detail))

    numeric_keys = (AGENT_STATE_KEY, ENVIRONMENT_STATE_KEY, ACTION_KEY)
    numeric_values: list[np.ndarray] = []
    numeric_conversion_ok = True
    for key in numeric_keys:
        try:
            numeric_values.append(np.asarray(values[key], dtype=np.float64))
        except (TypeError, ValueError):
            numeric_conversion_ok = False
            break
    finite = numeric_conversion_ok and all(np.isfinite(value).all() for value in numeric_values)
    checks.append(Check("no NaN / Inf values", finite, "all finite" if finite else "invalid value found"))

    if numeric_conversion_ok and numeric_values:
        in_range = all(np.abs(value).max(initial=0.0) <= 1024.0 for value in numeric_values)
        max_abs = max(float(np.abs(value).max(initial=0.0)) for value in numeric_values)
        range_detail = f"maximum absolute value={max_abs:.1f} (limit=1024.0)"
    else:
        in_range = False
        range_detail = "numeric conversion failed"
    checks.append(Check("state / action range", in_range, range_detail))

    episodes = np.asarray(values[EPISODE_KEY]).reshape(-1)
    if episodes.size:
        _, episode_lengths = np.unique(episodes, return_counts=True)
        shortest = int(episode_lengths.min())
        length_ok = shortest >= 5
        length_detail = f"shortest episode={shortest} frames (minimum=5)"
    else:
        length_ok = False
        length_detail = "no episodes"
    checks.append(Check("episode length", length_ok, length_detail))
    return checks


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"Dataset not found: {args.data}\nRun: python src/download_dataset.py")
    checks = validate(load_validation_data(args.data))
    print("## Dataset Validation")
    print(f"Data: {args.data}")
    print()
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{status}: {check.label} — {check.detail}")
    passed = all(check.passed for check in checks)
    print()
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
