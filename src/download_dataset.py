from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

DATASET_ID = "lerobot/pusht_keypoints"
DATASET_REVISION = "ace8c161a68bc025c21a5f29f85b86a9a2c5e64b"
DEFAULT_OUTPUT = Path("data/processed/pusht_keypoints.npz")
DEFAULT_RAW_ROOT = Path("data/raw/lerobot_pusht_keypoints") / DATASET_REVISION
DATASET_FILES = (
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
    "data/chunk-000/file-000.parquet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small state-only LeRobot PushT dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=60,
        help="Number of episodes to materialize (default: 60 of 206).",
    )
    parser.add_argument("--force", action="store_true", help="Replace the processed NPZ if it exists.")
    return parser.parse_args()


def _to_numpy(value: object, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _download_pinned_files(raw_root: Path) -> None:
    """Fetch exact files without the Hub tree API used by snapshot_download.

    GitHub-hosted runners share outbound IP addresses, so unauthenticated tree
    listing can be rate-limited even for this public Dataset. The pinned
    revision has one data chunk, making direct file downloads deterministic.
    """
    for filename in DATASET_FILES:
        hf_hub_download(
            repo_id=DATASET_ID,
            filename=filename,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_dir=raw_root,
        )


def _cache_matches(output: Path, manifest_path: Path, requested_episodes: int) -> bool:
    if not output.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("dataset_id") == DATASET_ID
        and manifest.get("dataset_revision") == DATASET_REVISION
        and manifest.get("requested_episodes", manifest.get("episodes")) == requested_episodes
    )


def main() -> None:
    # この教材は動画を取得しないため、利用しないTorchCodecのfallback警告は表示しない。
    logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)
    # 公開Datasetは未認証でも取得できるため、rate-limit案内で初学者を迷わせない。
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

    args = parse_args()
    if args.max_episodes < 1:
        raise SystemExit("--max-episodes must be at least 1")
    manifest_path = args.output.with_suffix(".manifest.json")
    if not args.force and _cache_matches(args.output, manifest_path, args.max_episodes):
        print(f"Dataset already exists: {args.output}")
        print("Cached revision and episode count match the request.")
        return

    print("## Dataset Download")
    print(f"Source: https://huggingface.co/datasets/{DATASET_ID}")
    print(f"Pinned revision: {DATASET_REVISION[:12]}")

    _download_pinned_files(args.raw_root)
    metadata = LeRobotDatasetMetadata(
        DATASET_ID,
        root=args.raw_root,
        revision=DATASET_REVISION,
    )
    episode_count = min(args.max_episodes, metadata.total_episodes)
    episodes = list(range(episode_count))
    dataset = LeRobotDataset(
        DATASET_ID,
        root=args.raw_root,
        episodes=episodes,
        revision=DATASET_REVISION,
        download_videos=False,
    )

    agent_states: list[np.ndarray] = []
    environment_states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_indices: list[int] = []
    frame_indices: list[int] = []
    for sample in dataset:
        agent_states.append(_to_numpy(sample["observation.state"], np.float32))
        environment_states.append(
            _to_numpy(sample["observation.environment_state"], np.float32)
        )
        actions.append(_to_numpy(sample["action"], np.float32))
        episode_indices.append(int(_to_numpy(sample["episode_index"], np.int64).item()))
        frame_indices.append(int(_to_numpy(sample["frame_index"], np.int64).item()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **{
            "observation.state": np.stack(agent_states),
            "observation.environment_state": np.stack(environment_states),
            "action": np.stack(actions),
            "episode_index": np.asarray(episode_indices, dtype=np.int64),
            "frame_index": np.asarray(frame_indices, dtype=np.int64),
            "dataset_id": np.asarray(DATASET_ID),
            "dataset_revision": np.asarray(DATASET_REVISION),
            "fps": np.asarray(metadata.fps, dtype=np.int64),
        },
    )
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "requested_episodes": args.max_episodes,
        "episodes": episode_count,
        "frames": len(actions),
        "features": {
            "observation.state": [2],
            "observation.environment_state": [16],
            "action": [2],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Episodes: {episode_count}")
    print(f"Frames: {len(actions)}")
    print(f"Saved: {args.output}")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
