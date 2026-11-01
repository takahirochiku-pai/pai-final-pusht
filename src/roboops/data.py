from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AGENT_STATE_KEY = "observation.state"
ENVIRONMENT_STATE_KEY = "observation.environment_state"
ACTION_KEY = "action"
EPISODE_KEY = "episode_index"
FRAME_KEY = "frame_index"
REQUIRED_KEYS = (AGENT_STATE_KEY, ENVIRONMENT_STATE_KEY, ACTION_KEY, EPISODE_KEY)


@dataclass(frozen=True)
class DatasetBundle:
    agent_state: np.ndarray
    environment_state: np.ndarray
    action: np.ndarray
    episode_index: np.ndarray
    frame_index: np.ndarray
    dataset_id: str = "unknown"
    dataset_revision: str = "unknown"

    @property
    def state(self) -> np.ndarray:
        """State-only input: agent position followed by 8 PushT keypoints."""
        return np.concatenate((self.agent_state, self.environment_state), axis=1)

    @property
    def episode_count(self) -> int:
        return int(np.unique(self.episode_index).size)

    @property
    def frame_count(self) -> int:
        return int(self.action.shape[0])


def _read_scalar_text(values: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in values.files:
        return default
    return str(np.asarray(values[key]).item())


def load_npz_dataset(path: str | Path) -> DatasetBundle:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Run: python src/download_dataset.py"
        )
    with np.load(dataset_path, allow_pickle=False) as values:
        missing = [key for key in REQUIRED_KEYS if key not in values.files]
        if missing:
            raise ValueError(f"Dataset is missing required keys: {', '.join(missing)}")
        agent_state = np.asarray(values[AGENT_STATE_KEY], dtype=np.float32)
        environment_state = np.asarray(values[ENVIRONMENT_STATE_KEY], dtype=np.float32)
        action = np.asarray(values[ACTION_KEY], dtype=np.float32)
        episode_index = np.asarray(values[EPISODE_KEY], dtype=np.int64).reshape(-1)
        if FRAME_KEY in values.files:
            frame_index = np.asarray(values[FRAME_KEY], dtype=np.int64).reshape(-1)
        else:
            frame_index = np.arange(action.shape[0], dtype=np.int64)
        dataset_id = _read_scalar_text(values, "dataset_id", "unknown")
        revision = _read_scalar_text(values, "dataset_revision", "unknown")
    return DatasetBundle(
        agent_state=agent_state,
        environment_state=environment_state,
        action=action,
        episode_index=episode_index,
        frame_index=frame_index,
        dataset_id=dataset_id,
        dataset_revision=revision,
    )


def load_validation_data(path: str | Path) -> dict[str, np.ndarray]:
    """Load either the workshop NPZ or the deliberately broken JSON sample."""
    dataset_path = Path(path)
    if dataset_path.suffix == ".json":
        with dataset_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return {key: np.asarray(value) for key, value in payload.items()}
    with np.load(dataset_path, allow_pickle=False) as values:
        return {key: np.asarray(values[key]) for key in values.files}


def normalization_stats(
    bundle: DatasetBundle,
    indices: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    values = {
        AGENT_STATE_KEY: bundle.agent_state[indices],
        ENVIRONMENT_STATE_KEY: bundle.environment_state[indices],
        ACTION_KEY: bundle.action[indices],
    }
    return {
        key: {
            "mean": value.mean(axis=0).astype(np.float32),
            "std": np.maximum(value.std(axis=0), 1e-6).astype(np.float32),
        }
        for key, value in values.items()
    }


def action_chunks(
    bundle: DatasetBundle,
    indices: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """episodeの境界を越えない未来actionとpadding maskを作ります。"""
    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    chunks = np.empty((selected.size, chunk_size, bundle.action.shape[1]), dtype=np.float32)
    padding = np.ones((selected.size, chunk_size), dtype=np.bool_)
    selected_rows = {int(global_index): row for row, global_index in enumerate(selected)}

    for episode in np.unique(bundle.episode_index):
        positions = np.flatnonzero(bundle.episode_index == episode)
        positions = positions[np.argsort(bundle.frame_index[positions], kind="stable")]
        for offset, global_index in enumerate(positions):
            row = selected_rows.get(int(global_index))
            if row is None:
                continue
            future = positions[offset : offset + chunk_size]
            valid_count = int(future.size)
            chunks[row, :valid_count] = bundle.action[future]
            chunks[row, valid_count:] = bundle.action[future[-1]]
            padding[row, :valid_count] = False
    return chunks, padding
