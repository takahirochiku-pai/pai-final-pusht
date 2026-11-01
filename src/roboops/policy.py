from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def build_act_policy(model_config: dict[str, Any]) -> ACTPolicy:
    """LeRobot公式ACTをCI向けのstate-only構成で作ります。"""
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
            "observation.environment_state": PolicyFeature(FeatureType.ENV, (16,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        chunk_size=int(model_config["chunk_size"]),
        n_action_steps=1,
        pre_norm=True,
        dim_model=int(model_config["dim_model"]),
        n_heads=int(model_config["n_heads"]),
        dim_feedforward=int(model_config["dim_feedforward"]),
        n_encoder_layers=int(model_config["n_encoder_layers"]),
        n_decoder_layers=int(model_config["n_decoder_layers"]),
        use_vae=bool(model_config["use_vae"]),
        temporal_ensemble_coeff=float(model_config["temporal_ensemble_coeff"]),
        dropout=float(model_config["dropout"]),
        device="cpu",
    )
    return ACTPolicy(config)


def save_normalization(
    path: str | Path,
    stats: dict[str, dict[str, np.ndarray]],
) -> None:
    np.savez(
        path,
        **{
            f"{feature}.{stat}": value
            for feature, values in stats.items()
            for stat, value in values.items()
        },
    )


def load_normalization(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as values:
        for key in values.files:
            feature, stat = key.rsplit(".", 1)
            stats.setdefault(feature, {})[stat] = np.asarray(values[key], dtype=np.float32)
    return stats


def normalize(
    value: np.ndarray,
    stats: dict[str, dict[str, np.ndarray]],
    feature: str,
) -> np.ndarray:
    return (np.asarray(value, dtype=np.float32) - stats[feature]["mean"]) / stats[feature][
        "std"
    ]


def denormalize_action(
    value: np.ndarray,
    stats: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    action = value * stats["action"]["std"] + stats["action"]["mean"]
    return np.clip(action, 0.0, 512.0).astype(np.float32)
