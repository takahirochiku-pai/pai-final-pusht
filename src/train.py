from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from roboops.config import load_yaml
from roboops.data import DatasetBundle, action_chunks, load_npz_dataset, normalization_stats
from roboops.policy import build_act_policy, normalize, save_normalization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small official LeRobot ACT for CI.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_ci.yaml"))
    return parser.parse_args()


def split_indices(
    bundle: DatasetBundle,
    validation_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    # 隣接chunkの漏洩を避けるため、episode単位で分割します。
    episodes = np.unique(bundle.episode_index)
    if episodes.size < 2:
        raise ValueError("Training needs at least two episodes")
    shuffled = np.random.default_rng(seed).permutation(episodes)
    validation_count = max(1, round(episodes.size * validation_ratio))
    validation_mask = np.isin(bundle.episode_index, shuffled[:validation_count])
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def make_loader(
    bundle: DatasetBundle,
    indices: np.ndarray,
    stats: dict[str, dict[str, np.ndarray]],
    chunk_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    chunks, padding = action_chunks(bundle, indices, chunk_size)
    dataset = TensorDataset(
        torch.from_numpy(normalize(bundle.agent_state[indices], stats, "observation.state")),
        torch.from_numpy(
            normalize(
                bundle.environment_state[indices],
                stats,
                "observation.environment_state",
            )
        ),
        torch.from_numpy(normalize(chunks, stats, "action")),
        torch.from_numpy(padding),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def act_batch(values: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    state, environment, action, padding = values
    return {
        "observation.state": state.float(),
        "observation.environment_state": environment.float(),
        "action": action.float(),
        "action_is_pad": padding,
    }


def evaluate_loss(policy, loader: DataLoader) -> float:
    policy.eval()
    losses = []
    with torch.inference_mode():
        for values in loader:
            loss, _ = policy(act_batch(values))
            losses.append(float(loss.item()))
    return float(np.mean(losses))


def main() -> None:
    started_at = time.perf_counter()
    config = load_yaml(parse_args().config)
    training = config["training"]
    model_config = config["model"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(training["torch_threads"]))

    bundle = load_npz_dataset(config["dataset"]["path"])
    train_indices, validation_indices = split_indices(
        bundle,
        float(training["validation_ratio"]),
        seed,
    )
    stats = normalization_stats(bundle, train_indices)
    loader_args = {
        "bundle": bundle,
        "stats": stats,
        "chunk_size": int(model_config["chunk_size"]),
        "batch_size": int(training["batch_size"]),
        "seed": seed,
    }
    train_loader = make_loader(indices=train_indices, shuffle=True, **loader_args)
    validation_loader = make_loader(indices=validation_indices, shuffle=False, **loader_args)

    policy = build_act_policy(model_config)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    max_steps = int(training["max_steps"])
    history: list[dict[str, float | int]] = []
    global_step = 0

    print("## Official LeRobot ACT CI Training")
    print(f"Dataset: {bundle.dataset_id} ({bundle.episode_count} episodes)")
    print(f"Parameters: {sum(parameter.numel() for parameter in policy.parameters()):,}")
    for epoch in range(1, epochs + 1):
        policy.train()
        train_losses = []
        for values in train_loader:
            loss, _ = policy(act_batch(values))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().item()))
            global_step += 1
            if global_step >= max_steps:
                break
        train_loss = float(np.mean(train_losses))
        validation_loss = evaluate_loss(policy, validation_loader)
        history.append(
            {
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | step={global_step:03d} "
            f"| train_loss={train_loss:.4f} | validation_loss={validation_loss:.4f}"
        )
        if global_step >= max_steps:
            break

    output = config["output"]
    checkpoint_dir = Path(output["checkpoint"])
    metrics_path = Path(output["metrics"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)
    save_normalization(checkpoint_dir / "normalization.npz", stats)

    elapsed_seconds = time.perf_counter() - started_at
    metrics = {
        "result": "PASS",
        "architecture": "lerobot_act",
        "device": "cpu",
        "checkpoint": str(checkpoint_dir),
        "dataset": {
            "id": bundle.dataset_id,
            "revision": bundle.dataset_revision,
            "episodes": bundle.episode_count,
            "frames": bundle.frame_count,
        },
        "model": model_config,
        "training": training,
        "steps": global_step,
        "parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "elapsed_seconds": elapsed_seconds,
        "history": history,
        "versions": {"lerobot": importlib.metadata.version("lerobot")},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    checkpoint_size = sum(path.stat().st_size for path in checkpoint_dir.glob("*") if path.is_file())
    print(f"Checkpoint: {checkpoint_dir} ({checkpoint_size / 1024:.1f} KiB)")
    print(f"Training: {elapsed_seconds:.1f}s")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
