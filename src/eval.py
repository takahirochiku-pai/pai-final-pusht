from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy

from roboops.config import load_yaml
from roboops.policy import denormalize_action, load_normalization, normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the official LeRobot ACT in PushT.")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_ci.yaml"))
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    config = load_yaml(args.config)
    evaluation = config["evaluation"]
    if bool(evaluation["headless"]):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    import gym_pusht  # noqa: F401
    import gymnasium as gym

    checkpoint_dir = Path(evaluation["checkpoint"])
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_dir}. "
            "Run: python src/train.py --config configs/train_ci.yaml"
        )
    policy = ACTPolicy.from_pretrained(checkpoint_dir)
    policy.eval()
    stats = load_normalization(checkpoint_dir / "normalization.npz")
    training_metrics = json.loads(
        (checkpoint_dir / "training_metrics.json").read_text(encoding="utf-8")
    )

    episodes = int(evaluation["episodes"])
    max_steps = int(evaluation["max_steps"])
    seed = int(evaluation["seed"])
    save_video = args.save_video or bool(evaluation["save_video"])
    frames: list[np.ndarray] = []
    episode_results: list[dict[str, Any]] = []
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="rgb_array" if save_video else None,
        max_episode_steps=max_steps,
        visualization_width=384,
        visualization_height=384,
    )

    print("## Official LeRobot ACT CI Evaluation")
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            policy.reset()
            initial_coverage = env.unwrapped._get_coverage()
            initial_reward = float(
                np.clip(initial_coverage / env.unwrapped.success_threshold, 0.0, 1.0)
            )
            maximum_reward = initial_reward
            reward_sum = 0.0
            success = False
            steps = 0
            if save_video and episode == 0:
                frames.append(np.asarray(env.render()))
            for _ in range(max_steps):
                batch = {
                    "observation.state": torch.from_numpy(
                        normalize(observation["agent_pos"], stats, "observation.state")
                    ).float().unsqueeze(0),
                    "observation.environment_state": torch.from_numpy(
                        normalize(
                            observation["environment_state"],
                            stats,
                            "observation.environment_state",
                        )
                    ).float().unsqueeze(0),
                }
                with torch.inference_mode():
                    normalized_action = policy.select_action(batch)[0].cpu().numpy()
                action = denormalize_action(normalized_action, stats)
                observation, reward, terminated, truncated, info = env.step(action)
                steps += 1
                reward_sum += float(reward)
                maximum_reward = max(maximum_reward, float(reward))
                success = success or bool(info.get("is_success", False))
                if save_video and episode == 0:
                    frames.append(np.asarray(env.render()))
                if terminated or truncated:
                    break
            episode_results.append(
                {
                    "episode": episode + 1,
                    "seed": seed + episode,
                    "steps": steps,
                    "initial_reward": initial_reward,
                    "max_reward": maximum_reward,
                    "reward_gain": maximum_reward - initial_reward,
                    "mean_step_reward": reward_sum / max(steps, 1),
                    "success": success,
                }
            )
            print(
                f"Episode {episode + 1}/{episodes} | max_reward={maximum_reward:.3f} "
                f"| reward_gain={maximum_reward - initial_reward:.3f} "
                f"| success={'yes' if success else 'no'}"
            )
    finally:
        env.close()

    if save_video and frames:
        video_path = Path(evaluation["video_path"])
        video_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(video_path, np.stack(frames), fps=10, codec="libx264")
        print(f"Video: {video_path}")

    mean_reward = float(np.mean([item["max_reward"] for item in episode_results]))
    mean_reward_gain = float(np.mean([item["reward_gain"] for item in episode_results]))
    success_rate = float(np.mean([item["success"] for item in episode_results]))
    min_reward = float(evaluation["min_mean_reward"])
    min_reward_gain = float(evaluation["min_mean_reward_gain"])
    passed = mean_reward >= min_reward and mean_reward_gain >= min_reward_gain
    elapsed_seconds = time.perf_counter() - started_at
    metrics = {
        "result": "PASS" if passed else "FAIL",
        "architecture": "lerobot_act",
        "environment": "gym_pusht/PushT-v0",
        "checkpoint": str(checkpoint_dir),
        "dataset": training_metrics["dataset"],
        "model": training_metrics["model"],
        "versions": {
            "lerobot": importlib.metadata.version("lerobot"),
            "gym-pusht": importlib.metadata.version("gym-pusht"),
        },
        "episodes": episodes,
        "mean_max_reward": mean_reward,
        "mean_reward_gain": mean_reward_gain,
        "success_rate": success_rate,
        "elapsed_seconds": elapsed_seconds,
        "thresholds": {
            "min_mean_reward": min_reward,
            "min_mean_reward_gain": min_reward_gain,
        },
        "episode_results": episode_results,
    }
    metrics_path = Path(config["output"]["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Mean max reward: {mean_reward:.3f}")
    print(f"Mean reward gain: {mean_reward_gain:.3f}")
    print(f"Success rate: {success_rate:.0%}")
    print(f"Evaluation: {elapsed_seconds:.1f}s")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
