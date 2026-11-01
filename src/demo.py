"""最終課題デモ: 押すAI(ACT)と見るAI(Gemini)の2段構成でPushTを解きます。

1. 学習済みACTが観測からエージェントの次の位置を推論し、T字ブロックを押します。
2. 作業の途中でブロックを突き飛ばし(外乱)、立て直せるかを試します。
3. Geminiが数秒おきに画面を見て「ずれている/もう少し/完了」を判定し、
   完了と判定したらその回の作業を終わらせます。判定は字幕として動画に焼き込みます。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from PIL import Image, ImageDraw, ImageFont

from roboops.policy import denormalize_action, load_normalization, normalize

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_PROMPT = """This is a top-down view of a 2D robot task.
A blue circle (the robot) pushes a gray T-shaped block.
The goal is to place the gray T exactly on top of the light-green T outline.
Judge the current state and answer in JSON:
{"state": "off_target" | "almost" | "done", "reason": "<one short English sentence>"}
- off_target: the gray T is clearly away from or rotated relative to the green T.
- almost: the gray T mostly overlaps the green T but is not aligned yet.
- done: the gray T almost perfectly overlaps the green T."""
STATE_COLORS = {"off_target": (220, 60, 60), "almost": (230, 160, 0), "done": (40, 160, 70)}
CAPTION_HEIGHT = 96  # 384+96=480 で動画の16の倍数に揃える


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACT + Gemini watcher demo in PushT.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/act-final"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", help="回ごとのseedを直接指定(--episodesより優先)")
    parser.add_argument("--no-video", action="store_true", help="成功率の集計用。動画を書き出さない")
    parser.add_argument("--disturb-step", type=int, default=150, help="0で外乱なし")
    # ロボット向けモデルの無料枠は1日20回のため、見せる場面を絞ります。
    # 既定は外乱の直前・直後・その後(--check-after)と、各回の最後の4回です。
    parser.add_argument("--check-every", type=int, default=0, help="一定間隔でも見せる(step)。0で使わない")
    parser.add_argument("--check-after", type=int, default=80, help="外乱の何step後に見せるか")
    parser.add_argument("--no-fallback", action="store_true", help="上限時に軽いモデルへ切り替えない")
    parser.add_argument("--min-interval", type=float, default=13.0, help="無料枠のためのAPI呼び出し間隔(秒)")
    parser.add_argument("--no-gemini", action="store_true")
    parser.add_argument("--video", type=Path, default=Path("artifacts/demo/demo.mp4"))
    parser.add_argument("--log", type=Path, default=Path("artifacts/demo/demo_log.json"))
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


class GeminiWatcher:
    """シミュレーターの画面をGeminiに見せて、作業の状態を判定させます。"""

    def __init__(
        self, api_key: str, model: str, fallback_model: str, min_interval: float
    ) -> None:
        self.api_key = api_key
        self.models = [name for name in (model, fallback_model) if name]
        self.min_interval = min_interval
        self.last_call = 0.0

    def _request(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            GEMINI_URL.format(model=model),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        self.last_call = time.monotonic()
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())

    def judge(self, frame: np.ndarray) -> dict[str, str]:
        wait = self.min_interval - (time.monotonic() - self.last_call)
        if wait > 0:
            time.sleep(wait)
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        body = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                            }
                        },
                        {"text": GEMINI_PROMPT},
                    ]
                }
            ],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
        }
        # ロボット向けモデルの無料枠は小さいため、上限(429)に当たったら軽いモデルで聞き直します。
        # 両方とも上限なら30秒待って、もう一巡だけ試します。
        for attempt in range(2):
            for model in self.models:
                try:
                    payload = self._request(model, body)
                    text = payload["candidates"][0]["content"]["parts"][0]["text"]
                    result = json.loads(text)
                    if isinstance(result, list):
                        result = result[0]
                    state = str(result.get("state", "")).strip()
                    if state not in STATE_COLORS:
                        return {"state": "unknown", "reason": f"unexpected: {text[:60]}", "model": model}
                    return {"state": state, "reason": str(result.get("reason", "")), "model": model}
                except urllib.error.HTTPError as error:
                    if error.code != 429:
                        print(f"Gemini error ({model}): {error} {error.read().decode('utf-8')[:200]}")
                        return {"state": "unknown", "reason": "Gemini call failed", "model": model}
                    print(f"Gemini rate limited ({model})")
                except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as error:
                    print(f"Gemini error ({model}): {error}")
                    return {"state": "unknown", "reason": "Gemini call failed", "model": model}
            if attempt == 0:
                time.sleep(30)
        return {"state": "unknown", "reason": "Gemini rate limited", "model": "-"}


def short_model(model: str) -> str:
    """字幕に、どのモデルが判定したかを短く出します。"""
    if "robotics-er" in model:
        return "Gemini ER"
    if "flash-lite" in model:
        return "Gemini Lite"
    return "Gemini"


def disturb(env: Any, rng: np.random.Generator) -> None:
    """T字ブロックを横から突き飛ばしたように、位置と角度を大きくずらします。"""
    block = env.unwrapped.block
    offset = rng.choice([-1, 1], size=2) * rng.uniform(70, 110, size=2)
    position = np.clip(np.array(block.position) + offset, 120, 392)
    block.position = position.tolist()
    block.angle = block.angle + rng.choice([-1, 1]) * rng.uniform(0.6, 1.2)
    block.velocity = (0, 0)
    block.angular_velocity = 0
    env.unwrapped.space.step(env.unwrapped.dt)


def caption_lines(
    episode: int, total: int, step: int, coverage: float, judgement: dict[str, str], disturbed_now: bool
) -> list[tuple[str, tuple[int, int, int]]]:
    color = STATE_COLORS.get(judgement["state"], (120, 120, 120))
    return [
        (f"Episode {episode}/{total}  step {step}  overlap {coverage:.0%}", (30, 30, 30)),
        (f"{short_model(judgement.get('model', ''))}: {judgement['state']}", color),
        (judgement["reason"][:50], color),
        (
            "!! DISTURBANCE: block knocked away !!" if disturbed_now else "Pusher: ACT (imitation learning)",
            (200, 0, 0) if disturbed_now else (90, 90, 90),
        ),
    ]


def caption(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    image = Image.fromarray(frame)
    canvas = Image.new("RGB", (image.width, image.height + CAPTION_HEIGHT), (255, 255, 255))
    canvas.paste(image, (0, CAPTION_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    for row, (text, color) in enumerate(lines):
        draw.text((8, 4 + row * 22), text, fill=color, font=font)
    return np.asarray(canvas)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
    import gym_pusht  # noqa: F401
    import gymnasium as gym

    load_env_file()
    watcher = None
    if not args.no_gemini:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is not set. Put it in .env or use --no-gemini.")
        model = os.environ.get("GEMINI_MODEL", "gemini-robotics-er-2-preview")
        fallback_model = "" if args.no_fallback else os.environ.get(
            "GEMINI_FALLBACK_MODEL", "gemini-flash-lite-latest"
        )
        watcher = GeminiWatcher(api_key, model, fallback_model, args.min_interval)
        print(f"Watcher: {model} (fallback: {fallback_model or 'none'})")

    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.eval()
    stats = load_normalization(args.checkpoint / "normalization.npz")
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
        visualization_width=384,
        visualization_height=384,
    )
    seeds = args.seeds or [args.seed + episode for episode in range(args.episodes)]
    frames: list[np.ndarray] = []
    episode_logs: list[dict[str, Any]] = []
    try:
        for episode, seed in enumerate(seeds):
            observation, _ = env.reset(seed=seed)
            # 外乱の向きと強さもseedごとに決め、同じseedなら同じ場面を再現できるようにします。
            rng = np.random.default_rng(seed)
            policy.reset()
            judgement = {"state": "unknown", "reason": "not checked yet"}
            checks: list[dict[str, Any]] = []
            events: list[str] = []
            sim_success_step = None
            disturbed_at = None
            best_after_disturbance = 0.0
            end_reason = "max steps"
            for step in range(1, args.max_steps + 1):
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
                observation, _, terminated, truncated, _ = env.step(
                    denormalize_action(normalized_action, stats)
                )
                coverage = float(env.unwrapped._get_coverage())
                if terminated and sim_success_step is None:
                    sim_success_step = step

                if args.disturb_step and step == args.disturb_step:
                    disturb(env, rng)
                    # 外乱で観測が変わるため、ACTの行動チャンクの蓄積をリセットします。
                    policy.reset()
                    disturbed_at = step
                    events.append(f"step {step}: disturbance")
                    observation = env.unwrapped.get_obs()
                    coverage = float(env.unwrapped._get_coverage())

                if disturbed_at is not None:
                    best_after_disturbance = max(best_after_disturbance, coverage)
                frame = None if args.no_video and not watcher else np.asarray(env.render())
                around_disturbance = args.disturb_step and step in (
                    args.disturb_step - 1,
                    args.disturb_step,
                    args.disturb_step + args.check_after,
                )
                periodic = args.check_every and step % args.check_every == 0
                if watcher and (periodic or around_disturbance):
                    judgement = watcher.judge(frame)
                    checks.append({"step": step, "coverage": coverage, **judgement})
                    print(
                        f"Episode {episode + 1} step {step} | coverage={coverage:.2f} "
                        f"| Gemini={judgement['state']} ({judgement['reason']})"
                    )

                disturbed_now = disturbed_at is not None and step - disturbed_at < 15
                if not args.no_video:
                    lines = caption_lines(
                        episode + 1, len(seeds), step, coverage, judgement, disturbed_now
                    )
                    frames.append(caption(frame, lines))

                past_disturbance = not args.disturb_step or (disturbed_at is not None)
                if watcher and judgement["state"] == "done" and past_disturbance:
                    end_reason = "Gemini judged done"
                    break
                if terminated and past_disturbance:
                    end_reason = "simulator success"
                    break
                if truncated:
                    break

            final_coverage = float(env.unwrapped._get_coverage())
            if watcher:
                # 最後の場面もGeminiに見せ、その判定を止め絵の字幕に出します。
                judgement = watcher.judge(frame)
                checks.append({"step": step, "coverage": final_coverage, "final": True, **judgement})
                print(
                    f"Episode {episode + 1} final | coverage={final_coverage:.2f} "
                    f"| Gemini={judgement['state']} ({judgement['reason']})"
                )
                if not args.no_video:
                    lines = caption_lines(
                        episode + 1, len(seeds), step, final_coverage, judgement, False
                    )
                    frames[-1] = caption(frame, lines)
            if not args.no_video:
                # 終わった場面が見えるよう、最後の1コマを2秒ほど止めます。
                frames.extend([frames[-1]] * args.fps * 2)
            episode_logs.append(
                {
                    "episode": episode + 1,
                    "seed": seed,
                    "steps": step,
                    "disturbed_at": disturbed_at,
                    "first_success_step": sim_success_step,
                    "final_coverage": final_coverage,
                    "best_coverage_after_disturbance": best_after_disturbance,
                    "recovered": final_coverage > env.unwrapped.success_threshold,
                    "end_reason": end_reason,
                    "events": events,
                    "gemini_checks": checks,
                }
            )
            print(
                f"Episode {episode + 1}: steps={step} final_coverage={final_coverage:.2f} "
                f"end={end_reason}"
            )
    finally:
        env.close()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(episode_logs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Log: {args.log}")
    if frames:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        # 全コマを1つの配列に積むとメモリが足りなくなるため、1コマずつ書き出します。
        with iio.imopen(args.video, "w", plugin="pyav") as video:
            video.init_video_stream("libx264", fps=args.fps)
            for frame in frames:
                video.write_frame(frame)
        print(f"Video: {args.video} ({len(frames) / args.fps:.0f}s at {args.fps}fps)")


if __name__ == "__main__":
    main()
