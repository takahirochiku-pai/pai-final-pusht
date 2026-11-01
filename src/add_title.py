"""録画済みのデモ動画の冒頭に、課題と構成を説明する画面を足します(Geminiは呼びません)。"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TITLE_LINES = [
    ("PAI Final: PushT with two AIs", 18, (153, 0, 0)),
    ("", 10, (0, 0, 0)),
    ("Task: push the gray T onto the green T.", 14, (30, 30, 30)),
    ("", 8, (0, 0, 0)),
    ("Pusher AI: LeRobot ACT (imitation learning)", 14, (30, 30, 30)),
    ("  moves the blue robot tip step by step", 13, (90, 90, 90)),
    ("Watcher AI: Gemini Robotics-ER 2", 14, (30, 30, 30)),
    ("  looks at the screen and judges", 13, (90, 90, 90)),
    ("  off_target / almost / done", 13, (90, 90, 90)),
    ("", 8, (0, 0, 0)),
    ("At step 150 we knock the block away", 14, (200, 0, 0)),
    ("ON PURPOSE (disturbance) to test", 14, (200, 0, 0)),
    ("whether the pusher can recover.", 14, (200, 0, 0)),
    ("", 8, (0, 0, 0)),
    ("3 episodes: 2 recoveries, 1 failure", 14, (30, 30, 30)),
    ("Overlap % = ground truth from the simulator", 13, (90, 90, 90)),
    ("Playback speed: 1x", 13, (90, 90, 90)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepend a title card to the demo video.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/demo/demo.mp4"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo/demo_final.mp4"))
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def title_card(width: int, height: int) -> np.ndarray:
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    y = 40
    for text, size, color in TITLE_LINES:
        font = ImageFont.load_default(size=size)
        draw.text((16, y), text, fill=color, font=font)
        y += size + 8
    return np.asarray(canvas)


def main() -> None:
    args = parse_args()
    first = next(iio.imiter(args.input, plugin="pyav"))
    card = title_card(first.shape[1], first.shape[0])
    # 全コマを配列に積むとメモリが足りなくなるため、1コマずつ読んで書きます。
    with iio.imopen(args.output, "w", plugin="pyav") as video:
        video.init_video_stream("libx264", fps=args.fps)
        for _ in range(round(args.seconds * args.fps)):
            video.write_frame(card)
        count = 0
        for frame in iio.imiter(args.input, plugin="pyav"):
            video.write_frame(frame)
            count += 1
    total = round(args.seconds * args.fps) + count
    print(f"Video: {args.output} ({total / args.fps:.0f}s at {args.fps}fps)")


if __name__ == "__main__":
    main()
