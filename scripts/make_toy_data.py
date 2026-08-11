import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_image(seed, size=160):
    rng = np.random.default_rng(seed)
    img = Image.new("L", (size, size), int(rng.integers(30, 180)))
    draw = ImageDraw.Draw(img)
    for _ in range(18):
        x0, y0 = rng.integers(0, size, 2)
        x1, y1 = rng.integers(0, size, 2)
        gray = int(rng.integers(20, 240))
        width = int(rng.integers(1, 5))
        draw.line((x0, y0, x1, y1), fill=gray, width=width)
    for _ in range(8):
        x0, y0 = rng.integers(0, size - 20, 2)
        x1 = int(min(size, x0 + rng.integers(10, 60)))
        y1 = int(min(size, y0 + rng.integers(10, 60)))
        gray = int(rng.integers(20, 240))
        draw.rectangle((x0, y0, x1, y1), outline=gray, width=2)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--test", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root)
    train_dir = root / "train" / "Train400"
    test_dir = root / "test" / "Set11"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.train):
        make_image(i).save(train_dir / f"{i:03d}.png")
    for i in range(args.test):
        make_image(1000 + i).save(test_dir / f"{i:03d}.png")

    print(f"Wrote toy data to {root.resolve()}")


if __name__ == "__main__":
    main()

