import argparse
from collections import Counter
from pathlib import Path


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def summarize(path):
    root = Path(path)
    files = [p for p in root.rglob("*") if p.is_file()]
    suffixes = Counter(p.suffix.lower() or "<no_ext>" for p in files)
    images = [p for p in files if p.suffix.lower() in IMG_EXTS]
    print(f"\n{root}")
    print(f"  files: {len(files)}")
    print(f"  supported images: {len(images)}")
    print(f"  suffixes: {dict(suffixes)}")
    for p in images[:8]:
        print(f"  image: {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        summarize(path)


if __name__ == "__main__":
    main()

