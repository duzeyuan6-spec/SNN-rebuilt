import argparse
import os
import socket
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

SOURCES = {
    "bsds500": [
        "https://gh-proxy.com/https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip",
        "https://gh.llkk.cc/https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip",
        "https://mirror.ghproxy.com/https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip",
        "https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip",
        "https://codeload.github.com/BIDS/BSDS500/zip/refs/heads/master",
        "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz",
    ],
    "dncnn": [
        "https://gh-proxy.com/https://github.com/cszn/DnCNN/archive/refs/heads/master.zip",
        "https://gh.llkk.cc/https://github.com/cszn/DnCNN/archive/refs/heads/master.zip",
        "https://mirror.ghproxy.com/https://github.com/cszn/DnCNN/archive/refs/heads/master.zip",
        "https://github.com/cszn/DnCNN/archive/refs/heads/master.zip",
        "https://codeload.github.com/cszn/DnCNN/zip/refs/heads/master",
    ],
    "reconnet": [
        "https://gh-proxy.com/https://github.com/KuldeepKulkarni/ReconNet/archive/refs/heads/master.zip",
        "https://gh.llkk.cc/https://github.com/KuldeepKulkarni/ReconNet/archive/refs/heads/master.zip",
        "https://mirror.ghproxy.com/https://github.com/KuldeepKulkarni/ReconNet/archive/refs/heads/master.zip",
        "https://github.com/KuldeepKulkarni/ReconNet/archive/refs/heads/master.zip",
        "https://codeload.github.com/KuldeepKulkarni/ReconNet/zip/refs/heads/master",
    ],
}


def download(url, dst, timeout=30):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {dst.name} already exists")
        return
    print(f"[download] {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    with urlopen(url, timeout=timeout) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp.rename(dst)


def download_any(urls, dst, timeout=30):
    errors = []
    for url in urls:
        try:
            download(url, dst, timeout=timeout)
            return
        except (OSError, socket.timeout, TimeoutError) as exc:
            errors.append(f"{url}: {exc}")
            print(f"[failed] {url}: {exc}")
    raise RuntimeError("All download mirrors failed:\n" + "\n".join(errors))


def extract(archive, out_dir):
    marker = out_dir / ".extracted"
    if marker.exists():
        print(f"[skip] {archive.name} already extracted")
        return

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive}")

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(out_dir)
    elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(out_dir)
    else:
        raise ValueError(f"Unsupported archive: {archive}")

    marker.touch()


def image_files(root):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def copy_images(src_dir, dst_dir, prefix):
    files = image_files(src_dir)
    if not files:
        raise FileNotFoundError(f"No images found under {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, src in enumerate(files):
        ext = ".png" if src.suffix.lower() == ".png" else src.suffix.lower()
        dst = dst_dir / f"{prefix}_{i:04d}{ext}"
        if not dst.exists():
            shutil.copy2(src, dst)
        count += 1
    print(f"[copy] {count} images: {src_dir} -> {dst_dir}")


def find_named_dir(root, name):
    name_lower = name.lower()
    matches = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == name_lower]
    if not matches:
        raise FileNotFoundError(f"Cannot find directory named {name} under {root}")
    matches.sort(key=lambda p: len(str(p)))
    return matches[0]


def prepare_train400(cache_dir, data_root):
    archive = cache_dir / "BSDS500.zip"
    out_dir = cache_dir / "bsds500"
    download_any(SOURCES["bsds500"], archive)
    extract(archive, out_dir)

    train_dir = find_named_dir(out_dir, "train")
    test_dir = find_named_dir(out_dir, "test")

    dst = data_root / "train" / "Train400"
    dst.mkdir(parents=True, exist_ok=True)
    copy_images(train_dir, dst, "bsds_train")
    copy_images(test_dir, dst, "bsds_test")


def prepare_dncnn_tests(cache_dir, data_root):
    archive = cache_dir / "DnCNN-master.zip"
    out_dir = cache_dir / "dncnn"
    download_any(SOURCES["dncnn"], archive)
    extract(archive, out_dir)

    for name in ["BSD68", "Urban100"]:
        src = find_named_dir(out_dir, name)
        dst = data_root / "test" / name
        copy_images(src, dst, name.lower())


def prepare_set11(cache_dir, data_root):
    archive = cache_dir / "ReconNet-master.zip"
    out_dir = cache_dir / "reconnet"
    download_any(SOURCES["reconnet"], archive)
    extract(archive, out_dir)

    src = find_named_dir(out_dir, "Set11")
    dst = data_root / "test" / "Set11"
    copy_images(src, dst, "set11")


def summarize(data_root):
    print("\nDataset summary:")
    for rel in ["train/Train400", "test/Set11", "test/BSD68", "test/Urban100"]:
        root = data_root / rel
        print(f"  {rel}: {len(image_files(root)) if root.exists() else 0} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--skip-train400", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    cache_dir = Path(args.cache_dir or os.environ.get("SPIKECS_CACHE", "/root/gpufree-data/spikecs_cache")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"data_root: {data_root}")
    print(f"cache_dir: {cache_dir}")

    if not args.skip_train400:
        prepare_train400(cache_dir, data_root)
    if not args.skip_tests:
        prepare_dncnn_tests(cache_dir, data_root)
        prepare_set11(cache_dir, data_root)

    summarize(data_root)


if __name__ == "__main__":
    main()
