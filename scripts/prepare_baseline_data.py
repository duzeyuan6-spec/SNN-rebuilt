import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(root):
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"No supported images found under {root}")
    return files


def load_gray_uint8(path):
    return Image.open(path).convert("L")


def save_image_variants(img, stem, png_dir, tif_dir):
    png_dir.mkdir(parents=True, exist_ok=True)
    tif_dir.mkdir(parents=True, exist_ok=True)
    img.save(png_dir / f"{stem}.png")
    img.save(tif_dir / f"{stem}.tif")


def save_mat_variants(img, stem, mat_dir):
    mat_dir.mkdir(parents=True, exist_ok=True)
    arr_u8 = np.asarray(img, dtype=np.uint8)
    arr_f32 = arr_u8.astype(np.float32) / 255.0

    # Different CS repositories use different key names. Saving several common
    # aliases lets most MATLAB/Python loaders work with minimal or no edits.
    savemat(
        mat_dir / f"{stem}.mat",
        {
            "img": arr_f32,
            "im": arr_f32,
            "image": arr_f32,
            "label": arr_f32,
            "x": arr_f32,
            "im_gt_y": arr_f32,
            "I": arr_u8,
        },
    )


def export_split(src_root, split_name, out_root):
    files = list_images(src_root)
    png_dir = out_root / "images" / split_name
    tif_dir = out_root / "tif" / split_name
    mat_dir = out_root / "mat" / f"{split_name}_mat"

    for idx, path in enumerate(files, 1):
        img = load_gray_uint8(path)
        stem = f"{idx:04d}_{path.stem}"
        save_image_variants(img, stem, png_dir, tif_dir)
        save_mat_variants(img, stem, mat_dir)

    print(f"{split_name}: exported {len(files)} images")
    print(f"  png: {png_dir}")
    print(f"  tif: {tif_dir}")
    print(f"  mat: {mat_dir}")


def mirror_to_baselines(out_root, project_root):
    baselines = project_root / "baselines"

    plv = baselines / "PLV-CSNet"
    if plv.exists():
        for split in ["Set11", "BSD68", "Urban100"]:
            src = out_root / "mat" / f"{split}_mat"
            dst = plv / "test" / f"{split}_mat"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"mirrored PLV test {split}: {dst}")
        train_src = out_root / "mat" / "Train400_mat"
        train_dst = plv / "train" / "Train400_mat"
        if train_dst.exists():
            shutil.rmtree(train_dst)
        shutil.copytree(train_src, train_dst)
        print(f"mirrored PLV train: {train_dst}")

    ista = baselines / "ISTA-Net"
    if ista.exists():
        for split in ["Set11", "BSD68", "Urban100"]:
            src = out_root / "tif" / split
            dst = ista / f"Test_Image_{split}"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"mirrored ISTA test {split}: {dst}")
        train_src = out_root / "tif" / "Train400"
        train_dst = ista / "Train_Image_Train400"
        if train_dst.exists():
            shutil.rmtree(train_dst)
        shutil.copytree(train_src, train_dst)
        print(f"mirrored ISTA train: {train_dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-root", default="baseline_data")
    parser.add_argument("--mirror", action="store_true", help="copy exported data into known baseline folders")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root
    if not out_root.is_absolute():
        out_root = project_root / out_root

    splits = {
        "Train400": data_root / "train" / "Train400",
        "Set11": data_root / "test" / "Set11",
        "BSD68": data_root / "test" / "BSD68",
        "Urban100": data_root / "test" / "Urban100",
    }

    for split_name, src_root in splits.items():
        export_split(src_root, split_name, out_root)

    if args.mirror:
        mirror_to_baselines(out_root, project_root)


if __name__ == "__main__":
    main()
