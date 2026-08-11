import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(root):
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not files:
        raise FileNotFoundError(f"No images found under {root}")
    return sorted(files)


def load_gray(path):
    return Image.open(path).convert("L")


def pil_to_tensor(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class PatchImageDataset(Dataset):
    def __init__(self, root, patch_size=99, augment=True):
        self.files = list_images(root)
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = load_gray(self.files[idx])
        w, h = img.size
        ps = self.patch_size

        if w < ps or h < ps:
            scale = max(ps / w, ps / h)
            img = img.resize((int(w * scale) + 1, int(h * scale) + 1), Image.BICUBIC)
            w, h = img.size

        left = random.randint(0, w - ps)
        top = random.randint(0, h - ps)
        img = img.crop((left, top, left + ps, top + ps))

        if self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            k = random.randint(0, 3)
            if k:
                img = img.rotate(90 * k)

        return pil_to_tensor(img)


class FullImageDataset(Dataset):
    def __init__(self, root, block_size=33):
        self.files = list_images(root)
        self.block_size = block_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = load_gray(self.files[idx])
        x = pil_to_tensor(img)
        _, h, w = x.shape
        bs = self.block_size
        h2 = h - h % bs
        w2 = w - w % bs
        x = x[:, :h2, :w2]
        return x, self.files[idx].name


def collate_full_images(batch):
    return batch
