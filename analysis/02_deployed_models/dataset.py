"""
ACDC data handling: stratified patient-level train/val split + a 2D slice dataset.

Key design decisions (do not change without thinking):
  * Split is BY PATIENT, never by slice (no leakage of adjacent slices).
  * Split is STRATIFIED by diagnostic group so EF distributions stay comparable.
  * Each (patient, frame=ED|ES, slice) is one 2D training sample.
  * EF / volume measurement is done per-PATIENT elsewhere (eval), not here.
"""
import glob
import os
import random
from collections import defaultdict

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

import config as C


# ----------------------------- metadata -----------------------------
def parse_info_cfg(patient_dir):
    info = {}
    with open(os.path.join(patient_dir, "Info.cfg")) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    return {"ED": int(info["ED"]), "ES": int(info["ES"]), "Group": info["Group"]}


def list_patients(root):
    return sorted(glob.glob(os.path.join(str(root), "patient*")))


# ----------------------- stratified patient split -------------------
def stratified_split(train_root, val_fraction, seed):
    """Return (train_patient_dirs, val_patient_dirs), stratified by Group."""
    by_group = defaultdict(list)
    for pdir in list_patients(train_root):
        g = parse_info_cfg(pdir)["Group"]
        by_group[g].append(pdir)

    rng = random.Random(seed)
    train, val = [], []
    for g, dirs in by_group.items():
        dirs = sorted(dirs)
        rng.shuffle(dirs)
        # at least 1 in val, but never leave the group's train empty
        n_val = max(1, round(len(dirs) * val_fraction))
        n_val = min(n_val, len(dirs) - 1) if len(dirs) > 1 else 0
        val.extend(dirs[:n_val])
        train.extend(dirs[n_val:])
    return sorted(train), sorted(val)


# --------------------------- slice indexing -------------------------
def frame_file(patient_dir, pid, frame_idx, gt=False):
    suffix = "_gt" if gt else ""
    return os.path.join(patient_dir, f"{pid}_frame{frame_idx:02d}{suffix}.nii.gz")


def build_slice_index(patient_dirs):
    """Flatten patients into a list of slice records (one per 2D slice)."""
    records = []
    for pdir in patient_dirs:
        pid = os.path.basename(pdir)
        info = parse_info_cfg(pdir)
        for phase, fidx in (("ED", info["ED"]), ("ES", info["ES"])):
            img_path = frame_file(pdir, pid, fidx, gt=False)
            gt_path = frame_file(pdir, pid, fidx, gt=True)
            n_slices = nib.load(img_path).shape[2]
            for s in range(n_slices):
                records.append({
                    "patient": pid, "group": info["Group"], "phase": phase,
                    "img_path": img_path, "gt_path": gt_path, "slice": s,
                })
    return records


# --------------------------- preprocessing --------------------------
def _resample_inplane(slice2d, src_spacing, target_spacing, is_mask):
    """Simple in-plane resample by zoom. Nearest for masks, linear for images."""
    from scipy.ndimage import zoom
    zoom_factors = (src_spacing[0] / target_spacing[0],
                    src_spacing[1] / target_spacing[1])
    order = 0 if is_mask else 1
    return zoom(slice2d, zoom_factors, order=order)


def _crop_or_pad(arr, size, pad_value=0):
    """Center crop or pad a 2D array to `size`."""
    out = np.full(size, pad_value, dtype=arr.dtype)
    h, w = arr.shape
    th, tw = size
    # crop
    sh = max(0, (h - th) // 2); sw = max(0, (w - tw) // 2)
    arr_c = arr[sh:sh + th, sw:sw + tw]
    # pad into center
    ch, cw = arr_c.shape
    dh = (th - ch) // 2; dw = (tw - cw) // 2
    out[dh:dh + ch, dw:dw + cw] = arr_c
    return out


def _normalize(img2d):
    """Z-score per slice on nonzero region; robust to empty slices."""
    m = img2d[img2d > 0]
    if m.size == 0:
        return img2d.astype(np.float32)
    mean, std = m.mean(), m.std() + 1e-6
    return ((img2d - mean) / std).astype(np.float32)


# ----------------------------- dataset ------------------------------
class ACDCSliceDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img_nii = nib.load(r["img_path"])
        gt_nii = nib.load(r["gt_path"])
        sx, sy = img_nii.header.get_zooms()[:2]

        img = np.asarray(img_nii.dataobj)[:, :, r["slice"]].astype(np.float32)
        gt = np.asarray(gt_nii.dataobj)[:, :, r["slice"]].astype(np.int64)

        img = _resample_inplane(img, (sx, sy), C.TARGET_SPACING, is_mask=False)
        gt = _resample_inplane(gt, (sx, sy), C.TARGET_SPACING, is_mask=True)

        img = _crop_or_pad(img, C.PATCH_SIZE, pad_value=0)
        gt = _crop_or_pad(gt, C.PATCH_SIZE, pad_value=0)

        img = _normalize(img)

        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1]); gt = np.ascontiguousarray(gt[:, ::-1])
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[::-1, :]); gt = np.ascontiguousarray(gt[::-1, :])

        img_t = torch.from_numpy(img).unsqueeze(0)        # (1, H, W)
        gt_t = torch.from_numpy(gt).long()                # (H, W)
        return {"image": img_t, "label": gt_t,
                "patient": r["patient"], "group": r["group"], "phase": r["phase"]}


# ------------------------- quick self-test --------------------------
if __name__ == "__main__":
    tr, va = stratified_split(C.TRAIN_DIR, C.VAL_FRACTION, C.SEED)
    print(f"train patients: {len(tr)}  val patients: {len(va)}")

    # confirm stratification
    def groups(dirs):
        from collections import Counter
        return Counter(parse_info_cfg(d)["Group"] for d in dirs)
    print("train groups:", dict(groups(tr)))
    print("val   groups:", dict(groups(va)))

    recs = build_slice_index(tr[:3])
    print(f"slices from first 3 train patients: {len(recs)}")
    ds = ACDCSliceDataset(recs, augment=True)
    sample = ds[0]
    print("sample image:", sample["image"].shape, sample["image"].dtype,
          "| label:", sample["label"].shape, "| uniq:", torch.unique(sample["label"]).tolist())
