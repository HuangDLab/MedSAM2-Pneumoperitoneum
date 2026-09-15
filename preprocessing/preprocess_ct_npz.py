#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess CT volumes and lesion labels into the NPZ format used by MedSAM2 training
and by the validation script.

Each output .npz contains the FULL volume (no z-cropping):
    imgs    : uint8  (D, H, W)  CT after windowing, rescaled to [0, 255]
    gts     : uint8  (D, H, W)  label mask
    spacing : tuple            voxel spacing from the label image header

Small-component removal
-----------------------
Tiny connected components in the label are treated as annotation noise and removed
for TRAINING data only. Validation / test labels are left untouched so that the
evaluation ground truth is exactly what the annotator drew.

    --mode train        ->  3D min size 10 voxels, 2D min size 5 pixels
    --mode val / test   ->  filtering disabled
    (override any time with --min-size-3d / --min-size-2d)

A CSV audit report is always written, listing every connected component and how much
volume the filter removed, so the effect of this step can be inspected.

Examples
--------
    # training set (noise filtering on)
    python preprocess_ct_npz.py --mode train \
        --nii-path  /path/to/images \
        --gt-path   /path/to/labels \
        --split-txt /path/to/splits \
        --out-dir   ./data/npz_train

    # validation set (labels kept as annotated)
    python preprocess_ct_npz.py --mode val \
        --nii-path  /path/to/images \
        --gt-path   /path/to/labels_val \
        --split-txt /path/to/splits \
        --out-dir   ./data/npz_val
"""

import argparse
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage import measure
from tqdm import tqdm

# Default small-component thresholds per split.
# Training labels are denoised; validation/test labels are not.
DEFAULT_THRESHOLDS = {
    "train": (10, 5),   # (3D min voxels, 2D min pixels)
    "val":   (1, 1),    # 1 == disabled
    "test":  (1, 1),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert CT + label NIfTI pairs into MedSAM2 NPZ volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=["train", "val", "test"],
                   help="Which split to process. Also selects the default filtering thresholds.")
    p.add_argument("--nii-path", required=True,
                   help="Folder with the original CT volumes (.nii.gz).")
    p.add_argument("--gt-path", required=True,
                   help="Folder with the label volumes (.nii.gz).")
    p.add_argument("--split-txt", required=True,
                   help="Either the split .txt file itself, or a folder containing <mode>.txt.")
    p.add_argument("--out-dir", required=True,
                   help="Where the .npz files and the audit CSV are written.")

    # CT windowing. Defaults are the lung window used in this work.
    p.add_argument("--window-level", type=int, default=-600, help="CT window level (HU).")
    p.add_argument("--window-width", type=int, default=1500, help="CT window width (HU).")

    # Small-component removal. None -> use DEFAULT_THRESHOLDS[mode].
    p.add_argument("--min-size-3d", type=int, default=None,
                   help="Remove 3D components smaller than this many voxels. 1 disables it.")
    p.add_argument("--min-size-2d", type=int, default=None,
                   help="Remove 2D components smaller than this many pixels per slice. 1 disables it.")

    p.add_argument("--prefix", default="CT_Abd_",
                   help="Filename prefix for the output .npz files.")
    p.add_argument("--id-marker", default="G_",
                   help="Token marking the start of the case ID inside a split-file line.")
    p.add_argument("--img-suffix", default=".nii.gz", help="Image filename suffix.")
    p.add_argument("--gt-suffix", default=".nii.gz", help="Label filename suffix.")

    args = p.parse_args()

    if args.min_size_3d is None or args.min_size_2d is None:
        d3, d2 = DEFAULT_THRESHOLDS[args.mode]
        if args.min_size_3d is None:
            args.min_size_3d = d3
        if args.min_size_2d is None:
            args.min_size_2d = d2
    return args


def resolve_split_file(split_txt, mode):
    """Accept either a .txt file or a folder holding <mode>.txt."""
    if os.path.isdir(split_txt):
        return os.path.join(split_txt, f"{mode}.txt")
    return split_txt


def get_split_ids(split_file, id_marker):
    """
    Read the split file and pull out the core case ID from each line.

    Lines may be bare IDs or full npz names; everything before `id_marker`
    (e.g. the "CT_Abd_" prefix) is stripped:
        CT_Abd_P_2024_0000123.npz  ->  P_2024_0000123
    """
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    target_ids = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            clean_name = line.replace(".npz", "")
            if id_marker in clean_name:
                target_ids.append(clean_name[clean_name.find(id_marker):])
            else:
                target_ids.append(clean_name)
    return target_ids


def apply_ct_window(image_data, window_level, window_width):
    """
    Clip to the CT window, then rescale to [0, 255].

    Note: the rescaling uses the min/max of the clipped volume rather than the
    window bounds themselves. This is kept exactly as used during training so
    that inference preprocessing stays consistent with it.
    """
    lower_bound = window_level - window_width / 2
    upper_bound = window_level + window_width / 2
    image_data = np.clip(image_data, lower_bound, upper_bound)

    min_val = np.min(image_data)
    max_val = np.max(image_data)
    if max_val == min_val:
        return np.zeros_like(image_data, dtype=np.uint8)

    image_data = (image_data - min_val) / (max_val - min_val) * 255.0
    return np.uint8(image_data)


def remove_small_objects_skimage(mask, min_size, connectivity=1):
    """
    Drop connected components smaller than `min_size`.
    connectivity: 3 = 26-neighbour (3D), 2 = 8-neighbour (2D).
    min_size <= 1 is a no-op.
    """
    if min_size <= 1:
        return mask

    labels = measure.label(mask, connectivity=connectivity, background=0)
    if labels.max() == 0:
        return mask

    sizes = np.bincount(labels.ravel())
    keep_mask = sizes >= min_size
    keep_mask[0] = False
    return keep_mask[labels].astype(np.uint8)


def audit_components(name, gt_data, min_size_3d):
    """Record every 3D component and how much volume the 3D filter would remove."""
    labels_audit = measure.label(gt_data, connectivity=3, background=0)
    if labels_audit.max() == 0:
        return {
            "Volume_Name": name, "Initial_Total_Volume": 0, "Removed_Volume": 0,
            "Removed_Percentage": 0, "Num_Components_Total": 0,
            "Num_Components_Removed": 0, "All_Component_Sizes": "[]",
            "Removed_Component_Sizes": "[]",
        }

    component_sizes = np.bincount(labels_audit.ravel())[1:]
    removed_sizes = component_sizes[component_sizes < min_size_3d]
    total_before = np.sum(component_sizes)
    total_removed = np.sum(removed_sizes)
    pct = (total_removed / total_before) * 100 if total_before > 0 else 0

    return {
        "Volume_Name": name,
        "Initial_Total_Volume": total_before,
        "Removed_Volume": total_removed,
        "Removed_Percentage": round(pct, 4),
        "Num_Components_Total": len(component_sizes),
        "Num_Components_Removed": len(removed_sizes),
        "All_Component_Sizes": str(sorted(component_sizes.tolist(), reverse=True)),
        "Removed_Component_Sizes": str(sorted(removed_sizes.tolist(), reverse=True)),
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    filt = "OFF (labels kept as annotated)" if args.min_size_3d <= 1 and args.min_size_2d <= 1 \
        else f"3D>={args.min_size_3d} voxels, 2D>={args.min_size_2d} pixels"
    print("=" * 60)
    print(f"mode              : {args.mode}")
    print(f"CT window         : level {args.window_level}, width {args.window_width}")
    print(f"small-component   : {filt}")
    print(f"output            : {args.out_dir}")
    print("=" * 60)

    split_file = resolve_split_file(args.split_txt, args.mode)
    target_ids = get_split_ids(split_file, args.id_marker)
    print(f"[1] {len(target_ids)} case IDs read from {os.path.basename(split_file)}")

    all_gts = sorted(n for n in os.listdir(args.gt_path) if n.endswith(args.gt_suffix))
    print(f"[2] {len(all_gts)} label files found in gt-path")

    matched_gts = [n for n in all_gts if any(tid in n for tid in target_ids)]
    print(f"[3] {len(matched_gts)} of them match the split list")
    if len(matched_gts) < len(target_ids):
        found = [tid for tid in target_ids if any(tid in n for n in all_gts)]
        missing = sorted(set(target_ids) - set(found))
        print(f"    WARNING: no label for {len(missing)} IDs, e.g. {missing[:5]}")

    names, missing_images = [], []
    for n in matched_gts:
        img_name = n.split(args.gt_suffix)[0] + args.img_suffix
        if os.path.exists(os.path.join(args.nii_path, img_name)):
            names.append(n)
        else:
            missing_images.append(img_name)
    print(f"[4] {len(names)} cases have both image and label")
    if missing_images:
        print(f"    WARNING: no image for {len(missing_images)} cases, e.g. {missing_images[:5]}")

    print(f"\nProcessing {len(names)} volumes...\n")

    audit_data, n_saved, n_skipped = [], 0, 0

    for name in tqdm(names):
        image_name = name.split(args.gt_suffix)[0] + args.img_suffix

        gt_sitk = sitk.ReadImage(os.path.join(args.gt_path, name))
        gt_data_ori = sitk.GetArrayFromImage(gt_sitk).astype(np.uint8)

        img_sitk = sitk.ReadImage(os.path.join(args.nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)

        audit_data.append(audit_components(name, gt_data_ori, args.min_size_3d))

        gt_clean = remove_small_objects_skimage(gt_data_ori, args.min_size_3d, connectivity=3)
        if args.min_size_2d > 1:
            gt_clean = gt_clean.copy()
            for i in range(gt_clean.shape[0]):
                gt_clean[i] = remove_small_objects_skimage(
                    gt_clean[i], args.min_size_2d, connectivity=2
                )

        if np.sum(gt_clean) == 0:
            print(f"  skipping {name}: label is empty after filtering")
            n_skipped += 1
            continue

        img_pre = apply_ct_window(image_data, args.window_level, args.window_width)

        save_name = args.prefix + name.split(args.gt_suffix)[0] + ".npz"
        np.savez_compressed(
            os.path.join(args.out_dir, save_name),
            imgs=img_pre,
            gts=gt_clean,
            spacing=gt_sitk.GetSpacing(),
        )
        n_saved += 1

    audit_csv = os.path.join(args.out_dir, f"preprocessing_audit_{args.mode}.csv")
    if audit_data:
        df = pd.DataFrame(audit_data)
        df.to_csv(audit_csv, index=False)

        total_removed = df["Removed_Volume"].sum()
        total_initial = df["Initial_Total_Volume"].sum()
        pct = (total_removed / total_initial * 100) if total_initial > 0 else 0

        print("\n" + "=" * 60)
        print(f"saved {n_saved} npz, skipped {n_skipped} empty")
        print(f"audit report: {audit_csv}")
        print(f"total label volume before filtering : {total_initial}")
        print(f"total label volume removed          : {total_removed}  ({pct:.4f}%)")
        if args.min_size_3d <= 1 and args.min_size_2d <= 1:
            print("note: filtering was disabled; the 'removed' figures above are")
            print("      what WOULD have been removed, reported for reference only.")
        print("=" * 60)


if __name__ == "__main__":
    main()
