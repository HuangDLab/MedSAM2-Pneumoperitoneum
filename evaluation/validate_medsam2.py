#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate merged MedSAM2 checkpoints on preprocessed NPZ volumes.

Protocol ("largest-component prompt")
-------------------------------------
For each volume the ground-truth mask is split into 3D connected components
(26-neighbour) and the LARGEST one, by voxel count, is selected. A single
positive point prompt is placed inside that pocket, and the prediction is
propagated forwards and backwards through the volume from the prompted slice.

`--prompt deepest3d` (the default) places the prompt at the voxel furthest from
the surface of the pocket, accounting for voxel spacing, which guarantees it lies
on annotated air. `--prompt centroid` uses the centre of mass instead; for a
concave pocket that can fall outside the air. The two modes are not comparable.

Two properties of this protocol must be stated whenever results from it are
reported:

  1. The prompt is derived from the ground truth. This is an oracle prompt, the
     same convention used by the MedSAM2 reference evaluation, and it means the
     pipeline is interactive rather than fully automatic.
  2. Only the largest component is prompted, but Dice is computed against the
     COMPLETE ground-truth mask. Any additional component that propagation does
     not reach counts as a false negative.

Reported metrics
----------------
  Mean Dice          : unweighted mean of per-volume Dice
  Weighted Mean Dice : per-volume Dice weighted by ground-truth volume
  Global Dice        : 2 * sum(intersection) / sum(volumes), pooled over the set

The "loss proxy" columns are convenience diagnostics computed from the masks,
not the training loss. They use the same weights as the training config
(mask 5, dice 2, iou 2) but the mask term is a mean squared error on the binary
masks, not the focal loss used during training. Do not report them as losses.

Example
-------
    python validate_medsam2.py \
        --imgs-path  ./data/npz_val \
        --ckpt-dir   ./exp_log/stage3_.../merged_checkpoints \
        --output-dir ./results/stage3_val \
        --ckpt-interval 10
"""

import argparse
import os
import re
import sys
import time
from collections import OrderedDict
from glob import glob
from os.path import basename, join
from typing import Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

from sam2.build_sam import build_sam2_video_predictor_npz

torch.set_float32_matmul_precision("high")

# Loss-proxy weights, kept in sync with trainer.loss.all.weight_dict in the
# training config. These are diagnostics only; see the module docstring.
W_MASK, W_DICE, W_IOU = 5.0, 2.0, 2.0


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def compute_dice_score(mask_pred: np.ndarray,
                       mask_gt: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (dice, intersection, volume_sum, gt_volume)."""
    mask_pred = mask_pred.astype(bool)
    mask_gt = mask_gt.astype(bool)

    intersection = float(np.logical_and(mask_pred, mask_gt).sum())
    volume_sum = float(mask_pred.sum() + mask_gt.sum())
    gt_volume = float(mask_gt.sum())

    if volume_sum == 0:
        # Nothing predicted and nothing annotated: treated as a perfect match.
        return 1.0, 0.0, 0.0, 0.0

    return (2.0 * intersection) / volume_sum, intersection, volume_sum, gt_volume


def compute_iou_score(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    mask_pred = mask_pred.astype(bool)
    mask_gt = mask_gt.astype(bool)

    union = float(np.logical_or(mask_pred, mask_gt).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(mask_pred, mask_gt).sum()) / union


def compute_mask_loss_surrogate(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """MSE between the binary masks. A diagnostic, not the training mask loss."""
    return float(np.mean((mask_pred.astype(np.float32) - mask_gt.astype(np.float32)) ** 2))


def compute_all_metrics_and_losses(segs_3D, gts_3D_ori, unique_labs):
    dice_scores, iou_scores, mask_losses = [], [], []
    total_inter = total_vol_sum = total_gt_vol = 0.0

    for ulab in unique_labs:
        gt_mask = (gts_3D_ori == ulab).astype(np.uint8)
        pred_mask = (segs_3D == ulab).astype(np.uint8)

        dice, inter, vol_sum, gt_vol = compute_dice_score(pred_mask, gt_mask)
        dice_scores.append(dice)
        iou_scores.append(compute_iou_score(pred_mask, gt_mask))
        mask_losses.append(compute_mask_loss_surrogate(pred_mask, gt_mask))

        total_inter += inter
        total_vol_sum += vol_sum
        total_gt_vol += gt_vol

    mean_dice = float(np.mean(dice_scores))
    mean_iou = float(np.mean(iou_scores))
    mean_mask_loss = float(np.mean(mask_losses))
    mean_dice_loss = 1.0 - mean_dice
    mean_iou_loss = 1.0 - mean_iou

    core_proxy = (W_MASK * mean_mask_loss) + (W_DICE * mean_dice_loss) + (W_IOU * mean_iou_loss)
    class_proxy = mean_mask_loss * 0.1
    all_proxy = core_proxy + class_proxy

    return (all_proxy, mean_mask_loss, mean_dice_loss, mean_iou_loss, class_proxy,
            mean_dice, mean_iou, total_inter, total_vol_sum, total_gt_vol)


# -----------------------------------------------------------------------------
# Prompt point selection
# -----------------------------------------------------------------------------
def _sampling_from_spacing(spacing):
    """
    Convert a SimpleITK spacing to the per-axis sampling expected by
    distance_transform_edt.

    SimpleITK reports spacing as (x, y, z) while the array axes are (z, y, x),
    so the order has to be reversed. Without this the transform treats one slice
    step as equal to one in-plane pixel, which biases the deepest point towards
    the through-plane direction on anisotropic CT.
    """
    try:
        sx, sy, sz = (float(v) for v in np.asarray(spacing).ravel()[:3])
        if min(sx, sy, sz) <= 0:
            return None
        return (sz, sy, sx)
    except Exception:
        return None


def select_prompt_point(component, spacing=None, mode="deepest3d", centroid=None):
    """
    Choose the (z, y, x) voxel used as the positive point prompt.

    component : bool array (D, H, W), True inside the chosen air pocket
    mode:
      deepest3d -- voxel furthest from the 3D surface of the pocket (default).
                   Physical spacing is taken into account so that the choice is
                   not skewed by anisotropic slice thickness.
      deepest2d -- deepest voxel within the centroid slice only.
      centroid  -- centre of mass. NOT guaranteed to lie inside a concave pocket.

    Returns (z, y, x, depth), where depth is the distance to the pocket boundary
    in millimetres for the deepest* modes and 0.0 for centroid.
    """
    if mode == "centroid":
        # Truncation, not rounding: a one-voxel shift in the prompt can change
        # the segmentation substantially.
        z, y, x = (int(c) for c in centroid)
        return z, y, x, 0.0

    sampling = _sampling_from_spacing(spacing)

    if mode == "deepest3d":
        # Pad so that a pocket touching the volume border is not credited with
        # extra depth: distance_transform_edt measures distance to the nearest
        # zero, and the array edge is not a zero.
        padded = np.pad(component, 1, mode="constant", constant_values=False)
        dt = ndi.distance_transform_edt(padded, sampling=sampling)[1:-1, 1:-1, 1:-1]
        z, y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
        return int(z), int(y), int(x), float(dt[z, y, x])

    if mode == "deepest2d":
        z = int(centroid[0])   # same slice as centroid mode
        sl = np.pad(component[z], 1, mode="constant", constant_values=False)
        s2 = sampling[1:] if sampling else None
        dt = ndi.distance_transform_edt(sl, sampling=s2)[1:-1, 1:-1]
        y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
        return z, int(y), int(x), float(dt[y, x])

    raise ValueError(f"unknown prompt mode: {mode}")


# -----------------------------------------------------------------------------
# Image helper
# -----------------------------------------------------------------------------
def resize_grayscale_to_rgb_and_resize(array: np.ndarray, image_size: int) -> np.ndarray:
    """(D, H, W) grayscale -> (D, 3, image_size, image_size)."""
    from PIL import Image

    d = array.shape[0]
    resized = np.zeros((d, 3, image_size, image_size), dtype=np.uint8)
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_resized = img_pil.resize((image_size, image_size))
        img_array = np.array(img_resized)
        resized[i] = np.stack([img_array] * 3, axis=0)
    return resized


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
@torch.inference_mode()
def infer_3d(img_npz_file, predictor, save_nifti, nifti_path, prompt_mode="deepest3d"):
    start_time = time.time()
    npz_name = basename(img_npz_file)

    npz_data = np.load(img_npz_file, "r", allow_pickle=True)
    if "gts" not in npz_data.files:
        print(f"  warning: no 'gts' in {npz_name}, skipping")
        return None

    spacing = npz_data["spacing"]
    img_3D_ori = npz_data["imgs"]
    gts_3D_ori = npz_data["gts"]

    if np.sum(gts_3D_ori) == 0:
        return None

    video_height, video_width = img_3D_ori.shape[1:3]
    if video_height != 512 or video_width != 512:
        img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
    else:
        img_resized = img_3D_ori[:, None].repeat(3, axis=1)

    img_norm = torch.from_numpy(img_resized / 255.0).cuda()
    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None].cuda()
    img_norm -= img_mean
    img_norm /= img_std

    segs_3D = np.zeros(gts_3D_ori.shape, dtype=np.uint8)

    # Largest-component prompt: see the module docstring for what this implies.
    lab = label(gts_3D_ori)          # connectivity defaults to ndim -> 26-neighbour
    regions = regionprops(lab)
    if not regions:
        print(f"  warning: {npz_name} has GT voxels but no connected components")
        return None

    largest_region = max(regions, key=lambda r: r.area)
    component = (lab == largest_region.label)

    z_mid, y_mid, x_mid, depth = select_prompt_point(
        component, spacing=spacing, mode=prompt_mode, centroid=largest_region.centroid
    )
    prompt_inside = bool(gts_3D_ori[z_mid, y_mid, x_mid] > 0)
    if prompt_mode != "centroid" and not prompt_inside:
        raise RuntimeError(
            f"{npz_name}: prompt point ({z_mid},{y_mid},{x_mid}) is outside the label"
        )

    points = np.array([[x_mid, y_mid]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)  # 1 = positive point

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Forward pass from the prompted slice.
        state = predictor.init_state(img_norm, video_height, video_width)
        _, _, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=state, frame_idx=z_mid, obj_id=1,
            points=points, labels=labels,
        )
        segs_3D[z_mid] = np.logical_or(
            segs_3D[z_mid],
            (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8),
        )
        for idx, _, logits in predictor.propagate_in_video(
            state, start_frame_idx=z_mid, reverse=False
        ):
            segs_3D[idx] = np.logical_or(segs_3D[idx], (logits[0] > 0.0).cpu().numpy()[0])
        predictor.reset_state(state)

        # Backward pass: the state is rebuilt because propagate_in_video cannot
        # be run twice in opposite directions on one state.
        state = predictor.init_state(img_norm, video_height, video_width)
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=z_mid, obj_id=1,
            points=points, labels=labels,
        )
        for idx, _, logits in predictor.propagate_in_video(
            state, start_frame_idx=z_mid, reverse=True
        ):
            segs_3D[idx] = np.logical_or(segs_3D[idx], (logits[0] > 0.0).cpu().numpy()[0])
        predictor.reset_state(state)

    unique_labs = np.unique(gts_3D_ori)
    unique_labs = unique_labs[unique_labs != 0]
    if len(unique_labs) == 0:
        unique_labs = [1]

    metrics = compute_all_metrics_and_losses(segs_3D, gts_3D_ori, unique_labs)
    duration = time.time() - start_time
    flag = "" if prompt_inside else "  [PROMPT OUTSIDE LABEL]"
    print(f"  {npz_name}: {duration:.2f}s  Dice {metrics[5]:.4f}  "
          f"IoU {metrics[6]:.4f}{flag}")

    if save_nifti:
        sitk_img = sitk.GetImageFromArray(img_3D_ori)
        sitk_img.SetSpacing(spacing)
        sitk.WriteImage(sitk_img, join(nifti_path, npz_name.replace(".npz", "_imgs.nii.gz")))
        sitk_seg = sitk.GetImageFromArray(segs_3D)
        sitk_seg.SetSpacing(spacing)
        sitk.WriteImage(sitk_seg, join(nifti_path, npz_name.replace(".npz", "_segs.nii.gz")))

    prompt_info = (z_mid, y_mid, x_mid, round(depth, 3), prompt_inside,
                   int(len(regions)), int(largest_region.area))
    return (npz_name, duration) + metrics + prompt_info


# -----------------------------------------------------------------------------
# Checkpoint selection
# -----------------------------------------------------------------------------
def select_checkpoints(ckpt_dir, interval=None, names=None, epochs=None):
    """
    Choose which checkpoints to validate.

    Exactly one selection mode is used, in this order of precedence:
      names    -- validate these filenames only
      epochs   -- validate these epoch numbers only
      interval -- validate every checkpoint whose epoch is a multiple of interval

    Files that carry no epoch number (e.g. checkpoint.pt) are always included.
    """
    all_files = sorted(glob(join(ckpt_dir, "*.pt")))
    if not all_files:
        return []

    if names:
        wanted = set(names)
        return [f for f in all_files if basename(f) in wanted]

    pattern = re.compile(r"checkpoint_(\d+)\.pt$")
    selected = []
    for path in all_files:
        m = pattern.search(basename(path))
        if not m:
            selected.append(path)          # no epoch in the name: always keep
            continue
        epoch = int(m.group(1))
        if epochs is not None:
            if epoch in epochs:
                selected.append(path)
        elif interval is None or epoch % interval == 0:
            selected.append(path)
    return selected


# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Validate merged MedSAM2 checkpoints on NPZ volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--imgs-path", required=True,
                   help="Folder of preprocessed validation .npz volumes.")
    p.add_argument("--ckpt-dir", required=True,
                   help="Folder of merged checkpoints (.pt) to evaluate.")
    p.add_argument("-o", "--output-dir", required=True,
                   help="Where the summary CSV and optional NIfTI outputs go.")
    p.add_argument("--cfg", default="sam2/configs",
                   help="Directory holding the model config.")
    p.add_argument("--cfg-name", default="sam2.1_hiera_t512.yaml",
                   help="Model config filename inside --cfg.")

    sel = p.add_argument_group("checkpoint selection (use at most one)")
    sel.add_argument("--ckpt-interval", type=int, default=10,
                     help="Evaluate checkpoints whose epoch is a multiple of this.")
    sel.add_argument("--ckpt-epochs", type=int, nargs="+", default=None,
                     help="Evaluate exactly these epoch numbers.")
    sel.add_argument("--ckpt-names", type=str, nargs="+", default=None,
                     help="Evaluate exactly these checkpoint filenames.")

    p.add_argument("--prompt", choices=["deepest3d", "deepest2d", "centroid"],
                   default="deepest3d",
                   help="How the positive point prompt is placed inside the largest "
                        "air pocket. 'centroid' reproduces the original behaviour and "
                        "is not guaranteed to fall inside a concave pocket.")
    p.add_argument("--save-nifti", action="store_true",
                   help="Write the predicted masks as NIfTI volumes.")
    p.add_argument("--save-per-case", action="store_true",
                   help="Also write one row per volume, not just the summary.")
    p.add_argument("--summary-name", default="validation_summary.csv",
                   help="Filename of the summary CSV inside --output-dir.")
    p.add_argument("--seed", type=int, default=2024)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    nifti_path = join(args.output_dir, "segs_nifti")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_nifti:
        os.makedirs(nifti_path, exist_ok=True)

    checkpoint_files = select_checkpoints(
        args.ckpt_dir, args.ckpt_interval, args.ckpt_names, args.ckpt_epochs
    )
    if not checkpoint_files:
        print(f"Error: no matching checkpoints in {args.ckpt_dir}")
        sys.exit(1)

    img_npz_files = sorted(glob(join(args.imgs_path, "*.npz")))
    if not img_npz_files:
        print(f"Error: no .npz files in {args.imgs_path}")
        sys.exit(1)

    print("=" * 64)
    print(f"validation data : {len(img_npz_files)} volumes from {args.imgs_path}")
    print(f"checkpoints     : {len(checkpoint_files)} from {args.ckpt_dir}")
    print(f"prompt placement: {args.prompt}")
    for f in checkpoint_files:
        print(f"                  {basename(f)}")
    print("=" * 64)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Hydra treats a leading '//' as an absolute filesystem path.
    full_model_cfg = "//" + join(script_dir, args.cfg, args.cfg_name)

    summary_rows, per_case_frames = [], []

    for ckpt_path in checkpoint_files:
        ckpt_name = basename(ckpt_path)
        print(f"\n--- {ckpt_name} ---")

        try:
            predictor = build_sam2_video_predictor_npz(full_model_cfg, ckpt_path)
        except Exception as e:
            print(f"  could not load checkpoint: {e}. Skipping.")
            continue

        global_inter = global_vol_sum = 0.0
        rows = OrderedDict((k, []) for k in [
            "checkpoint_name", "image", "duration",
            "Losses/all_loss_proxy", "Losses/mask_loss_proxy",
            "Losses/dice_loss_proxy", "Losses/iou_loss_proxy",
            "Losses/class_loss_proxy",
            "Metrics/Dice_Score", "Metrics/IoU_Score",
            "Calculated_GT_Volume", "Vol_inter", "Vol_predict",
            "prompt_z", "prompt_y", "prompt_x", "prompt_depth_mm",
            "prompt_inside_label", "n_air_pockets", "largest_pocket_voxels",
        ])

        n_skipped = 0
        for img_file in img_npz_files:
            try:
                result = infer_3d(img_file, predictor, args.save_nifti, nifti_path,
                                  prompt_mode=args.prompt)
            except Exception as e:
                print(f"  error on {basename(img_file)}: {type(e).__name__}: {e}")
                n_skipped += 1
                continue
            if result is None:
                n_skipped += 1
                continue

            (npz_name, duration, all_loss, mask_loss, dice_loss, iou_loss,
             class_loss, dice, iou, vol_inter, vol_sum, gt_vol,
             pz, py, px, pdepth, pinside, n_pockets, largest_vox) = result

            global_inter += vol_inter
            global_vol_sum += vol_sum

            for key, val in zip(rows, [ckpt_name, npz_name, duration, all_loss,
                                       mask_loss, dice_loss, iou_loss, class_loss,
                                       dice, iou, gt_vol, vol_inter, vol_sum,
                                       pz, py, px, pdepth, pinside,
                                       n_pockets, largest_vox]):
                rows[key].append(val)

        if not rows["image"]:
            print(f"  no volumes processed for {ckpt_name}")
            continue

        df = pd.DataFrame(rows)
        per_case_frames.append(df)

        gt_total = df["Calculated_GT_Volume"].sum()
        weighted_dice = (
            (df["Metrics/Dice_Score"] * df["Calculated_GT_Volume"]).sum() / gt_total
            if gt_total > 0 else 0.0
        )
        global_dice = (2.0 * global_inter / global_vol_sum) if global_vol_sum > 0 else 1.0

        n_outside = int((~df["prompt_inside_label"].astype(bool)).sum())
        pocket_frac = (df["largest_pocket_voxels"].sum()
                       / df["Calculated_GT_Volume"].sum()) if gt_total > 0 else float("nan")

        summary_rows.append({
            "Checkpoint": ckpt_name,
            "Prompt_Mode": args.prompt,
            "Prompt_Outside_Label": n_outside,
            "Largest_Pocket_Fraction": round(float(pocket_frac), 4),
            "Global_Dice_Score": global_dice,
            "Weighted_Mean_Dice_Score": weighted_dice,
            "Mean_Dice_Score": df["Metrics/Dice_Score"].mean(),
            "Mean_IoU_Score": df["Metrics/IoU_Score"].mean(),
            "Mean_Loss_Proxy": df["Losses/all_loss_proxy"].mean(),
            "Total_Files": len(df),
            "Skipped_Files": n_skipped,
        })
        print(f"  Mean Dice {df['Metrics/Dice_Score'].mean():.4f}  "
              f"Weighted {weighted_dice:.4f}  Global {global_dice:.4f}  "
              f"({len(df)} volumes, {n_skipped} skipped)")
        if n_outside:
            print(f"  WARNING: the prompt fell outside the label in {n_outside} volumes")

    if not summary_rows:
        print("\nNo checkpoint produced any result.")
        sys.exit(1)

    summary_path = join(args.output_dir, args.summary_name)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print("\n" + "=" * 64)
    print(f"summary written to {summary_path}")

    if args.save_per_case and per_case_frames:
        per_case_path = join(args.output_dir, "validation_per_case.csv")
        pd.concat(per_case_frames, ignore_index=True).to_csv(per_case_path, index=False)
        print(f"per-case results written to {per_case_path}")

    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("=" * 64)


if __name__ == "__main__":
    main()
