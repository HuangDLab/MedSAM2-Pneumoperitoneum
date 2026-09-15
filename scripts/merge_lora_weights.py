#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge trained LoRA adapters back into the base SAM 2 / MedSAM2 weights.

Training saves checkpoints that still contain separate LoRA adapter tensors, so
they cannot be loaded by the standard (non-PEFT) model used at inference time.
This script rebuilds the LoRA-wrapped model, loads each checkpoint, calls
merge_and_unload() to fold the adapter deltas into the base weights, and saves a
plain checkpoint that inference can consume without any code changes.

The LoRA rank and which modules were adapted are read from the SAME training
config used to produce the checkpoints, so the two can never drift apart.

Example
-------
    python merge_lora_weights.py \
        --config lora_encode_decode_sam2.1_hiera_tiny512_FLARE_RECIST.yaml \
        --input-dir  ./exp_log/stage3_.../checkpoints \
        --output-dir ./exp_log/stage3_.../merged_checkpoints \
        --interval 10
"""

import argparse
import os
import re
import sys

import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from peft import LoraConfig, get_peft_model

# Which submodules receive LoRA. These must match training/trainer.py.
ENCODER_TARGET_MODULES = ["qkv", "proj"]
DECODER_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]
LORA_DROPOUT = 0.05


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge LoRA adapters into base weights for every Nth checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="Training config filename, relative to --config-path.")
    p.add_argument("--config-path", default="sam2/configs",
                   help="Hydra config directory, relative to this script.")
    p.add_argument("--input-dir", required=True,
                   help="Folder containing the raw checkpoint_<epoch>.pt files.")
    p.add_argument("--output-dir", required=True,
                   help="Where merged_checkpoint_<epoch>.pt files are written.")
    p.add_argument("--interval", type=int, default=10,
                   help="Only merge checkpoints whose epoch is a multiple of this.")
    p.add_argument("--epochs", type=int, nargs="+", default=None,
                   help="Merge exactly these epochs instead of using --interval.")
    p.add_argument("--rank", type=int, default=None,
                   help="Override the LoRA rank. Default: read from the config.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-merge checkpoints whose output file already exists.")
    return p.parse_args()


def load_config(config_path, config_name):
    """Load the training config and return (model_cfg, custom_config)."""
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)

    if "model" in cfg:
        model_cfg = cfg.model
    elif "trainer" in cfg and "model" in cfg.trainer:
        model_cfg = cfg.trainer.model
    else:
        raise KeyError("Could not find a 'model' section in the config.")

    custom = cfg.trainer.get("custom_config", {}) if "trainer" in cfg else {}
    return model_cfg, custom


def build_lora_model(model_cfg, rank, enable_encoder_lora, enable_decoder_lora):
    """
    Build a fresh LoRA-wrapped model.

    A new model is built for every checkpoint: merge_and_unload() strips the PEFT
    wrappers, so a merged model cannot be reused for the next load.
    """
    model = instantiate(model_cfg, _recursive_=True)

    if enable_encoder_lora:
        model.image_encoder.trunk = get_peft_model(
            model.image_encoder.trunk,
            LoraConfig(r=rank, lora_alpha=rank * 2,
                       target_modules=ENCODER_TARGET_MODULES,
                       lora_dropout=LORA_DROPOUT, bias="none"),
        )
    if enable_decoder_lora:
        model.sam_mask_decoder = get_peft_model(
            model.sam_mask_decoder,
            LoraConfig(r=rank, lora_alpha=rank * 2,
                       target_modules=DECODER_TARGET_MODULES,
                       lora_dropout=LORA_DROPOUT, bias="none"),
        )
    return model


def load_checkpoint_strictly(model, src_path):
    """
    Load a checkpoint and raise if the keys do not line up.

    A key mismatch would otherwise load almost nothing and still produce a
    "merged" file containing only base weights, so it is a hard error.
    """
    sd = torch.load(src_path, map_location="cpu", weights_only=True)
    if "model" in sd:
        sd = sd["model"]

    missing, unexpected = model.load_state_dict(sd, strict=False)

    lora_loaded = sum(1 for k in sd if "lora_" in k)
    if lora_loaded == 0:
        raise RuntimeError(
            f"No LoRA tensors found in {os.path.basename(src_path)}. "
            "Is this really a LoRA training checkpoint?"
        )
    if missing:
        raise RuntimeError(
            f"{len(missing)} keys missing when loading {os.path.basename(src_path)}, "
            f"e.g. {missing[:5]}. The model structure does not match the checkpoint."
        )
    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} unexpected keys in {os.path.basename(src_path)}, "
            f"e.g. {unexpected[:5]}. The model structure does not match the checkpoint."
        )
    return lora_loaded


def find_checkpoints(input_dir, interval, epochs):
    """Return [(epoch, filename), ...] sorted by epoch."""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pattern = re.compile(r"^checkpoint_(\d+)\.pt$")
    tasks = []
    for fname in os.listdir(input_dir):
        m = pattern.match(fname)
        if not m:
            continue
        epoch = int(m.group(1))
        if epochs is not None:
            if epoch in epochs:
                tasks.append((epoch, fname))
        elif epoch % interval == 0:
            tasks.append((epoch, fname))
    tasks.sort(key=lambda t: t[0])
    return tasks


def main():
    args = parse_args()

    model_cfg, custom = load_config(args.config_path, args.config)
    rank = args.rank if args.rank is not None else int(custom.get("lora_rank", 32))
    enc = bool(custom.get("enable_encoder_lora", True))
    dec = bool(custom.get("enable_decoder_lora", True))

    print("=" * 64)
    print(f"config        : {args.config}")
    print(f"LoRA rank     : {rank}" + ("" if args.rank is None else "  (overridden)"))
    print(f"encoder LoRA  : {enc}")
    print(f"decoder LoRA  : {dec}")
    print(f"input         : {args.input_dir}")
    print(f"output        : {args.output_dir}")
    print("=" * 64)

    if not enc and not dec:
        print("Neither encoder nor decoder LoRA is enabled; there is nothing to merge.")
        sys.exit(1)

    tasks = find_checkpoints(args.input_dir, args.interval, args.epochs)
    if not tasks:
        sel = f"epochs {args.epochs}" if args.epochs else f"every {args.interval} epochs"
        print(f"No checkpoints matching {sel} found in {args.input_dir}")
        sys.exit(1)
    print(f"\n{len(tasks)} checkpoints to merge: {[e for e, _ in tasks]}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    n_ok, failures = 0, []
    for epoch, fname in tasks:
        src = os.path.join(args.input_dir, fname)
        dst = os.path.join(args.output_dir, f"merged_checkpoint_{epoch}.pt")

        if os.path.exists(dst) and not args.overwrite:
            print(f"epoch {epoch}: already merged, skipping (use --overwrite to redo)")
            n_ok += 1
            continue

        print(f"epoch {epoch}: merging {fname} ...")
        try:
            model = build_lora_model(model_cfg, rank, enc, dec)
            n_lora = load_checkpoint_strictly(model, src)

            if enc:
                model.image_encoder.trunk = model.image_encoder.trunk.merge_and_unload()
            if dec:
                model.sam_mask_decoder = model.sam_mask_decoder.merge_and_unload()

            merged_sd = model.state_dict()
            leftover = [k for k in merged_sd if "lora_" in k]
            if leftover:
                raise RuntimeError(
                    f"{len(leftover)} LoRA tensors survived the merge, e.g. {leftover[:3]}"
                )

            torch.save({"model": merged_sd}, dst)
            print(f"           merged {n_lora} LoRA tensors -> {dst}")
            n_ok += 1
        except Exception as e:
            print(f"           FAILED: {type(e).__name__}: {e}")
            failures.append((epoch, str(e)))

    print("\n" + "=" * 64)
    print(f"merged {n_ok}/{len(tasks)} checkpoints")
    if failures:
        print(f"{len(failures)} failed:")
        for epoch, msg in failures:
            print(f"  epoch {epoch}: {msg}")
        sys.exit(1)
    print("=" * 64)


if __name__ == "__main__":
    main()
