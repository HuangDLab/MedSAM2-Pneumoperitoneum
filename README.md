# MedSAM2-Pneumoperitoneum

LoRA fine-tuning of [MedSAM2](https://github.com/bowang-lab/MedSAM2) for free-air
segmentation on abdominal CT.

This repository contains only the files we modified or added. Everything else in
MedSAM2 is used unmodified; `setup.sh` fetches it.

| | |
|---|---|
| Modified | `training/trainer.py` (LoRA injection, ~79 lines), `training/train.py` (1 line) |
| Added | the training config, `preprocessing/`, `scripts/`, `evaluation/` |

---

## Setup

```bash
git clone https://github.com/HuangDLab/MedSAM2-Pneumoperitoneum.git
cd MedSAM2-Pneumoperitoneum
bash setup.sh ./MedSAM2      # clones upstream MedSAM2 and applies our files
cd MedSAM2
```

Install MedSAM2 as described in its own README, then add the few packages it does
not already pull in:

```bash
pip install -r /path/to/MedSAM2-Pneumoperitoneum/requirements.txt
```

Tested with Python 3.11.2, PyTorch 2.10.0, CUDA 12.8, peft 0.18.1, SimpleITK 2.5.3,
scikit-image 0.26.0. Training uses bfloat16 AMP and needs an Ampere or newer GPU.

---

## Data preparation

Each CT volume and its label become one `.npz` holding the full volume:
`imgs` (uint8, D×H×W), `gts` (uint8, D×H×W), `spacing`.

```bash
# training set
python preprocessing/preprocess_ct_npz.py --mode train \
    --nii-path /path/to/images --gt-path /path/to/labels \
    --split-txt /path/to/splits --out-dir ./data/npz_train

# validation / test
python preprocessing/preprocess_ct_npz.py --mode val \
    --nii-path /path/to/images --gt-path /path/to/labels_val \
    --split-txt /path/to/splits --out-dir ./data/npz_val
```

Lung window (level −600 HU, width 1500 HU). `--mode train` removes connected
components below 10 voxels in 3D and 5 pixels per slice; `--mode val` and
`--mode test` leave labels exactly as annotated.

---

## Training

```bash
bash scripts/train_lora.sh
```

The reported model came from three successive fine-tuning stages. Stage 1 starts
from the public MedSAM2 weights and each later stage starts from the previous
stage's merged checkpoint; the configuration is otherwise identical across the
three, so only the stage-3 config is included here. To run an earlier stage,
change `checkpoint_path` and the dataset `folder` in that config.

| | |
|---|---|
| Backbone | SAM 2.1 Hiera-tiny, 512 × 512 |
| LoRA | encoder + decoder, rank 32, alpha 64, dropout 0.05 |
| Loss | upstream `MultiStepMultiMasksAndIous` — mask 5, dice 2, IoU 2, class 1 |
| Optimizer | AdamW, weight decay 0.1, gradient clip 0.1 |
| LR | cosine 2.0e-4 → 2.0e-5, layer-wise decay 0.9 on the encoder trunk |
| Schedule | 600 epochs, batch 8 × 8 frames, single GPU, bfloat16 AMP |

Every run writes the settings it actually used to
`<output_path>/config_resolved.yaml`.

---

## Merging LoRA weights

Training checkpoints keep the LoRA adapters separate and cannot be loaded by the
standard inference model. Fold them in first:

```bash
python scripts/merge_lora_weights.py \
    --config lora_encode_decode_sam2.1_hiera_tiny512_FLARE_RECIST.yaml \
    --input-dir  ./exp_log/stage3_.../checkpoints \
    --output-dir ./exp_log/stage3_.../merged_checkpoints \
    --interval 10
```

---

## Evaluation

```bash
python evaluation/validate_medsam2.py \
    --imgs-path  ./data/npz_val \
    --ckpt-dir   ./exp_log/stage3_.../merged_checkpoints \
    --output-dir ./results/stage3_val \
    --ckpt-names merged_checkpoint_xxx.pt
```

Reports Mean Dice, volume-weighted Mean Dice and Global Dice.

The prompt is a single positive point inside the largest 3D connected component of
the ground truth, and the prediction is propagated both ways from that slice.
`--prompt deepest3d` (default) places it at the voxel furthest from the surface of
that component; `--prompt centroid` uses the centre of mass instead. The two are not
comparable — see the paper for which was used.

---

## Data availability

The CT data is retrospective clinical imaging and cannot be released. Model
checkpoints from the intermediate fine-tuning stages are not public either, so the
reported model cannot be reproduced bit-for-bit; the code and the exact training
configuration are provided so the method can be applied to other data.

---

## License

Apache-2.0, matching upstream MedSAM2. Files derived from MedSAM2 / SAM 2 keep their
original copyright headers and carry a notice at the top of the file describing what
was changed. No model weights are redistributed here.

If you use this code, please cite MedSAM2 and SAM 2 alongside our paper.
