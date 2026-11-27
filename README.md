# Infrared UAV Detection and Tracking

A compact research implementation for detecting and tracking a single small UAV
in thermal infrared video. The project combines a YOLOv12-S detector with
BoT-SORT or TrackTrack and explores two detector extensions: infrared-specific
augmentation and a parallel anomaly-aware objectness head.

This repository was developed as a course/research project from the public
[Dist-Tracker](https://github.com/earth-insights/Dist-Tracker) codebase. It is
not an official implementation of Dist-Tracker, AA-YOLO, BoT-SORT, or
TrackTrack.

## Motivation

Infrared UAV tracking is difficult because the target is often only a few
pixels wide, has weak appearance cues, may share its temperature with the
background, and frequently disappears behind clutter. A detector can recover
from long target absence, while a tracker can suppress isolated false positives
and maintain temporal continuity. This project studies both parts with a small,
reproducible set of variants rather than a large collection of heuristics.

## Method

### Detector variants

| Detector | Description |
|---|---|
| **Baseline** | One-class YOLOv12-S with P3/P4/P5 detection heads, CIoU box loss, BCE classification loss, and DFL. |
| **IR** | Baseline trained with sinusoidal IR augmentation at probability `0.1`. The transform preserves the image shape and supports both bright and dark targets. |
| **AA** | YOLOv12 detector trained with the same IR augmentation and an additional anomaly-aware objectness branch at P3/P4/P5. The original classification and box branches are retained. |

For the AA detector, Task-Aligned Assignment uses the class branch so that a
randomly initialized AA head cannot collapse assignment at the beginning of
training. At inference, the final confidence is

```text
p = sigmoid(class_logit) * p_AA
```

The AA statistical test is evaluated in FP32 under mixed precision. There is no
duplicate sigmoid. Training uses `box:cls:AA:DFL = 7.5:0.5:0.5:1.5`.

### Tracking variants

| Public variant | Detector | Tracker | Detection floor |
|---|---|---|---:|
| `baseline_botsort` | Baseline | BoT-SORT | `0.01` |
| `ir_botsort` | IR | BoT-SORT | `0.01` |
| `ir_tracktrack` | IR | TrackTrack | `0.01` |
| `aa_botsort` | AA | BoT-SORT | `0.001` |
| `aa_tracktrack` | AA | TrackTrack | `0.001` |

Both trackers run without ReID and use sparse-optical-flow GMC. TrackTrack is a
minimal port with NSA-Kalman filtering, HMIoU/confidence/corner-angle
association, iterative matching, Track-Aware Initialization, and loose-NMS
recovery. The final output policy emits at most one UAV box per frame and
applies `min_box_area=10` only at output selection; it does not remove internal
tracks or alter association.

## Results on Anti-UAV410

The detector was trained on YOLO-formatted Anti-UAV410 frames sampled with
temporal stride 5. Detection and tracking results below use the full-frame-rate
120-video Anti-UAV410 test subset. These are project results, not numbers
reported by the original papers.

### Detection

| Detector | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline | **0.9554** | 0.8094 | 0.8690 | 0.4736 |
| IR | 0.9523 | 0.8067 | 0.8779 | 0.4772 |
| AA | 0.9527 | **0.8131** | **0.8840** | **0.4894** |

### Tracking

AOA is the official Anti-UAV CVPR 2025 Track-2 metric, computed per video and
then macro-averaged.

| Variant | AOA | SA | Mean miss rate |
|---|---:|---:|---:|
| `baseline_botsort` | 0.6280 | 0.6639 | 0.0534 |
| `ir_botsort` | 0.6355 | 0.6708 | 0.0563 |
| `ir_tracktrack` | **0.6517** | **0.6756** | 0.0294 |
| `aa_botsort` | 0.6303 | 0.6651 | 0.0492 |
| `aa_tracktrack` | 0.6454 | 0.6690 | **0.0251** |

## Checkpoints

The five tracking variants use three detector checkpoints:

| Checkpoint | Used by | Download |
|---|---|---|
| `baseline.pt` | `baseline_botsort` | [Download](https://github.com/thanhh-tran/infrared-uav-tracking/releases/latest/download/baseline.pt) |
| `ir.pt` | `ir_botsort`, `ir_tracktrack` | [Download](https://github.com/thanhh-tran/infrared-uav-tracking/releases/latest/download/ir.pt) |
| `aa.pt` | `aa_botsort`, `aa_tracktrack` | [Download](https://github.com/thanhh-tran/infrared-uav-tracking/releases/latest/download/aa.pt) |

Download all three into `pretrained/`:

```bash
mkdir -p pretrained

base=https://github.com/thanhh-tran/infrared-uav-tracking/releases/latest/download
curl -L "$base/baseline.pt" -o pretrained/baseline.pt
curl -L "$base/ir.pt"       -o pretrained/ir.pt
curl -L "$base/aa.pt"       -o pretrained/aa.pt
```

## Data

Training and reported evaluation use
[Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410). The unlabeled CVPR
2025 challenge test sets are available from the
[official 4th Anti-UAV release](https://zenodo.org/records/15103888) and are
used only for inference/visualization.

The detector expects standard one-class YOLO labels:

```text
class_id x_center y_center width height
```

All coordinates are normalized to `[0, 1]`, and the UAV class is `0`. A minimal
dataset YAML is:

```yaml
path: /absolute/path/to/anti_uav410_yolo
train: images/train
val: images/val
test: images/test
names:
  0: UAV
```

Keep train/validation/test separation at video level. Do not randomly split
individual frames, as that leaks nearly identical neighboring frames across
splits.

## Installation

Python 3.10–3.12 is recommended. Install a PyTorch build compatible with the
local CUDA driver first, then install the remaining dependencies:

```bash
git clone https://github.com/thanhh-tran/infrared-uav-tracking.git
cd infrared-uav-tracking

python -m venv .venv
source .venv/bin/activate

# Select the correct CUDA command at https://pytorch.org/get-started/locally/
pip install torch torchvision
pip install -r requirements-disttracker.txt
```

The vendored Ultralytics source in this repository is used directly; installing
another `ultralytics` package is not required.

## Inference

The portable wrapper accepts one video, one directory of frames, or a root
whose immediate subdirectories are videos:

```bash
# Baseline + BoT-SORT
bash scripts/infer_variant.sh \
  baseline_botsort pretrained/baseline.pt /path/to/video.mp4

# IR + TrackTrack
bash scripts/infer_variant.sh \
  ir_tracktrack pretrained/ir.pt /path/to/frame_directory

# AA + TrackTrack
bash scripts/infer_variant.sh \
  aa_tracktrack pretrained/aa.pt /path/to/test_root
```

Outputs are written under `runs/inference/<variant>/` and include a rendered
MP4 plus one compact `predictions.jsonl` file per video. Runtime settings are
overridable without editing the script:

```bash
DEVICE=0 IMGSZ=1280 FPS=25 \
  bash scripts/infer_variant.sh ir_botsort pretrained/ir.pt /path/to/frames
```

The Python entrypoint can also be called directly:

```bash
python infer.py \
  --weights pretrained/ir.pt \
  --source /path/to/frames \
  --tracker ultralytics/cfg/trackers/botsort_uav.yaml \
  --imgsz 1280 --conf 0.01 --device 0 \
  --save-video --compact-labels --fps 25
```

## Training

### Baseline and IR detector

```bash
# Continue/fine-tune a standard YOLOv12-UAV checkpoint without IR augmentation
DEVICE=0 BATCH=16 EPOCHS=10 \
  bash scripts/train_detector.sh \
  baseline /path/to/data.yaml /path/to/initial_detector.pt baseline

# IR augmentation probability defaults to 0.1 in IR mode
DEVICE=0 BATCH=16 EPOCHS=10 IR_SINE_P=0.1 \
  bash scripts/train_detector.sh \
  ir /path/to/data.yaml /path/to/initial_detector.pt ir
```

### AA detector

AA training has two phases. Phase 1 uses base LR `1e-6` and AA-head LR `1e-3`.
Phase 2 loads the phase-1 best weights into a fresh optimizer and fine-tunes all
parameters at LR `1e-4`.

```bash
DEVICE=0 BATCH=16 \
  bash scripts/train_aa.sh \
  /path/to/data.yaml pretrained/ir.pt
```

Use `WANDB_OFFLINE=1` to enable offline Weights & Biases logging. The scripts
disable multi-scale training and use input size 1280 by default.

## Validation

```bash
SPLIT=test DEVICE=0 BATCH=16 \
  bash scripts/validate_detector.sh \
  pretrained/ir.pt /path/to/data.yaml ir
```

This writes JSON and CSV summaries below `runs/validation/ir/`. For one-pass
video-attribute evaluation, use `validate_attributes.py` with an attribute
manifest:

```bash
python validate_attributes.py \
  --weights pretrained/ir.pt \
  --data /path/to/data.yaml \
  --attribute-manifest /path/to/attribute_manifest.json \
  --output runs/validation/ir/attribute_metrics.json \
  --imgsz 1280 --device 0
```

## Repository layout

```text
.
├── train.py                       # detector/AA training entrypoint
├── validate.py                    # standard detector validation
├── validate_attributes.py         # one-pass attribute validation
├── infer.py                       # detector + tracker + rendering
├── scripts/                       # portable Bash entrypoints
└── ultralytics/
    ├── cfg/models/12/             # YOLOv12-UAV and AA model definitions
    ├── cfg/trackers/              # BoT-SORT and TrackTrack configs
    └── trackers/                  # tracker implementations
```

Datasets, checkpoints, experiment runs, logs, Slurm scripts, and local virtual
environments are intentionally excluded from Git.

## Notes and limitations

- ReID is disabled in all supplied tracker configurations.
- Challenge Track 1/2 test labels are not public; those sets can be used for
  inference and rendering but not local metric computation.
- The reported results depend on the prepared Anti-UAV410 split and should not
  be interpreted as official challenge submissions.
- See [`IMPLEMENTATION.md`](IMPLEMENTATION.md) for lower-level implementation
  notes and design decisions.

## Citation

If this repository is useful, please cite the original Dist-Tracker paper:

```bibtex
@inproceedings{wang2025dist,
  title     = {Dist-Tracker: A Small Object-aware Detector and Tracker for UAV Tracking},
  author    = {Wang, Wenzhen and Fu, Jing and Song, Jiayi and Li, Kaiyu and
               Qiao, Hui and Liu, Jiang and Sun, Hao and Cao, Xiangyong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition Workshops},
  pages     = {6601--6609},
  year      = {2025}
}
```

## Acknowledgements

This project builds on Dist-Tracker, Ultralytics YOLO, BoT-SORT, TrackTrack,
Anti-UAV410, and the 4th Anti-UAV Challenge.
