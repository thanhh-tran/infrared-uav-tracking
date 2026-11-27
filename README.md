# [DIST-Track]

> **Project reimplementation:** reproducible YOLOv12/AA detector training,
> validation, BoT-SORT and TrackTrack inference are implemented in this checkout.
> Use the root-level `train.py`, `validate.py`, and `infer.py`; see
> [`IMPLEMENTATION.md`](IMPLEMENTATION.md) for reproducible commands, verified
> behavior, and assumptions not specified by the paper.

## 🏆 Dist-Tracker: A Small Object-aware Detector and Tracker for UAV Tracking (CVPRW2025)

As the core technology of the international championship team in ​Track 3 (Multi-UAV Complex Scene Tracking)​​ of the ​4th Anti-UAV Challenge at CVPR, ​Dist-Tracker​ innovatively combines an enhanced detector with a lightweight tracker architecture. The system achieves breakthrough performance on ​300 video sequences (200 for training, 100 for testing)​​ captured in ​thermal infrared (640×512 resolution)

- ​**​[2025] [​Track 3 (Multi-UAV Complex Scene Tracking)​​ of the ​4th Anti-UAV Challenge at CVPR]​**​

- The international championship **(Top-1 MOTA: 81.32%)**

Paper:​​ https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/papers/Wang_Dist-Tracker_A_Small_Object-aware_Detector_and_Tracker_for_UAV_Tracking_CVPRW_2025_paper.pdf

## 🛠️ Pipeline

![Pipeline Diagram](docs/pipeline.png)

## 📌 Key innovations

- YOLOv12 small-object detector with optional IR augmentation

- BoT-SORT and a non-ReID TrackTrack port for UAV tracking

- GMC (Global Motion Compensation) module​ based on ​sparse optical flow, mitigating camera shake and fast motion

## 📊 Tracking Results

Performance Visualization

![Results](docs/results.png)

## 🗂 Dataset Organization

Directory Structure

```bash
dataset/
├── train/
│   ├── images/          # Thermal images (640x512)
│   │   ├── seq01_0001.png
│   │   └── ...
│   └── labels/          # YOLO-format annotations
│       ├── seq01_0001.txt
│       └── ...
├── test/                # Same structure as train/
└── Videos/              # Official test videos
```

​Annotation Format (YOLO)​

Each .txt file contains one line per object:

```bash
<class_id> <x_center> <y_center> <width> <height>
```

Coordinates are ​normalized​ to [0, 1]

class_id=0 for UAVs (only one class in this dataset)

## 🚀 Quick start

### ​Environment Configuration

```bash
python -m venv .venv
source .venv/bin/activate
# Install a CUDA-compatible torch/torchvision first, then:
pip install -r requirements-disttracker.txt
```

### Run Inference

```bash
# Train detector:
python train.py --data /path/to/anti_uav.yaml --weights ultralytics/best.pt

# Validate detector:
python validate.py --weights runs/train/yolov12_uav/weights/best.pt --data /path/to/anti_uav.yaml

# Infer + BoT-SORT tracking:
python infer.py --weights ultralytics/best.pt --source ultralytics/dataset/Videos/MultiUAV-257.mp4
```

### Public Bash entrypoints

The scripts below are portable wrappers around the Python entrypoints. They do
not require Slurm, environment modules, Jean-Zay paths, or a particular cluster.
All data and checkpoint paths are supplied by the caller.

```bash
# Baseline or IR-augmented detector training
bash scripts/train_detector.sh baseline /path/to/data.yaml /path/to/initial.pt
bash scripts/train_detector.sh ir       /path/to/data.yaml /path/to/initial.pt

# Two-phase AA training: AA warm-up followed by fresh-optimizer fine-tuning
bash scripts/train_aa.sh /path/to/data.yaml /path/to/ir_detector.pt

# One of: baseline_botsort, ir_botsort, ir_tracktrack,
#         aa_botsort, aa_tracktrack
bash scripts/infer_variant.sh ir_tracktrack /path/to/ir.pt /path/to/video_or_frames

# Passing the raw test root processes all immediate video directories serially
bash scripts/infer_variant.sh ir_tracktrack /path/to/ir.pt /path/to/anti_uav410_raw_test

# Standard YOLO metrics
bash scripts/validate_detector.sh /path/to/best.pt /path/to/data.yaml ir

# Official per-video SA/AOA; the optional third path enables 10-attribute summaries
bash scripts/evaluate_aoa.sh \
  /path/to/anti_uav410_raw_test \
  runs/inference/ir_tracktrack \
  /path/to/official/annos/test/att
```

Common runtime settings can be overridden without editing the scripts, for
example `DEVICE=0,1 BATCH=32`, `EPOCHS=20`, or `PYTHON_BIN=/path/to/python`.

TrackTrack is available as a clean non-ReID port of the Ultralytics 8.4.63+
backend. It includes NSA-Kalman, HMIoU/confidence/corner-angle association,
iterative matching, Track-Aware Initialization, GMC, and loose-NMS recovery:

```bash
python infer.py --weights /path/to/best.pt --source /path/to/frames \
  --tracker ultralytics/cfg/trackers/tracktrack_uav.yaml
```

For the AA checkpoint, use `tracktrack_aa.yaml` (detector and
tracking floor 0.001; high/new-track threshold 0.05). ReID is not included in
this minimal port; the supplied TrackTrack configs use `with_reid: false`.

IR augmentation:

```bash
python train.py --data datasets/anti_uav_combined.yaml --weights ultralytics/best.pt --ir-sine-p 0.5
```
## ⭐️ Citation

If you find this project useful, welcome to cite us.

```bib
@inproceedings{wang2025dist,
  title={Dist-Tracker: A Small Object-aware Detector and Tracker for UAV Tracking},
  author={Wang, Wenzhen and Fu, Jing and Song, Jiayi and Li, Kaiyu and Qiao, Hui and Liu, Jiang and Sun, Hao and Cao, Xiangyong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={6601--6609},
  year={2025}
}
```

## 🙏 Acknowledgement

This work is made possible by foundational open-source projects including the [YOLO series](https://github.com/ultralytics), whose pioneering contributions revolutionized object detection.
