# Dist-Tracker: hướng dẫn implementation và tái lập

Repo này giữ framework Ultralytics 8.3.88/YOLOv12 của tác giả và bổ sung quy
trình train, validation và inference detector + BoT-SORT/TrackTrack có thể tái lập.

## Những gì đã được bổ sung

### Detector YOLOv12

Detector dùng box loss CIoU và Distribution Focal Loss chuẩn của YOLOv12, với
các gain mặc định `box:cls:dfl = 7.5:0.5:1.5` của Ultralytics.

YOLOv12 trong Ultralytics 8.3.88 là detector anchor-free, dùng BCE cho class và
Distribution Focal Loss cho box distribution; nó không có objectness head riêng.
Implementation giữ kiến trúc thực tế này thay vì tự thêm một head không tồn tại,
mặc dù ký hiệu Eq. 9 trong paper gọi thành phần cuối là `L_obj`.

### Tracker

Pipeline cuối giữ BoT-SORT gốc cho baseline và một TrackTrack non-ReID riêng.
TrackTrack sử dụng NSA-Kalman riêng, HMIoU/confidence/corner-angle association,
Track-Aware Initialization, GMC và loose-NMS recovery. BoT-SORT không chứa
logic hoặc cost tùy biến của tracker khác.

## Tham số và mức độ chắc chắn

| Thành phần | Giá trị | Nguồn |
|---|---:|---|
| Detection threshold | 0.01 | Paper, Sec. 4.3.2 |
| New-track threshold | 0.25 | Paper, Sec. 4.3.2 |
| Quality window | 100 iteration | Paper, Eq. 5–6 |
| Input chính | 1280 | Paper, Sec. 4.2 |
| `w_s` | 0.8 | Public source; paper chỉ nói `w_s > 0.5` |
| Shape scale | 0.6 | Giá trị public source truyền vào loss; paper không công bố |
| Quality sigma | 2.5 | Giá trị tương ứng trong public source; paper không công bố |
| Assignment threshold | 0.8 | Public `botsort.yaml`; paper không công bố |
| Track buffer | 30 frame | Public `botsort.yaml`; paper không công bố |

## Cài đặt

Python 3.9–3.11 được khuyến nghị. Cài PyTorch phù hợp với CUDA của máy train
trước, sau đó cài dependency còn lại:

```bash
python -m venv .venv
source .venv/bin/activate

# Chọn lệnh PyTorch/CUDA chính xác tại https://pytorch.org/get-started/locally/
pip install torch torchvision
pip install -r requirements-disttracker.txt
```

Không dùng `requirements.txt` gốc để tạo môi trường mới: đó là snapshot máy tác
giả và chứa đường dẫn private `-e /data1/wwz/YOLO12`.

## Dataset

YAML mặc định dùng sample rất nhỏ đi kèm repo. Dataset đầy đủ nên có dạng:

```text
dataset/
├── Yolo_train/
│   ├── images/
│   └── labels/
└── Yolo_val/
    ├── images/
    └── labels/
```

Mỗi label theo format YOLO:

```text
class_id x_center y_center width height
```

Tạo YAML riêng cho dataset thật, ưu tiên `path` tuyệt đối:

```yaml
path: /absolute/path/to/dataset
train: Yolo_train/images
val: Yolo_val/images
test: Yolo_test/images  # nếu có
names:
  0: drone
```

Paper chia 200 video thành 160 train và 40 validation nhưng không công bố seed
hoặc danh sách sequence. Muốn so sánh chính xác với paper cần split list của tác
giả; không nên chia frame ngẫu nhiên vì sẽ gây video leakage.

## Train

Train từ checkpoint detector do repo cung cấp:

```bash
python train.py \
  --data /path/to/anti_uav.yaml \
  --weights ultralytics/best.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 1280 \
  --device 0
```

Train từ đầu bằng YOLOv12-S:

```bash
python train.py --data /path/to/anti_uav.yaml --epochs 50 --imgsz 1280 --device 0
```

Resume:

```bash
python train.py --resume --weights runs/train/yolov12_uav/weights/last.pt
```

## Validation detector

```bash
python validate.py \
  --weights runs/train/yolov12_uav/weights/best.pt \
  --data /path/to/anti_uav.yaml \
  --imgsz 1280 \
  --device 0
```

Lệnh in `AP50` và `AP50-95`. YOLO label không chứa track ID, nên chưa thể tính
MOTA/ID switches chỉ từ dataset format công khai trong repo. Tracking validation
cần annotation có `frame_id, track_id, bbox`; khi có ground truth thật, cần thêm
converter/evaluator tương ứng thay vì suy đoán format.

## Inference detector + tracker

```bash
python infer.py \
  --weights runs/train/yolov12_uav/weights/best.pt \
  --source /path/to/video.mp4 \
  --imgsz 1280 \
  --conf 0.01 \
  --device 0
```

Mặc định output nằm trong `runs/track/uav_botsort`, gồm video và text
label. Dòng text có dạng Ultralytics:

```text
class x_center y_center width height confidence track_id
```

`--source` cũng nhận một ảnh hoặc thư mục ảnh. Dùng `--max-frames N` để smoke
test một đoạn ngắn và `--no-save` nếu không muốn ghi video.

### IR sinusoidal augmentation

Augmentation của Ustun et al. được bật bằng `--ir-sine-p`. Một phase
`theta ~ Uniform(0, 2*pi)` được lấy cho toàn ảnh và áp dụng công thức
`y=(sin(2*pi*x+theta)+1)/2`; validation và inference không dùng transform này.
Paper không công bố probability tốt nhất, vì vậy CLI mặc định dùng 0.5 và cần
ablation với 0, 0.25, 0.5 và 1.0.

```bash
python train.py --data datasets/anti_uav_combined.yaml \
  --weights ultralytics/best.pt --ir-sine-p 0.5 --epochs 50 --imgsz 1280 --device 0
```

## Kiểm tra

```bash
python -m unittest discover -s tests -v
```

Test bao phủ BoT-SORT/TrackTrack, AA head, CIoU/DFL và fallback identity của
GMC khi frame không có feature. Trong môi trường phát triển hiện
tại, unit test, train 1 epoch, validation và inference 3 frame đều đã chạy thành
công.

## Giới hạn tái lập kết quả paper

- Repo chỉ kèm một ảnh train, một ảnh val và một video mẫu; không đủ để tái lập
  AP/MOTA báo cáo.
- Không có split 160/40 chính thức.
- Paper dùng hai RTX 4090 và input 1280; smoke test CPU/input nhỏ không phải kết
  quả chất lượng.

## Dataset chính thức đã chuẩn bị

- Track 3 được tải từ `MultiUAV_Train.zip` chính thức (200 video), chia theo
  sequence bằng seed 42 thành 160 train và 40 validation. Converter:
  `tools/prepare_anti_uav.py`.
- Bộ train dùng chung Track 1/Track 2 gồm 223 sequence. Converter
  `tools/prepare_anti_uav_track12.py` giữ 247,458 frame và loại 121 frame có
  `exist=1` nhưng bbox zero-area theo quyết định đã xác nhận; không biến chúng
  thành negative.
- `datasets/anti_uav_combined.yaml` dùng protocol đã xác nhận: Track 2 được
  chia ở cấp sequence bằng seed 42 thành 178 train/45 validation, Track 3 giữ
  nguyên split 160 train/40 validation. Sau khi gộp có 338 sequence train và
  85 sequence validation. Track 2 train lấy frame đầu rồi mỗi 5 frame tiếp theo;
  validation giữ nguyên full frame rate. Phần split Track 2 là protocol tùy
  chỉnh; manifest nằm ở
  `datasets/yolo_combined_338_85_stride5/manifest.json`.

Train đầy đủ trên GPU:

```bash
python train.py --data datasets/anti_uav_combined.yaml --weights ultralytics/best.pt \
  --epochs 50 --batch 16 --imgsz 1280 --device 0
```

Manifest tái lập nằm tại `datasets/yolo_track3/split_manifest.json` và
`datasets/yolo_track12/manifest.json`. Toàn bộ `datasets/` được git-ignore vì
kích thước lớn.

## AA-YOLOv12 experiment

`ultralytics/cfg/models/12/yolo12-UAV-AA.yaml` retains the one-class YOLOv12
classification branch and adds three independent Anomaly-Aware objectness
heads in parallel. The regression/DFL branch, backbone and neck remain
YOLOv12. The statistical test is evaluated in FP32 under AMP and returns a
probability in `[0, 1]`.

There are deliberately two score contracts:

- baseline `Detect`: class logits -> sigmoid for TAL/inference, BCE classification;
- `AnomalyDetect`: class probability alone drives TAL, while inference uses
  `sigmoid(class_logit) * AA_objectness`; AA objectness uses an additional MSE loss.

This separation prevents applying sigmoid twice and prevents the randomly
initialized AA branch from destabilizing TAL. During the first ten loss
batches, the trainer logs foreground count, GT count, score range and target
score sum so an early Task-Aligned Assigner collapse is visible. AA heads always
use the same standard CIoU and DFL regression losses as the baseline.

Example training from the existing YOLOv12 detector checkpoint:

```bash
python train.py \
  --model ultralytics/cfg/models/12/yolo12-UAV-AA.yaml \
  --weights /path/to/yolov12_baseline_best.pt \
  --data datasets/anti_uav410_stride5/dataset.yaml
```

Inference uses the normal `infer.py`; no AA-specific post-processing is needed.
Inspect raw P3/P4/P5 confidence maps with:

```bash
python tools/visualize_aa_confidence.py \
  --weights /path/to/aa_best.pt --source /path/to/frame.jpg \
  --imgsz 1280 --device 0 --output runs/aa_maps
```
