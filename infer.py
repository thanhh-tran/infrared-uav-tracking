"""Run a YOLO detector with BoT-SORT, ByteTrack, or TrackTrack."""

import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from ultralytics import YOLO
from ultralytics.trackers.output_selection import StrongBaselineOutputSelector
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.trackers.track_tracker import extract_loose_detections
from ultralytics.utils import IterableSimpleNamespace, TQDM, yaml_load
from ultralytics.utils.checks import check_yaml


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def synchronize_device(device):
    """Wait for asynchronous accelerator work before recording wall-clock timings."""
    device = str(device or "")
    if device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(ROOT / "ultralytics/best.pt"))
    parser.add_argument("--source", default=str(ROOT / "ultralytics/dataset/Videos/MultiUAV-257.mp4"))
    parser.add_argument("--tracker", default=str(ROOT / "ultralytics/cfg/trackers/botsort_uav.yaml"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default=str(ROOT / "runs/track"))
    parser.add_argument("--name", default="uav_botsort")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--save-video", action="store_true", help="Encode an image-directory source as one MP4")
    parser.add_argument(
        "--compact-labels",
        action="store_true",
        help="Write one predictions.jsonl per source instead of one TXT file per frame",
    )
    parser.add_argument("--fps", type=float, default=25.0, help="Output FPS for an image-directory video")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--max-frames", type=int, help="Optional limit for smoke tests")
    return parser.parse_args()


def iter_source(source):
    """Yield (path, frame, fps, is_video) from an image, directory, or video."""
    path = Path(source)
    if path.is_dir():
        for image_path in sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Cannot read {image_path}")
            yield image_path, frame, 1.0, False
        return
    if path.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Cannot read {path}")
        yield path, frame, 1.0, False
        return
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield path, frame, fps, True
    finally:
        capture.release()


def source_frame_count(source):
    """Return the number of directly processable frames in an image, directory, or video source."""
    path = Path(source)
    if path.is_dir():
        return sum(item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES for item in path.iterdir())
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return 1
    capture = cv2.VideoCapture(str(path))
    try:
        return max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))) if capture.isOpened() else 0
    finally:
        capture.release()


def save_track_labels(path, tracks, shape):
    """Save class, normalized xywh, confidence, and track id."""
    height, width = shape[:2]
    rows = []
    for x1, y1, x2, y2, track_id, score, cls, _ in tracks:
        rows.append(
            f"{int(cls)} {(x1 + x2) / (2 * width):.8f} {(y1 + y2) / (2 * height):.8f} "
            f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f} {score:.8f} {int(track_id)}"
        )
    path.write_text("\n".join(rows) + ("\n" if rows else ""))


def compact_prediction(frame_name, boxes, shape):
    """Build one compact JSON-serializable prediction row from tracker-format boxes."""
    height, width = shape[:2]
    predictions = []
    for x1, y1, x2, y2, track_id, score, cls, _ in boxes:
        predictions.append(
            {
                "class": int(cls),
                "xywhn": [
                    float((x1 + x2) / (2 * width)),
                    float((y1 + y2) / (2 * height)),
                    float((x2 - x1) / width),
                    float((y2 - y1) / height),
                ],
                "confidence": float(score),
                "track_id": int(track_id),
            }
        )
    return {"frame": frame_name, "predictions": predictions}


def run_tracked(args, model):
    """Run full-frame detection and update the configured tracker once per source frame."""
    tracker_cfg = IterableSimpleNamespace(**yaml_load(check_yaml(args.tracker)))
    if tracker_cfg.tracker_type not in TRACKER_MAP:
        raise ValueError(f"Unsupported tracker_type={tracker_cfg.tracker_type!r}; expected one of {sorted(TRACKER_MAP)}")

    def setting(name, default):
        cli_value = getattr(args, name, None)
        return getattr(tracker_cfg, name, default) if cli_value is None else cli_value

    min_box_area = float(setting("min_box_area", 10))
    tracker_class = TRACKER_MAP[tracker_cfg.tracker_type]
    if tracker_cfg.tracker_type == "tracktrack":
        model.add_callback("on_predict_start", lambda predictor: tracker_class.setup_predictor(predictor))
    output = Path(args.project) / args.name
    labels_dir = output / "labels"
    if not args.compact_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)
    compact_file = None
    tracker = None
    selector = StrongBaselineOutputSelector(min_box_area)
    writer = None
    writer_source = None
    frame_count = 0
    total_frames = source_frame_count(args.source)
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames) if total_frames else args.max_frames
    progress = TQDM(total=total_frames or None, desc=f"Tracking {Path(args.source).name}")
    synchronize_device(args.device)
    started = time.perf_counter()
    try:
        if args.compact_labels:
            output.mkdir(parents=True, exist_ok=True)
            compact_file = (output / "predictions.jsonl").open("w")
        for source_path, frame, fps, is_video in iter_source(args.source):
            if tracker is None:
                # Image-directory sources do not carry video metadata. iter_source()
                # uses 1.0 as a sentinel, but tracker buffers must use the requested
                # playback FPS rather than shrinking a 30-frame buffer to one frame.
                tracker_fps = fps if is_video else args.fps
                tracker = tracker_class(args=tracker_cfg, frame_rate=round(tracker_fps))
                print(
                    f"tracker={tracker_cfg.tracker_type} frame_rate={round(tracker_fps)} "
                    f"track_buffer={tracker_cfg.track_buffer} max_time_lost={tracker.max_time_lost} "
                    f"high_thresh={tracker_cfg.track_high_thresh} low_thresh={tracker_cfg.track_low_thresh} "
                    f"new_track_thresh={tracker_cfg.new_track_thresh}"
                )
                print(f"output_selection=strong_baseline min_box_area={min_box_area}")
            prediction = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                project=str(Path(args.project) / "_ultralytics_internal"),
                name=args.name,
                exist_ok=True,
                save=False,
                save_frames=False,
                save_txt=False,
                save_conf=False,
                save_crop=False,
                save_json=False,
                save_hybrid=False,
                show=False,
                visualize=False,
                augment=False,
                plots=False,
                verbose=False,
            )[0]
            update_kwargs = {}
            if tracker_cfg.tracker_type == "tracktrack":
                update_kwargs["dets_del"] = extract_loose_detections(prediction)
            tracker.update(prediction.boxes.cpu().numpy(), frame, **update_kwargs)
            tracks = selector.select(tracker)
            frame_count += 1
            progress.update(1)
            frame_name = f"{source_path.stem}_{frame_count:06d}" if is_video else source_path.stem
            if args.compact_labels:
                compact_file.write(json.dumps(compact_prediction(frame_name, tracks, frame.shape)) + "\n")
            else:
                save_track_labels(labels_dir / f"{frame_name}.txt", tracks, frame.shape)

            annotated = frame.copy()
            for x1, y1, x2, y2, track_id, score, cls, _ in tracks:
                cv2.rectangle(annotated, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 1)
                cv2.putText(
                    annotated,
                    f"{int(track_id)} {score:.2f}",
                    (round(x1), max(12, round(y1) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            if args.save_frames:
                frames_dir = output / "frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frames_dir / f"{source_path.stem}_{frame_count:06d}.jpg"), annotated)
            if not args.no_save and (is_video or args.save_video):
                video_source = source_path if is_video else Path(args.source)
                if writer is None or writer_source != video_source:
                    if writer is not None:
                        writer.release()
                    height, width = frame.shape[:2]
                    output_fps = fps if is_video else args.fps
                    writer = cv2.VideoWriter(
                        str(output / f"{video_source.stem}.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create output video in {output}")
                    writer_source = video_source
                writer.write(annotated)
            elif not args.no_save:
                cv2.imwrite(str(output / source_path.name), annotated)
            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        if compact_file is not None:
            compact_file.close()
        if writer is not None:
            writer.release()
        progress.close()
    synchronize_device(args.device)
    elapsed = time.perf_counter() - started
    fps = frame_count / elapsed if elapsed else 0.0
    print(f"Processed {frame_count} tracked frame(s) in {elapsed:.3f}s ({fps:.3f} FPS). Results: {output}")


def main():
    args = parse_args()
    model = YOLO(args.weights)
    run_tracked(args, model)


if __name__ == "__main__":
    main()
