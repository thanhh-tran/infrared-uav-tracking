"""Validate a trained Dist-Tracker detector (AP50 and AP50-95)."""

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=str(ROOT / "ultralytics/data/MOT-UAV.yaml"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-val-batches", type=int, default=-1, help="Cap validation batches for smoke testing")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--project", default=str(ROOT / "runs/val"))
    parser.add_argument("--name", default="yolov12_uav")
    parser.add_argument("--output", type=Path, help="Write summary metrics to this JSON path and a sibling CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        max_val_batches=args.max_val_batches,
        split=args.split,
        project=args.project,
        name=args.name,
        conf=0.001,
        iou=0.7,
        # Validation only needs aggregate detector metrics.  Explicitly
        # override persisted Ultralytics settings so a previous inference run
        # cannot make this job write one prediction TXT per test frame.
        save_txt=False,
        save_json=False,
        plots=False,
        exist_ok=True,
    )
    print(f"AP50={metrics.box.map50:.6f} AP50-95={metrics.box.map:.6f}")
    if args.output:
        summary = {
            "weights": str(Path(args.weights).resolve()),
            "data": str(Path(args.data).resolve()),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            **{key: float(value) for key, value in metrics.results_dict.items()},
            **{f"speed/{key}_ms_per_image": float(value) for key, value in metrics.speed.items()},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        csv_path = args.output.with_suffix(".csv")
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
        print(f"JSON: {args.output}\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
