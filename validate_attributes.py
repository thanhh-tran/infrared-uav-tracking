"""Run one full YOLO validation pass and report Anti-UAV410 video-attribute metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.modules.head import AnomalyDetect
from ultralytics.utils.metrics import DetMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--attribute-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Combined attribute metrics JSON")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-val-batches", type=int, default=-1, help="Smoke-test cap; requires --allow-partial")
    parser.add_argument("--allow-partial", action="store_true", help="Allow a partial full-pass smoke test")
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="antiuav410_attributes")
    parser.add_argument("--plots", action="store_true", help="Save Ultralytics PR/F1/P/R validation curves")
    parser.add_argument(
        "--confidence-distribution",
        action="store_true",
        help="Save p(confidence|TP) and p(confidence|FP), using the IoU=0.50 matching column",
    )
    parser.add_argument("--confidence-bins", type=int, default=100)
    parser.add_argument(
        "--aa-score-mode",
        choices=("product", "aa_only"),
        default="product",
        help="AA-head inference confidence: p_cls*p_AA (default) or p_AA only; rejected for a non-AA checkpoint",
    )
    return parser.parse_args()


def video_from_image(image: str) -> str:
    match = re.fullmatch(r"test__(.+)__(.+)", Path(image).stem)
    if not match:
        raise ValueError(f"Cannot recover Anti-UAV410 video from image name: {image}")
    return match.group(1)


class AttributeDetectionValidator(DetectionValidator):
    """Detection validator retaining AP inputs per image for zero-reinference subsets."""

    def __init__(
        self,
        *args,
        attribute_manifest: Path,
        attribute_output: Path,
        allow_partial: bool,
        confidence_distribution: bool,
        confidence_bins: int,
        aa_score_mode: str,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attribute_manifest_path = Path(attribute_manifest)
        self.attribute_output = Path(attribute_output)
        self.allow_partial = allow_partial
        self.confidence_distribution = confidence_distribution
        self.confidence_bins = confidence_bins
        self.aa_score_mode = aa_score_mode
        self.image_stats: list[dict[str, object]] = []

    def init_metrics(self, model):
        super().init_metrics(model)
        self.image_stats = []

    def update_metrics(self, preds, batch):
        """Mirror Ultralytics detection matching while retaining the image association."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = {
                "conf": torch.zeros(0, device=self.device),
                "pred_cls": torch.zeros(0, device=self.device),
                "tp": torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            }
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()

            if npr:
                if self.args.single_cls:
                    pred[:, 5] = 0
                predn = self._prepare_pred(pred, pbatch)
                stat["conf"] = predn[:, 4]
                stat["pred_cls"] = predn[:, 5]
                if nl:
                    stat["tp"] = self._process_batch(predn, bbox, cls)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            elif nl and self.args.plots:
                self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)

            # Preserve the upstream behavior: an empty background image adds no
            # AP inputs, while a background false positive still does.
            if npr or nl:
                for key in self.stats:
                    self.stats[key].append(stat[key])

            self.image_stats.append(
                {
                    "image": batch["im_file"][si],
                    "tp": stat["tp"].detach().cpu().numpy(),
                    "conf": stat["conf"].detach().cpu().numpy(),
                    "pred_cls": stat["pred_cls"].detach().cpu().numpy(),
                    "target_cls": stat["target_cls"].detach().cpu().numpy(),
                }
            )

    @staticmethod
    def _concatenate(records: list[dict[str, object]]) -> dict[str, np.ndarray]:
        return {
            "tp": np.concatenate([record["tp"] for record in records], axis=0),
            "conf": np.concatenate([record["conf"] for record in records]),
            "pred_cls": np.concatenate([record["pred_cls"] for record in records]),
            "target_cls": np.concatenate([record["target_cls"] for record in records]),
        }

    def get_stats(self):
        full_results = super().get_stats()
        full_stats = self._concatenate(self.image_stats)
        if self.confidence_distribution:
            self._plot_confidence_distribution(full_stats)
        manifest = json.loads(self.attribute_manifest_path.read_text())
        attributes = manifest["attributes"]
        output_root = self.attribute_manifest_path.parent
        seen_videos = {video_from_image(record["image"]) for record in self.image_stats}
        expected_videos = set()
        video_sets: dict[str, set[str]] = {}
        for slug, info in attributes.items():
            video_index = Path(info["video_index"])
            if not video_index.is_absolute():
                video_index = output_root / video_index
            videos = set(video_index.read_text().splitlines())
            if len(videos) != info["videos"]:
                raise ValueError(f"{video_index}: expected {info['videos']} unique videos, got {len(videos)}")
            video_sets[slug] = videos
            expected_videos.update(videos)
        if not self.allow_partial and seen_videos != expected_videos:
            raise ValueError(
                "Full validation videos do not match attribute manifest: "
                f"missing={sorted(expected_videos - seen_videos)}, extra={sorted(seen_videos - expected_videos)}"
            )
        if self.allow_partial and not seen_videos <= expected_videos:
            raise ValueError(f"Validation contains videos absent from the manifest: {sorted(seen_videos - expected_videos)}")

        rows: list[dict[str, object]] = []
        reports: dict[str, object] = {}
        for slug, info in attributes.items():
            selected_videos = video_sets[slug] & seen_videos
            if not selected_videos:
                continue
            records = [record for record in self.image_stats if video_from_image(record["image"]) in selected_videos]
            stats = self._concatenate(records)
            metrics = DetMetrics(save_dir=self.save_dir, plot=False, names=self.names)
            metrics.process(**stats)
            result = metrics.results_dict
            row = {
                "attribute": slug,
                "attribute_name": info["name"],
                "videos": len(selected_videos),
                "images": len(records),
                "instances": int(len(stats["target_cls"])),
                "predictions": int(len(stats["conf"])),
                "precision": float(result["metrics/precision(B)"]),
                "recall": float(result["metrics/recall(B)"]),
                "mAP50": float(result["metrics/mAP50(B)"]),
                "mAP50-95": float(result["metrics/mAP50-95(B)"]),
                "fitness": float(result["fitness"]),
            }
            rows.append(row)
            reports[slug] = row

        self.attribute_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": str(Path(self.args.model).resolve()) if self.args.model else None,
            "data": str(Path(self.args.data).resolve()),
            "aa_score_mode": self.aa_score_mode,
            "images": len(self.image_stats),
            "full_metrics": {key: float(value) for key, value in full_results.items()},
            "attributes": reports,
        }
        self.attribute_output.write_text(json.dumps(payload, indent=2) + "\n")
        csv_path = self.attribute_output.with_suffix(".csv")
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Attribute JSON: {self.attribute_output}")
        print(f"Attribute CSV: {csv_path}")
        return full_results

    def _plot_confidence_distribution(self, stats: dict[str, np.ndarray]) -> None:
        """Plot normalized confidence densities for IoU=0.50 true and false positive predictions."""
        import matplotlib.pyplot as plt

        confidence = np.asarray(stats["conf"], dtype=np.float64)
        tp50 = np.asarray(stats["tp"][:, 0], dtype=bool)
        tp_conf = confidence[tp50]
        fp_conf = confidence[~tp50]
        bins = int(self.confidence_bins)
        if bins < 2:
            raise ValueError(f"confidence_bins must be >= 2, got {bins}")
        edges = np.linspace(0.0, 1.0, bins + 1)
        width = np.diff(edges)
        tp_counts, _ = np.histogram(tp_conf, bins=edges)
        fp_counts, _ = np.histogram(fp_conf, bins=edges)
        tp_density = tp_counts / (max(int(tp_counts.sum()), 1) * width)
        fp_density = fp_counts / (max(int(fp_counts.sum()), 1) * width)
        centers = (edges[:-1] + edges[1:]) / 2

        self.save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.save_dir / "confidence_distribution.csv"
        with csv_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["bin_left", "bin_right", "bin_center", "tp_count", "fp_count", "tp_density", "fp_density"])
            writer.writerows(zip(edges[:-1], edges[1:], centers, tp_counts, fp_counts, tp_density, fp_density))

        summary = {
            "definition": "TP iff the Ultralytics one-to-one class-aware match is correct at IoU >= 0.50; FP otherwise",
            "validation_conf_floor": float(self.args.conf),
            "aa_score_mode": self.aa_score_mode,
            "predictions": int(confidence.size),
            "tp50": int(tp_conf.size),
            "fp50": int(fp_conf.size),
            "tp_confidence_mean": float(tp_conf.mean()) if tp_conf.size else None,
            "tp_confidence_median": float(np.median(tp_conf)) if tp_conf.size else None,
            "fp_confidence_mean": float(fp_conf.mean()) if fp_conf.size else None,
            "fp_confidence_median": float(np.median(fp_conf)) if fp_conf.size else None,
            "bins": bins,
        }
        (self.save_dir / "confidence_distribution.json").write_text(json.dumps(summary, indent=2) + "\n")

        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        ax.stairs(tp_density, edges, label=f"p(c | TP), n={tp_conf.size:,}", color="#1874D1", linewidth=2)
        ax.stairs(fp_density, edges, label=f"p(c | FP), n={fp_conf.size:,}", color="#D64545", linewidth=2)
        ax.set(xlim=(0, 1), xlabel="Detection confidence c", ylabel="Probability density", title="Confidence distribution at IoU=0.50")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(self.save_dir / "confidence_distribution.png", dpi=200)
        plt.close(fig)
        print(f"Confidence distribution PNG/CSV/JSON: {self.save_dir}")


def main() -> None:
    args = parse_args()
    if args.max_val_batches > 0 and not args.allow_partial:
        raise SystemExit("--max-val-batches requires --allow-partial to avoid reporting incomplete subsets as final")
    validator = partial(
        AttributeDetectionValidator,
        attribute_manifest=args.attribute_manifest.resolve(),
        attribute_output=args.output.resolve(),
        allow_partial=args.allow_partial,
        confidence_distribution=args.confidence_distribution,
        confidence_bins=args.confidence_bins,
        aa_score_mode=args.aa_score_mode,
    )
    model = YOLO(args.weights)
    head = model.model.model[-1]
    if args.aa_score_mode != "product":
        if not isinstance(head, AnomalyDetect):
            raise ValueError(f"--aa-score-mode={args.aa_score_mode} requires an AnomalyDetect checkpoint")
        head.inference_score_mode = args.aa_score_mode
    print(f"AA inference score mode: {args.aa_score_mode if isinstance(head, AnomalyDetect) else 'not-applicable'}")
    metrics = model.val(
        validator=validator,
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        max_val_batches=args.max_val_batches,
        split="test",
        project=args.project,
        name=args.name,
        conf=0.001,
        iou=0.7,
        save_txt=False,
        save_json=False,
        plots=args.plots,
        exist_ok=True,
    )
    print(f"Full AP50={metrics.box.map50:.6f} Full AP50-95={metrics.box.map:.6f}")


if __name__ == "__main__":
    main()
