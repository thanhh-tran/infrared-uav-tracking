"""Train the YOLOv12 UAV detector."""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import SETTINGS
from ultralytics.utils import LOGGER


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "ultralytics/data/MOT-UAV.yaml"))
    parser.add_argument("--model", default=str(ROOT / "ultralytics/cfg/models/12/yolo12-UAV.yaml"))
    parser.add_argument("--weights", help="Optional detector checkpoint used to initialize the model")
    parser.add_argument(
        "--aa-weights",
        help="Optional AA checkpoint; load only aa_filters and anomaly_tests after initializing --weights",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--nbs", type=int, default=64, help="Nominal batch size used for gradient accumulation")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None, help="e.g. 0, 0,1, cpu or mps")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--optimizer",
        default="auto",
        choices=("auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"),
    )
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial/base learning rate")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum or Adam beta1")
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1)
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine learning-rate decay")
    parser.add_argument("--project", default=str(ROOT / "runs/train"))
    parser.add_argument("--name", default="yolov12_uav")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training images to use")
    parser.add_argument("--max-train-batches", type=int, default=-1, help="Cap batches per epoch for smoke testing")
    parser.add_argument("--max-val-batches", type=int, default=-1, help="Cap validation batches per rank")
    parser.add_argument("--ir-sine-p", type=float, default=0.5, help="IR sinusoidal augmentation probability")
    parser.add_argument("--no-multi-scale", action="store_true", help="Disable random multi-scale image resizing")
    parser.add_argument("--aa-lr", type=float, default=0.0, help="Optional separate LR for the parallel AA head")
    parser.add_argument("--wandb-offline", action="store_true", help="Log epochs to a local offline W&B run")
    parser.add_argument("--wandb-dir", default=str(ROOT / "wandb-offline"), help="Offline W&B storage directory")
    parser.add_argument("--quiet", action="store_true", help="Disable per-batch progress output")
    parser.add_argument("--no-plots", action="store_true", help="Disable training and validation plots")
    parser.add_argument("--no-val", action="store_true", help="Skip validation (useful for a quick smoke test)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.wandb_offline:
        wandb_dir = Path(args.wandb_dir).resolve()
        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = str(wandb_dir)
        os.environ.setdefault("WANDB_SILENT", "true")
        SETTINGS.update({"wandb": True})

    if args.resume:
        if not args.weights:
            raise SystemExit("--resume requires --weights path/to/last.pt")
        model = YOLO(args.weights)
    else:
        model = YOLO(args.model, task="detect")
        if args.weights:
            model.load(args.weights)
        if args.aa_weights:
            source, _ = attempt_load_one_weight(args.aa_weights, device="cpu", fuse=False)
            source_state = source.state_dict()
            target_state = model.model.state_dict()
            aa_state = {
                key: value
                for key, value in source_state.items()
                if (".aa_filters." in key or ".anomaly_tests." in key)
                and key in target_state
                and value.shape == target_state[key].shape
            }
            aa_filter_keys = [key for key in aa_state if ".aa_filters." in key]
            if not aa_filter_keys:
                raise SystemExit(f"No compatible AA filter weights found in {args.aa_weights}")
            incompatible = model.model.load_state_dict(aa_state, strict=False)
            LOGGER.info(
                f"Loaded {len(aa_state)} AA-only tensors from {args.aa_weights} "
                f"({len(aa_filter_keys)} aa_filters tensors); kept all non-AA tensors from {args.weights or args.model}"
            )
            del source, source_state, target_state, incompatible

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        nbs=args.nbs,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=args.momentum,
        warmup_epochs=args.warmup_epochs,
        warmup_bias_lr=args.warmup_bias_lr,
        cos_lr=args.cos_lr,
        project=args.project,
        name=args.name,
        seed=args.seed,
        fraction=args.fraction,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        ir_sine_p=args.ir_sine_p,
        multi_scale=not args.no_multi_scale,
        aa_lr=args.aa_lr,
        val=not args.no_val,
        resume=args.resume,
        exist_ok=args.exist_ok,
        verbose=not args.quiet,
        plots=not args.no_plots,
        # Dist-Tracker paper, Sec. 4.1.
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        box=7.5,
        cls=0.5,
        aa=0.5,
        dfl=1.5,
    )


if __name__ == "__main__":
    main()
