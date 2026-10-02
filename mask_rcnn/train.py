"""Train Mask R-CNN on BUSI with a seeded, saved train/val/test split.

The validation split is evaluated after every epoch for monitoring only: no
checkpoint is selected on it, and the hyperparameters are the ones from the
original coursework notebook. The test split is not touched here; run
evaluate.py once training is finished.

    python mask_rcnn/train.py --data-root data/Dataset_BUSI_with_GT
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torchvision

import busi
import utils
from engine import evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=busi.DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-file", type=Path, default=None,
                        help="default: mask_rcnn/splits/busi_seed<seed>.json")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: mask_rcnn/runs/seed<seed>")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--lr-step", type=int, default=3)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="start from random weights instead of COCO (smoke tests only)")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    args.split_file = args.split_file or here / "splits" / f"busi_seed{args.seed}.json"
    args.out = args.out or here / "runs" / f"seed{args.seed}"
    return args


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=busi.REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    busi.seed_everything(args.seed)
    device = busi.pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    records = busi.discover(args.data_root)
    busi.verify_decodable(args.data_root, records)
    fingerprint = busi.dataset_fingerprint(args.data_root, records)
    if fingerprint != busi.KAGGLE_SHA256:
        print(f"WARNING: {args.data_root} differs from the pinned {busi.KAGGLE_HANDLE}; "
              "results will not be comparable. Run mask_rcnn/download_data.py --force.")
    split = busi.load_or_create_split(records, args.split_file, args.seed)
    train_ds = busi.BUSIDataset(args.data_root, busi.records_for(records, split, "train"),
                                busi.get_transform(train=True))
    val_ds = busi.BUSIDataset(args.data_root, busi.records_for(records, split, "val"),
                              busi.get_transform(train=False))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=utils.collate_fn, worker_init_fn=busi.seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers,
        collate_fn=utils.collate_fn, worker_init_fn=busi.seed_worker,
    )

    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "split_sizes": {name: len(split[name]) for name in busi.SPLIT_NAMES},
        "dataset_sha256": fingerprint,
        "dataset_source": busi.KAGGLE_HANDLE if fingerprint == busi.KAGGLE_SHA256 else "unverified",
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "command": " ".join(sys.argv),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    model = busi.get_model(pretrained=not args.no_pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step,
                                                   gamma=args.lr_gamma)

    history = []
    for epoch in range(args.epochs):
        logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
        lr_scheduler.step()
        coco_evaluator = evaluate(model, val_loader, device=device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(logger.meters["loss"].global_avg, 4),
            "val": busi.coco_stats(coco_evaluator),
        })
        (args.out / "val_history.json").write_text(json.dumps(history, indent=2) + "\n")

    torch.save(model.state_dict(), args.out / "model_final.pth")
    print(f"Saved final weights and logs to {args.out}")


if __name__ == "__main__":
    main()
