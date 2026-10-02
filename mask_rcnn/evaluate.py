"""Evaluate a trained run once on the held-out test split.

Reports COCO box and mask AP/AR (class-aware, so a correct outline with the
wrong benign/malignant label counts as a miss) and an image-level 3x3 confusion
matrix. An image's predicted class is the label of its highest-scoring detection
at or above --score-threshold, or "normal" when there is none. The threshold
defaults to 0.5, the value used in the original notebook; it was not tuned.

    python mask_rcnn/evaluate.py --run mask_rcnn/runs/seed42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import busi
import utils
from engine import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="directory written by train.py")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="default: the data root recorded in the run's config.json")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_run(run: Path, data_root: Path | None, device: torch.device):
    config = json.loads((run / "config.json").read_text())
    busi.seed_everything(config["seed"], config.get("deterministic", False))
    data_root = data_root or Path(config["data_root"])
    records = busi.discover(data_root)
    if busi.dataset_fingerprint(data_root, records) != config["dataset_sha256"]:
        raise ValueError(f"Dataset at {data_root} differs from the one this run was trained on")
    split = busi.load_or_create_split(records, Path(config["split_file"]), config["seed"])
    model = busi.get_model(pretrained=False)
    model.load_state_dict(torch.load(run / "model_final.pth", map_location="cpu", weights_only=True))
    model.to(device).eval()
    return config, data_root, records, split, model


@torch.inference_mode()
def image_level(model, dataset, device, threshold: float) -> dict:
    classes = list(busi.CLASSES)
    matrix = [[0] * len(classes) for _ in classes]
    for idx in range(len(dataset)):
        img, _ = dataset[idx]
        pred = model([img.to(device)])[0]
        keep = pred["scores"] >= threshold
        if keep.any():
            top = int(pred["scores"][keep].argmax())
            predicted = busi.LABEL_NAMES[int(pred["labels"][keep][top])]
        else:
            predicted = "normal"
        true = dataset.records[idx]["class"]
        matrix[classes.index(true)][classes.index(predicted)] += 1

    def rate(num: int, den: int) -> dict:
        return {"value": round(num / den, 4) if den else None, "n": f"{num}/{den}"}

    n, b, m = (classes.index(c) for c in ("normal", "benign", "malignant"))
    lesion_rows = (b, m)
    return {
        "score_threshold": threshold,
        "classes": classes,
        "confusion_matrix_rows_true_cols_pred": matrix,
        # Any lesion found in a benign/malignant image, regardless of its label.
        "lesion_sensitivity": rate(sum(matrix[r][c] for r in lesion_rows for c in (b, m)),
                                   sum(sum(matrix[r]) for r in lesion_rows)),
        # Normal images with no detection at or above the threshold.
        "normal_specificity": rate(matrix[n][n], sum(matrix[n])),
        "malignant_sensitivity": rate(matrix[m][m], sum(matrix[m])),
        "accuracy_3class": rate(sum(matrix[i][i] for i in range(3)), sum(map(sum, matrix))),
    }


def main() -> None:
    args = parse_args()
    device = busi.pick_device(args.device)
    config, data_root, records, split, model = load_run(args.run, args.data_root, device)

    dataset = busi.BUSIDataset(data_root, busi.records_for(records, split, args.split),
                               busi.get_transform(train=False))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         collate_fn=utils.collate_fn)
    coco_evaluator = evaluate(model, loader, device=device)

    counts = {c: sum(r["class"] == c for r in dataset.records) for c in busi.CLASSES}
    results = {
        "split": args.split,
        "n_images": len(dataset),
        "images_per_class": counts,
        "coco": busi.coco_stats(coco_evaluator),
        "image_level": image_level(model, dataset, device, args.score_threshold),
        "dataset_sha256": config["dataset_sha256"],
    }
    out = args.run / f"{args.split}_metrics.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["image_level"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
