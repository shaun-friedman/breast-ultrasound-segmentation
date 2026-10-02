"""Render qualitative results for held-out test images only.

Picks the first --per-class test images of each class in sorted filename order
(no cherry-picking), and draws input, ground truth and prediction side by side.

    python mask_rcnn/figures.py --run mask_rcnn/runs/seed42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks  # noqa: E402

import busi  # noqa: E402
from evaluate import load_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path,
                        default=busi.REPO_ROOT / "docs" / "figures" / "mask_rcnn_test_examples.png")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def overlay(image: torch.Tensor, masks: torch.Tensor, boxes: torch.Tensor, color: str):
    if len(masks):
        image = draw_segmentation_masks(image, masks.bool(), alpha=0.45, colors=color)
        image = draw_bounding_boxes(image, boxes.round().long(), colors=color, width=3)
    return image.permute(1, 2, 0)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = busi.pick_device(args.device)
    _, data_root, records, split, model = load_run(args.run, args.data_root, device)
    test = busi.records_for(records, split, "test")
    chosen = [r for cls in busi.CLASSES for r in [t for t in test if t["class"] == cls][:args.per_class]]
    dataset = busi.BUSIDataset(data_root, chosen, busi.get_transform(train=False))

    fig, axes = plt.subplots(len(chosen), 3, figsize=(12, 4 * len(chosen)), squeeze=False)
    for row, (rec, ax) in enumerate(zip(chosen, axes)):
        img, target = dataset[row]
        pred = model([img.to(device)])[0]
        keep = pred["scores"] >= args.score_threshold
        pred_masks = (pred["masks"][keep, 0] >= args.mask_threshold).cpu()
        pred_boxes = pred["boxes"][keep].cpu()
        found = [f"{busi.LABEL_NAMES[int(label)]} {score:.2f}"
                 for label, score in zip(pred["labels"][keep], pred["scores"][keep])]
        if len(found) > 3:  # detections are sorted by score
            found = found[:3] + [f"+{len(found) - 3} more"]
        image_u8 = (img * 255).to(torch.uint8)

        ax[0].imshow(image_u8.permute(1, 2, 0))
        ax[0].set_title(f"Input: {rec['image']}", fontsize=9)
        ax[1].imshow(overlay(image_u8, target["masks"], target["boxes"], "lime"))
        ax[1].set_title(f"Ground truth: {rec['class']}", fontsize=9)
        ax[2].imshow(overlay(image_u8, pred_masks, pred_boxes, "red"))
        ax[2].set_title("Prediction: " + (", ".join(found) or "no detection"), fontsize=9)
        for a in ax:
            a.axis("off")

    fig.suptitle(f"Mask R-CNN on held-out BUSI test images (score >= {args.score_threshold})")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
