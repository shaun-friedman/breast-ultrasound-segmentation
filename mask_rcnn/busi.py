"""Shared pieces of the reproducible BUSI Mask R-CNN pipeline.

Dataset discovery, the seeded train/val/test split, the dataset class, transforms,
the model factory and seeding live here so that train.py, evaluate.py and
figures.py all see exactly the same data and model definition.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import masks_to_boxes
from torchvision.transforms import v2 as T

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "Dataset_BUSI_with_GT"

CLASSES = ("normal", "benign", "malignant")
# torchvision reserves label 0 for background, so normal images contribute no
# instances rather than a "normal" label. num_classes therefore counts
# background + benign + malignant.
LABEL_IDS = {"benign": 1, "malignant": 2}
LABEL_NAMES = {v: k for k, v in LABEL_IDS.items()}
NUM_CLASSES = 1 + len(LABEL_IDS)

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

COCO_STAT_NAMES = (
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
)


def seed_everything(seed: int) -> None:
    """Seed every RNG in play and request deterministic kernels.

    Must run before any CUDA work. Some Mask R-CNN CUDA kernels (e.g. RoIAlign
    backward) have no deterministic implementation, so GPU runs can still differ
    slightly; warn_only keeps training possible instead of raising.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def discover(data_root: Path) -> list[dict]:
    """List every ultrasound image with all of its mask files.

    BUSI stores one mask per lesion: "<name>_mask.png", "<name>_mask_1.png", ...
    All of them are kept, so images with several lesions get several instances.
    """
    records = []
    for cls in CLASSES:
        class_dir = Path(data_root) / cls
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder {class_dir}")
        # Skip hidden files such as macOS "._name.png" metadata left by some archivers.
        files = sorted(
            f for f in os.listdir(class_dir) if f.endswith(".png") and not f.startswith(".")
        )
        for name in files:
            if "_mask" in name:
                continue
            stem = name[: -len(".png")]
            masks = [f for f in files if f.startswith(stem + "_mask")]
            if not masks:
                raise FileNotFoundError(f"No mask found for {cls}/{name}")
            records.append({
                "image": f"{cls}/{name}",
                "masks": [f"{cls}/{m}" for m in masks],
                "class": cls,
            })
    return records


def verify_decodable(data_root: Path, records: list[dict]) -> None:
    """Decode every image and mask once before training.

    A file the decoder rejects otherwise surfaces mid-epoch as an opaque error
    inside a DataLoader worker; here every bad file is named at once.
    """
    counts = {cls: sum(r["class"] == cls for r in records) for cls in CLASSES}
    print(f"Found {len(records)} images {counts} (BUSI publishes 780: 133/437/210)")
    bad = []
    for rec in records:
        files = [(rec["image"], ImageReadMode.RGB)] + [(m, ImageReadMode.GRAY) for m in rec["masks"]]
        for rel, mode in files:
            try:
                read_image(str(Path(data_root) / rel), mode=mode)
            except (RuntimeError, ValueError, OSError) as err:
                bad.append(f"{rel}: {str(err).strip() or type(err).__name__}")
    if bad:
        raise RuntimeError(
            f"{len(bad)} file(s) under {data_root} could not be decoded:\n  " + "\n  ".join(bad)
        )


def dataset_fingerprint(data_root: Path, records: list[dict]) -> str:
    """SHA-256 over every image and mask file, to confirm two runs used the same data."""
    digest = hashlib.sha256()
    for rec in records:
        for rel in [rec["image"], *rec["masks"]]:
            digest.update(rel.encode())
            digest.update(hashlib.sha256((Path(data_root) / rel).read_bytes()).digest())
    return digest.hexdigest()


def make_split(records: list[dict], seed: int, fractions: dict = DEFAULT_FRACTIONS) -> dict:
    """Class-stratified, seeded image-level split.

    The public BUSI release has no patient identifiers, so a patient-level split
    is not possible; images of the same patient may land in different splits.
    """
    rng = random.Random(seed)
    split = {name: [] for name in SPLIT_NAMES}
    for cls in CLASSES:
        images = sorted(r["image"] for r in records if r["class"] == cls)
        rng.shuffle(images)
        n_test = round(len(images) * fractions["test"])
        n_val = round(len(images) * fractions["val"])
        split["test"] += images[:n_test]
        split["val"] += images[n_test:n_test + n_val]
        split["train"] += images[n_test + n_val:]
    return {
        "seed": seed,
        "fractions": fractions,
        "level": "image (BUSI has no public patient IDs)",
        **{name: sorted(split[name]) for name in SPLIT_NAMES},
    }


def load_or_create_split(records: list[dict], split_file: Path, seed: int) -> dict:
    """Reuse a saved split so every run and script sees identical partitions."""
    split_file = Path(split_file)
    if split_file.exists():
        split = json.loads(split_file.read_text())
        if split["seed"] != seed:
            raise ValueError(f"{split_file} was made with seed {split['seed']}, not {seed}")
        listed = {img for name in SPLIT_NAMES for img in split[name]}
        found = {r["image"] for r in records}
        if listed != found:
            raise ValueError(
                f"{split_file} does not match the dataset on disk "
                f"({len(listed - found)} listed but missing, {len(found - listed)} unlisted)"
            )
        return split
    split = make_split(records, seed)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(split, indent=1) + "\n")
    print(f"Wrote new split to {split_file}")
    return split


def records_for(records: list[dict], split: dict, name: str) -> list[dict]:
    """Records of one partition, each with a stable image_id shared across scripts."""
    ids = {r["image"]: i for i, r in enumerate(records)}
    wanted = set(split[name])
    return [{**r, "id": ids[r["image"]]} for r in records if r["image"] in wanted]


class BUSIDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: Path, records: list[dict], transforms=None):
        self.data_root = Path(data_root)
        self.records = records
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = read_image(str(self.data_root / rec["image"]), mode=ImageReadMode.RGB)
        height, width = img.shape[-2:]

        instances = []
        for rel in rec["masks"]:
            mask = read_image(str(self.data_root / rel), mode=ImageReadMode.GRAY)[0] > 127
            if mask.shape != (height, width):
                raise ValueError(f"{rel} is {tuple(mask.shape)}, image is {(height, width)}")
            if mask.any():
                instances.append(mask)
        if rec["class"] == "normal" and instances:
            raise ValueError(f"Normal image {rec['image']} has a non-empty mask")

        masks = (
            torch.stack(instances).to(torch.uint8)
            if instances
            else torch.zeros((0, height, width), dtype=torch.uint8)
        )
        boxes = masks_to_boxes(masks)
        # A lesion one pixel wide or tall gives a zero-area box, which torchvision rejects.
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        masks, boxes = masks[keep], boxes[keep]
        num = len(masks)
        label = LABEL_IDS.get(rec["class"], 0)

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(height, width)),
            "masks": tv_tensors.Mask(masks),
            "labels": torch.full((num,), label, dtype=torch.int64),
            "image_id": rec["id"],
            # COCO convention: instance area is the mask area, used for small/medium/large.
            "area": masks.flatten(1).sum(1).to(torch.float32),
            "iscrowd": torch.zeros((num,), dtype=torch.int64),
        }
        img = tv_tensors.Image(img)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


def get_transform(train: bool):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def get_model(pretrained: bool = True):
    """COCO-pretrained Mask R-CNN (ResNet-50 FPN) with new box and mask heads."""
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)
    return model


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def coco_stats(coco_evaluator) -> dict:
    """COCOeval summary as {iou_type: {stat: value}}; -1 (no instances) becomes None."""
    out = {}
    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
        out[iou_type] = {
            name: (None if value < 0 else round(float(value), 4))
            for name, value in zip(COCO_STAT_NAMES, coco_eval.stats)
        }
    return out
