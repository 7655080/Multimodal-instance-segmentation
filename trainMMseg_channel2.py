"""YOLOMM segmentation training entry for depth-channel and model ablation experiments."""

import argparse
import gc
import os
import re
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
MM_MODEL_ROOT = PROJECT_ROOT / "ultralytics/cfg/models/mm"
MODEL_ROOT = MM_MODEL_ROOT / "26"
DEFAULT_DATA = {
    "gray": WORKSPACE_ROOT / "datasetxiang8/data_RIR_seg.yaml",
    "color": WORKSPACE_ROOT / "datasetxiang8/data_IR_seg.yaml",
}

# Keep Ultralytics settings inside this project when the script is run directly.
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics_mm"))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# ================== Edit current experiment here ==================
# Depth source:
#   SCRIPT_DEPTH_SOURCE = "gray"   -> use images_depth
#   SCRIPT_DEPTH_SOURCE = "color"  -> use images_ir as color depth
#
# Depth channel selection:
#   gray:  "gray1" / "gray2" / "gray3"
#   color: "b" / "g" / "r" / "bg" / "br" / "gr" / "bgr"
SCRIPT_DEPTH_SOURCE = "gray"
SCRIPT_X_CHANNEL_SELECT = "gray1"
SCRIPT_MODEL_SCALE = "s"

# Train several models sequentially under the same depth-source/channel setting.
# Each model can use its own optimizer. Training stops if any model fails.
SCRIPT_MODEL_CONFIGS = [
    {
        "tag": "yolo26n_mm_mid_seg",
        "model": MODEL_ROOT / "yolo26n-mm-mid-seg.yaml",
        "optimizer": "MuSGD",
    },
    {
        "tag": "yolo12n_mm_mid_seg",
        "model": MM_MODEL_ROOT / "12/yolo12n-mm-mid-seg.yaml",
        "optimizer": "MuSGD",
    },
    # {
    #     "tag": "yolo12n_mm_mid_seg1",
    #     "model": MM_MODEL_ROOT / "12/yolo12n-mm-mid-seg.yaml",
    #     "optimizer": "MuSGD",
    # },
    # {
    #     "tag": "yolo12n_mm_mid_seg2",
    #     "model": MM_MODEL_ROOT / "12/yolo12n-mm-mid-seg.yaml",
    #     "optimizer": "MuSGD",
    # },
    # {
    #     "tag": "yolo12n_mm_mid_seg3",
    #     "model": MM_MODEL_ROOT / "12/yolo12n-mm-mid-seg.yaml",
    #     "optimizer": "MuSGD",
    # },
    {
        "tag": "yolo11n-mm-mid-seg",
        "model": MM_MODEL_ROOT / "yolo11n-mm-mid-seg-rgbdepth.yaml",
        "optimizer": "SGD",
    },
    {
        "tag": "yolov8n-mm-mid-seg",
        "model": MM_MODEL_ROOT / "v8/yolov8-mm-mid-seg-rgbdepth.yaml",
        "optimizer": "SGD",
    },
    {
        "tag": "yolov5_mm_mid_seg",
        "model": MM_MODEL_ROOT / "v5/yolov5-mm-mid-seg.yaml",
        "optimizer": "SGD",
    },
    # Add more models here, for example:
    # {
    #     "tag": "yolo26n_mm_mid_seg",
    #     "model": MODEL_ROOT / "yolo26n-mm-mid-seg.yaml",
    #     "optimizer": "MuSGD",
    # },
]
# ================================================================

CHANNEL_TO_XCH = {
    "gray1": 1,
    "gray2": 2,
    "gray3": 3,
    "b": 1,
    "g": 1,
    "r": 1,
    "bg": 2,
    "br": 2,
    "gr": 2,
    "bgr": 3,
}
CHANNELS_BY_SOURCE = {
    "gray": {"gray1", "gray2", "gray3"},
    "color": {"b", "g", "r", "bg", "br", "gr", "bgr"},
}


def parse_cache(value: str) -> bool | str:
    """Parse Ultralytics cache CLI values."""
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "ram"}:
        return True if value != "ram" else "ram"
    if value in {"false", "0", "no", "n", "none"}:
        return False
    if value == "disk":
        return "disk"
    raise argparse.ArgumentTypeError("cache must be one of true/false/ram/disk")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    return data


def resolve_data_path(depth_source: str, data: str | None) -> Path:
    return Path(data).resolve() if data else DEFAULT_DATA[depth_source]


def apply_depth_source(data: dict[str, Any], depth_source: str) -> None:
    """Keep RGB as primary and select the configured depth-like secondary source."""
    modality = data.get("modality")
    if not isinstance(modality, dict):
        return
    if depth_source == "gray" and "depth" in modality:
        data["modality_used"] = ["rgb", "depth"]
    elif depth_source == "color" and "ir" in modality:
        data["modality_used"] = ["rgb", "ir"]


def build_experiment_data_yaml(source_yaml: Path, depth_source: str, channel_select: str) -> Path:
    data = load_yaml(source_yaml)
    xch = CHANNEL_TO_XCH[channel_select]
    apply_depth_source(data, depth_source)
    data["path"] = source_yaml.parent.as_posix()
    data["Xch"] = xch
    data["x_channel_select"] = channel_select

    out_dir = PROJECT_ROOT / "generated_data_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_yaml.stem}_{depth_source}_{channel_select}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return out_path


def sanitize_run_token(value: object) -> str:
    """Convert a model tag or optimizer name into a filesystem-friendly run-name token."""
    token = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value).strip())
    token = token.strip("._-")
    return token or "unnamed"


def resolve_model_path(model: object) -> Path:
    """Resolve model paths from absolute, project-relative, MODEL_ROOT, or MM_MODEL_ROOT-relative values."""
    path = Path(model)
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            Path.cwd() / path,
            WORKSPACE_ROOT / path,
            PROJECT_ROOT / path,
            MODEL_ROOT / path,
            MM_MODEL_ROOT / path,
        ])

    parts = list(path.parts)
    lowered = [p.lower() for p in parts]
    for i, part in enumerate(lowered):
        if part == "mutilmodel" and i + 1 < len(parts) and lowered[i + 1] == "ultralytics":
            candidates.append(WORKSPACE_ROOT / Path(*parts[i:]))
        if part == "ultralytics":
            candidates.append(PROJECT_ROOT / Path(*parts[i:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else path.resolve()


def build_model_queue(args: argparse.Namespace) -> list[dict[str, str | Path]]:
    """Build the ordered model queue, with per-model optimizer settings."""
    if args.model:
        model_path = resolve_model_path(args.model)
        optimizer = str(args.optimizer).strip()
        tag = sanitize_run_token(model_path.stem)
        queue = [{"tag": tag, "model": model_path, "optimizer": optimizer}]
    else:
        if not SCRIPT_MODEL_CONFIGS:
            raise ValueError("SCRIPT_MODEL_CONFIGS must contain at least one model config.")
        queue = []
        for i, cfg in enumerate(SCRIPT_MODEL_CONFIGS, start=1):
            if not isinstance(cfg, dict):
                raise ValueError(f"SCRIPT_MODEL_CONFIGS[{i}] must be a dict.")
            missing = {"tag", "model", "optimizer"} - set(cfg)
            if missing:
                raise ValueError(f"SCRIPT_MODEL_CONFIGS[{i}] missing keys: {', '.join(sorted(missing))}")
            tag = sanitize_run_token(cfg["tag"])
            optimizer = str(cfg["optimizer"]).strip()
            queue.append({"tag": tag, "model": resolve_model_path(cfg["model"]), "optimizer": optimizer})

    seen_tags = set()
    for cfg in queue:
        tag = str(cfg["tag"])
        model_path = Path(cfg["model"])
        optimizer = str(cfg["optimizer"]).strip()
        if tag in seen_tags:
            raise ValueError(f"Duplicate model tag in training queue: {tag}")
        seen_tags.add(tag)
        if not model_path.exists():
            raise FileNotFoundError(f"Model YAML does not exist for tag={tag}: {model_path}")
        if not optimizer:
            raise ValueError(f"Optimizer is empty for model tag={tag}: {model_path}")
    return queue


def build_run_name(args: argparse.Namespace, model_tag: str, optimizer: str, queue_len: int) -> str:
    """Build an output run name that stays unique across queued models."""
    optimizer_token = sanitize_run_token(optimizer)
    if args.name:
        base = sanitize_run_token(args.name)
    else:
        base = f"depth_{args.depth_source}_{args.x_channel_select}"
    return f"{base}__scale-{args.model_scale}__{sanitize_run_token(model_tag)}__{optimizer_token}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOMM segmentation with RGB fixed at 3 channels and depth Xch set to 1/2/3."
    )
    parser.add_argument("--depth-source", choices=("gray", "color"), default=SCRIPT_DEPTH_SOURCE,
                        help="gray uses images_depth; color uses images_ir as the color depth source.")
    parser.add_argument("--x-channel-select", choices=sorted(CHANNEL_TO_XCH), default=SCRIPT_X_CHANNEL_SELECT,
                        help="Depth channel selector. Defaults come from SCRIPT_X_CHANNEL_SELECT.")
    parser.add_argument("--data", default=None, help="Optional source data YAML. Defaults depend on --depth-source.")
    parser.add_argument("--model", default=None,
                        help="Optional single model YAML. If omitted, SCRIPT_MODEL_CONFIGS is trained in order.")
    parser.add_argument("--model-scale", choices=("n", "s", "m", "l", "x"), default=SCRIPT_MODEL_SCALE,
                        help="Model compound scale from the selected YAML's `scales` mapping.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", default="MuSGD", help="Optimizer for --model single-model mode.")
    parser.add_argument("--project", default="ResTest/channel_ablation")
    parser.add_argument("--name", default=None, help="Run name prefix. Scale, model tag, and optimizer are appended.")
    parser.add_argument("--device", default="0", help="Ultralytics device argument, e.g. 0, 0,1, or cpu.")
    parser.add_argument("--cache", type=parse_cache, default=True)
    parser.add_argument("--cuda-visible-devices", default=None, help="Optionally sets CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1.")
    parser.add_argument("--close-mosaic", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.x_channel_select not in CHANNELS_BY_SOURCE[args.depth_source]:
        allowed = ", ".join(sorted(CHANNELS_BY_SOURCE[args.depth_source]))
        raise SystemExit(
            f"--x-channel-select={args.x_channel_select!r} is not valid for "
            f"--depth-source={args.depth_source!r}. Allowed: {allowed}"
        )
    return args


def make_train_kwargs(
    args: argparse.Namespace,
    experiment_yaml: Path,
    run_name: str,
    optimizer: str,
) -> dict[str, Any]:
    train_kwargs: dict[str, Any] = {
        "data": str(experiment_yaml),
        "model_scale": args.model_scale,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "cache": args.cache,
        "amp": args.amp,
        "workers": args.workers,
        "optimizer": optimizer,
        "cos_lr": args.cos_lr,
        "plots": args.plots,
        "exist_ok": args.exist_ok,
        "project": args.project,
        "name": run_name,
    }
    optional_args = {
        "device": args.device,
        "close_mosaic": args.close_mosaic,
        "patience": args.patience,
    }
    train_kwargs.update({k: v for k, v in optional_args.items() if v is not None})
    return train_kwargs


def cleanup_after_model() -> None:
    """Release Python and CUDA caches between sequential model runs."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices)

    from ultralytics import YOLOMM

    source_yaml = resolve_data_path(args.depth_source, args.data)
    experiment_yaml = build_experiment_data_yaml(source_yaml, args.depth_source, args.x_channel_select)
    model_queue = build_model_queue(args)

    print(f"Using source YAML: {source_yaml}")
    print(f"Using experiment YAML: {experiment_yaml}")
    print(f"RGB channels fixed at 3; Xch={CHANNEL_TO_XCH[args.x_channel_select]} ({args.x_channel_select})")
    print(f"Model scale: {args.model_scale}")
    print(f"Training model queue: {len(model_queue)} model(s)")

    for index, model_cfg in enumerate(model_queue, start=1):
        model_tag = str(model_cfg["tag"])
        model_path = Path(model_cfg["model"])
        optimizer = str(model_cfg["optimizer"])
        run_name = build_run_name(args, model_tag, optimizer, len(model_queue))
        train_kwargs = make_train_kwargs(args, experiment_yaml, run_name, optimizer)

        print(f"[{index}/{len(model_queue)}] Model tag: {model_tag}")
        print(f"[{index}/{len(model_queue)}] Model YAML: {model_path}")
        print(f"[{index}/{len(model_queue)}] Model scale: {args.model_scale}")
        print(f"[{index}/{len(model_queue)}] Optimizer: {optimizer}")
        print(f"[{index}/{len(model_queue)}] Run name: {run_name}")

        model = None
        try:
            model = YOLOMM(str(model_path))
            model.train(**train_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Training failed for tag={model_tag}, model={model_path}, optimizer={optimizer}"
            ) from exc
        finally:
            del model
            cleanup_after_model()


if __name__ == "__main__":
    main()
