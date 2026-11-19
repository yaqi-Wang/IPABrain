"""Simple, stable scheme 1 output and log locations."""
from __future__ import annotations

from pathlib import Path

from src.data.scheme1 import resolve_path, validate_tag


def training_identity(config: dict, max_steps: int = 0, max_val_steps: int = 0) -> dict:
    identity = {key: config[key] for key in ("model", "training", "validation", "preprocessing", "data_tag")}
    # Saving frequency is operational, not a learning parameter. Changing it
    # must not move a run or invalidate an otherwise compatible resume.
    identity["training"] = {key: value for key, value in config["training"].items() if key != "checkpoint_every_steps"}
    identity["debug_steps"] = [max_steps, max_val_steps]
    return identity


def training_name(config: dict, max_steps: int = 0, max_val_steps: int = 0) -> str:
    name = str(config.get("run_name", "brainvae"))
    validate_tag(name)
    return name + "_debug" if max_steps or max_val_steps else name


def training_directory(config: dict, root: Path, subject: int, **debug) -> Path:
    return resolve_path(config["output_root"], root) / f"subj{subject:02d}" / "runs" / training_name(config, **debug)


def evaluation_name(config: dict, root: Path, **overrides) -> str:
    name = str(overrides.get("aggregation", config["evaluation"].get("aggregation", "mean_fmri")))
    if overrides.get("max_items"):
        name += "_partial"
    if overrides.get("include_conceptual"):
        name += "_conceptual"
    return name


def experiment_layout(config: dict, root: Path, subject: int) -> dict[str, Path]:
    run = training_directory(config, root, subject)
    eval_name = evaluation_name(config, root)
    logs = resolve_path(config.get("log_root", "logs/scheme1_direct"), root) / f"subj{subject:02d}"
    return {"run_dir": run, "checkpoint": run / "checkpoints/best.pt",
            "evaluation_dir": run / "evaluation" / eval_name,
            "generation_dir": run / "evaluation" / config["evaluation"].get("generation_tag", eval_name),
            "log_dir": logs, "evaluation_log_dir": logs}
