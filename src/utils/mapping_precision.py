"""AMP and microbatch operations shared by mapping training and inference."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def validate_mapping_precision(device: torch.device, mixed_precision: str) -> None:
    if mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError("Mapping mixed_precision must be no, fp16 or bf16")
    if mixed_precision != "no" and device.type != "cuda":
        raise ValueError("Mixed precision requires CUDA; use --mixed-precision no for a CPU smoke test")
    if mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 requires a CUDA device with BF16 support")


def mapping_autocast(device: torch.device, mixed_precision: str):
    validate_mapping_precision(device, mixed_precision)
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=mixed_precision != "no")


def mapping_scaler(device: torch.device, mixed_precision: str):
    validate_mapping_precision(device, mixed_precision)
    return torch.amp.GradScaler("cuda", enabled=mixed_precision == "fp16")


def microbatch_slices(size: int, micro_batch_size: int):
    if size <= 0 or micro_batch_size < 0:
        raise ValueError("Invalid mapping batch / microbatch size")
    width = micro_batch_size or size
    for start in range(0, size, width):
        yield slice(start, min(start + width, size))


def predict_mapping(model, mean: torch.Tensor, mixed_precision: str = "no",
                    micro_batch_size: int = 0) -> torch.Tensor:
    predictions = []
    for part in microbatch_slices(len(mean), micro_batch_size):
        with mapping_autocast(mean.device, mixed_precision):
            prediction = model(mean[part])
        predictions.append(prediction.float())
    return predictions[0] if len(predictions) == 1 else torch.cat(predictions)


def backward_mapping_batch(model, mean: torch.Tensor, target: torch.Tensor, scaler,
                           mixed_precision: str = "no", micro_batch_size: int = 0) -> torch.Tensor:
    """Accumulate one logical batch's sample-weighted MSE before an optimizer step.

    Parameters and Adam moments stay FP32. Autocast covers the head forward;
    raw-coordinate MSE is explicitly computed in FP32, including under AMP.
    A short final microbatch receives its actual fraction of the logical batch.
    """
    if mean.shape != target.shape:
        raise ValueError("Mapping mean/target batch shapes differ")
    total = mean.new_zeros((), dtype=torch.float32)
    for part in microbatch_slices(len(mean), micro_batch_size):
        with mapping_autocast(mean.device, mixed_precision):
            prediction = model(mean[part])
        loss = F.mse_loss(prediction.float(), target[part].float(), reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite mapping microbatch loss")
        weight = len(mean[part]) / len(mean)
        weighted = loss if weight == 1 else loss * weight
        scaler.scale(weighted).backward()
        total += weighted.detach()
    return total
