"""Full-batch generation/semantic training; keep the legacy MSE path intact."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.losses.semantic_alignment import soft_clip_loss
from src.utils.mapping_precision import backward_mapping_batch, mapping_autocast


def mapping_objective(config: dict) -> dict:
    objective = dict(config.get("objective", {"type": "mse"}))
    kind = objective.get("type")
    if kind == "mse":
        if set(objective) != {"type"}:
            raise ValueError("MSE objective accepts only type")
        return objective
    if kind == "generation_contrastive":
        if config['model']['type'] != 'adatoken_refinement':
            raise ValueError('generation_contrastive requires adatoken_refinement')
        if set(objective) - {'type', 'weight', 'temperature'}:
            raise ValueError('Unknown generation_contrastive option')
        weight, temperature = float(objective.get('weight', 0.01)), float(objective.get('temperature', 0.125))
        if not math.isfinite(weight) or weight < 0 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Invalid contrastive weight/temperature')
        if min(config['training']['batch_size'], config['validation']['batch_size']) < 2:
            raise ValueError('Contrastive loss needs at least two samples')
        return {'type': kind, 'weight': weight, 'temperature': temperature}
    if kind != "generation_semantic":
        raise ValueError("Mapping objective must be mse or generation_semantic")
    if set(objective) - {"type", "semantic_weight", "temperature"}:
        raise ValueError("Unknown mapping objective option")
    weight = float(objective.get("semantic_weight", 0.01))
    temperature = float(objective.get("temperature", 0.125))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("semantic_weight must be finite and nonnegative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if config["model"]["type"] != "tokenformer_dual":
        raise ValueError("generation_semantic requires tokenformer_dual")
    if int(config["training"].get("micro_batch_size", 0)) != 0:
        raise ValueError("SoftCLIP requires a complete batch; micro_batch_size must be 0")
    if min(int(config["training"]["batch_size"]), int(config["validation"]["batch_size"])) < 2:
        raise ValueError("SoftCLIP batch sizes must be at least 2")
    if config["training"].get("drop_last", True) is not True:
        raise ValueError("Joint training requires drop_last=True")
    return {"type": kind, "semantic_weight": weight, "temperature": temperature}


def generation_semantic_losses(outputs: dict, target: torch.Tensor, objective: dict) -> dict:
    with torch.autocast(device_type=target.device.type, enabled=False):
        target = target.detach().float()
        generation, semantic = outputs["generation"].float(), outputs["semantic"].float()
        if generation.shape != target.shape or semantic.shape != target.shape:
            raise ValueError("Dual-head outputs and CLIP targets must have identical shapes")
        gen = F.mse_loss(generation, target)
        sem = soft_clip_loss(semantic, target, temp=objective["temperature"])
        total = gen + objective["semantic_weight"] * sem
    if not all(torch.isfinite(value) for value in (total, gen, sem)):
        raise FloatingPointError("Non-finite generation/semantic loss")
    return {"total": total, "gen_mse": gen, "sem_loss": sem}


def backward_mapping_objective(model, mean, target, scaler, objective,
                               mixed_precision="no", micro_batch_size=0) -> dict:
    if objective["type"] == "mse":
        loss = backward_mapping_batch(model, mean, target, scaler, mixed_precision, micro_batch_size)
        return {"total": loss, "gen_mse": loss}
    if objective['type'] == 'generation_contrastive':
        # Cache the gradient on the full logical batch, then replay deterministic
        # AdaToken microbatches. Negatives span all B samples, not each microbatch.
        from src.utils.mapping_precision import predict_mapping, microbatch_slices
        with torch.no_grad():
            prediction = predict_mapping(model, mean.detach(), mixed_precision, micro_batch_size)
        prediction = prediction.detach().requires_grad_(True)
        losses = generation_contrastive_losses(prediction, target, objective)
        gradient, = torch.autograd.grad(losses['total'], prediction)
        for part in microbatch_slices(len(mean), micro_batch_size):
            with mapping_autocast(mean.device, mixed_precision):
                replay = model(mean[part].detach())
            scaler.scale((replay.float() * gradient[part]).sum()).backward()
        return {key: value.detach() for key, value in losses.items()}
    if micro_batch_size or len(mean) < 2:
        raise ValueError("SoftCLIP requires an unsplit batch with at least 2 samples")
    with mapping_autocast(mean.device, mixed_precision):
        outputs = model.forward_train(mean.detach())
    losses = generation_semantic_losses(outputs, target, objective)
    scaler.scale(losses["total"]).backward()
    return {key: value.detach() for key, value in losses.items()}


def generation_contrastive_losses(prediction, target, objective):
    if len(prediction) < 2 or prediction.shape != target.shape:
        raise ValueError('Contrastive loss needs matching tensors and >=2 samples')
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction, target = prediction.float(), target.detach().float()
        mse = F.mse_loss(prediction, target)
        pred_features = F.normalize(prediction.flatten(1), dim=1)
        true_features = F.normalize(target.flatten(1), dim=1)
        logits = pred_features @ true_features.T / objective['temperature']
        labels = torch.arange(len(prediction), device=prediction.device)
        contrastive = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        total = mse + objective['weight'] * contrastive
    if not torch.isfinite(total):
        raise FloatingPointError('Non-finite contrastive mapping loss')
    return {'total': total, 'gen_mse': mse, 'contrastive_loss': contrastive}
