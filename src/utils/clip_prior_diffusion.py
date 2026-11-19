"""Training-only channel statistics, epsilon DDPM training and unclipped DDIM sampling."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data.scheme1 import digest_json
from src.utils.mapping_precision import mapping_autocast, microbatch_slices


class ClipChannelStats(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean, std = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)
        if (mean.ndim != 3 or mean.shape[:2] != (1,1) or mean.shape != std.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any()):
            raise ValueError('Invalid CLIP channel statistics')
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def normalize(self, value):
        return (value.float()-self.mean.to(value.device))/self.std.to(value.device)

    def denormalize(self, value):
        return self.mean.to(value.device)+self.std.to(value.device)*value.float()

    def fingerprint(self):
        return hashlib.sha256(self.mean.cpu().numpy().tobytes()+self.std.cpu().numpy().tobytes()).hexdigest()


def fit_clip_channel_stats(clip, train_indices, image_ids, source_identity, min_std=1e-6, chunk_size=16):
    indices = np.asarray(train_indices, dtype=np.int64)
    ids = np.asarray(image_ids, dtype=np.int64)
    if (clip.ndim != 3 or len(ids) != len(clip) or indices.ndim != 1 or not len(indices)
            or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(clip))
            or not math.isfinite(min_std) or min_std <= 0 or chunk_size <= 0):
        raise ValueError('Invalid training-only CLIP statistics source')
    total, squared = np.zeros(clip.shape[-1], dtype=np.float64), np.zeros(clip.shape[-1], dtype=np.float64)
    count = 0
    for start in range(0, len(indices), chunk_size):
        values = np.asarray(clip[indices[start:start+chunk_size]], dtype=np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError('Non-finite training CLIP targets')
        total += values.sum(axis=(0,1))
        squared += np.square(values).sum(axis=(0,1))
        count += values.shape[0]*values.shape[1]
    mean = total/count
    std = np.maximum(np.sqrt(np.maximum(squared/count-mean**2, 0)), min_std)
    stats = ClipChannelStats(mean.reshape(1,1,-1), std.reshape(1,1,-1))
    identity = {'fit_split':'internal_train_only', 'source_identity':source_identity,
                'train_image_ids':ids[indices].tolist(), 'train_indices':indices.tolist(),
                'population_count':count, 'min_std':min_std, 'statistics_sha256':stats.fingerprint()}
    identity['identity_sha256'] = digest_json(identity)
    return stats, identity


class DDPMSchedule(nn.Module):
    def __init__(self, num_train_steps=1000, beta_start=1e-4, beta_end=0.02, prediction_type='epsilon'):
        super().__init__()
        if (num_train_steps < 2 or not 0 < beta_start <= beta_end < 1 or prediction_type != 'epsilon'):
            raise ValueError('Expected a finite linear epsilon DDPM schedule')
        self.num_train_steps = num_train_steps
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, num_train_steps, dtype=torch.float32))
        self.register_buffer('alpha_bar', (1-self.betas).cumprod(0))

    def add_noise(self, x0, noise, timesteps):
        if (x0.shape != noise.shape or timesteps.shape != (len(x0),) or timesteps.dtype != torch.long
                or (timesteps < 0).any() or (timesteps >= self.num_train_steps).any()):
            raise ValueError('Invalid noise/timestep batch')
        alpha = self.alpha_bar.to(x0.device)[timesteps].reshape(-1,1,1)
        return alpha.sqrt()*x0+(1-alpha).sqrt()*noise

    def timesteps(self, steps):
        if not 2 <= steps <= self.num_train_steps:
            raise ValueError('DDIM steps must include both endpoints and be between 2 and T')
        return torch.linspace(0, self.num_train_steps-1, steps).round().long().flip(0).tolist()

    def ddim_step(self, xt, epsilon, timestep, previous_timestep):
        if xt.shape != epsilon.shape or not 0 <= timestep < self.num_train_steps:
            raise ValueError('Invalid DDIM state/timestep')
        if previous_timestep is not None and not 0 <= previous_timestep < timestep:
            raise ValueError('DDIM previous timestep must be strictly smaller')
        alpha = self.alpha_bar[timestep].to(xt.device)
        x0 = (xt-(1-alpha).sqrt()*epsilon)/alpha.sqrt()
        if previous_timestep is None:
            return x0
        previous = self.alpha_bar[previous_timestep].to(xt.device)
        return previous.sqrt()*x0+(1-previous).sqrt()*epsilon


def stable_prior_seed(image_id, seed, sample_index=0, namespace='prior_tokens_v1'):
    if sample_index < 0 or seed < 0:
        raise ValueError('Seeds/sample indices must be nonnegative')
    value = f'{namespace}|{image_id}|{seed}|{sample_index}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'little') % (2**63-1)


def initial_prior_noise(mean, image_ids, prior_seed, sample_index):
    if len(image_ids) != len(mean):
        raise ValueError('One stable target identity is required for each mean')
    return torch.stack([torch.randn(tuple(mean.shape[1:]), generator=torch.Generator().manual_seed(
        stable_prior_seed(image_id, prior_seed, sample_index))) for image_id in image_ids]).to(mean.device)


def validation_noise(mean, image_ids, scheduler, seed):
    times, noises = [], []
    for image_id in image_ids:
        generator = torch.Generator().manual_seed(stable_prior_seed(image_id, seed, namespace='prior_validation_v1'))
        times.append(torch.randint(scheduler.num_train_steps, (1,), generator=generator)[0])
        noises.append(torch.randn(tuple(mean.shape[1:]), generator=generator))
    return torch.stack(times).to(mean.device), torch.stack(noises).to(mean.device)


def prior_training_noise(mean, scheduler, generator=None):
    # Draw the whole logical batch before any microbatch split. CPU RNG is checkpointed.
    times = torch.randint(scheduler.num_train_steps, (len(mean),), generator=generator)
    noise = torch.randn(tuple(mean.shape), generator=generator)
    return times.to(mean.device), noise.to(mean.device)


def prior_loss(prior, mean, raw_clip, scheduler, stats, timesteps, noise, mixed_precision='no'):
    with torch.autocast(device_type=mean.device.type, enabled=False):
        xt = scheduler.add_noise(stats.normalize(raw_clip.detach()), noise.float(), timesteps)
    with mapping_autocast(mean.device, mixed_precision):
        condition = prior.encode_condition(mean.detach())
        prediction = prior.predict_noise(xt, timesteps, condition)
    with torch.autocast(device_type=mean.device.type, enabled=False):
        loss = F.mse_loss(prediction.float(), noise.float())
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite prior denoising loss')
    return loss


def backward_prior_batch(prior, mean, target, scheduler, stats, scaler, mixed_precision='no', micro_batch_size=0):
    times, noise = prior_training_noise(mean, scheduler)
    total = mean.new_zeros((), dtype=torch.float32)
    for part in microbatch_slices(len(mean), micro_batch_size):
        loss = prior_loss(prior, mean[part], target[part], scheduler, stats, times[part], noise[part], mixed_precision)
        weighted = loss*(len(mean[part])/len(mean))
        scaler.scale(weighted).backward()
        total += weighted.detach()
    return {'total':total, 'denoising_mse':total}


@torch.inference_mode()
def sample_clip_tokens(prior, mean, scheduler, stats, *, image_ids, prior_seed, sample_index=0,
                       steps=50, mixed_precision='no'):
    """Target-free inference. Only mu, identity-derived noise and trained parameters enter."""
    if prior.training:
        raise ValueError('Prior sampling requires eval mode')
    sequence = scheduler.timesteps(steps)
    value = initial_prior_noise(mean, image_ids, prior_seed, sample_index)
    with mapping_autocast(mean.device, mixed_precision):
        condition = prior.encode_condition(mean)
    for number, timestep in enumerate(sequence):
        t = torch.full((len(mean),), timestep, device=mean.device, dtype=torch.long)
        with mapping_autocast(mean.device, mixed_precision):
            epsilon = prior.predict_noise(value, t, condition).float()
        value = scheduler.ddim_step(value.float(), epsilon, timestep,
                                   sequence[number+1] if number+1 < len(sequence) else None)
    raw = stats.denormalize(value)
    if raw.shape != mean.shape or not torch.isfinite(raw).all():
        raise FloatingPointError('Invalid sampled raw CLIP tokens')
    return raw
