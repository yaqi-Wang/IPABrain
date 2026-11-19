"""Head-specific losses/state plugged into the shared frozen-source trainer."""
from __future__ import annotations

import math
from pathlib import Path

import torch

from src.data.scheme1 import digest_json, file_sha256, resolve_path
from src.models.clip_mapping import build_clip_mapper
from src.models.clip_diffusion_prior import build_clip_prior
from src.utils.clip_mapping import MAPPING_PROTOCOL, evaluate_mapper, load_head_source_config
from src.utils.mapping_objective import backward_mapping_objective, mapping_objective
from src.utils.mapping_precision import microbatch_slices
from src.utils.clip_prior_diffusion import (
    ClipChannelStats, DDPMSchedule, backward_prior_batch, fit_clip_channel_stats,
    prior_loss, validation_noise,
)

PRIOR_PROTOCOL = 'frozen_brain_conditional_clip_diffusion_v1'


def head_training_identity(config, max_steps, max_val_steps):
    identity = {'model':config['model'], 'training':{key:value for key,value in config['training'].items()
                if key not in {'checkpoint_every_steps','encode_batch_size'}},
                'validation':config['validation'], 'debug_steps':[max_steps,max_val_steps]}
    # Old configs retain their exact legacy identity; explicit new objectives are protected.
    if 'objective' in config:
        identity['objective'] = config['objective']
    return identity


class MappingTrainingTask:
    protocol = MAPPING_PROTOCOL
    stage = 'mapping'
    def __init__(self, config):
        self.config, self.objective = config, mapping_objective(config)
        self.joint = self.objective['type'] == 'generation_semantic'
        self.contrastive = self.objective['type'] == 'generation_contrastive'
        self.drop_last = self.joint or self.contrastive or config['training'].get('drop_last', False)
        self.loss_definition = ('MSE(G(H), C) + lambda * SoftCLIP(S(H), C); shared H, frozen mu'
                                if self.joint else 'MSE(g_phi(stopgrad(posterior_mean)), raw_CLIP_tokens), mean over all coordinates')
        if self.contrastive:
            self.loss_definition = 'raw token MSE + weight * symmetric InfoNCE on flattened generation tokens; full logical batch negatives'
        self.initialization = None

    def prepare(self, clip, split, source_identity, device):
        path = self.config['training'].get('initialize_from')
        if not path:
            return
        if self.config['model']['type'] != 'adatoken_refinement':
            raise ValueError('initialize_from is only supported for adatoken_refinement')
        path = resolve_path(path, Path(__file__).resolve().parents[2])
        if not (path.parent.parent / 'training_complete.json').is_file():
            raise ValueError('Initialization AdaToken training is not complete')
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        protocol = saved['protocol_metadata']
        refined_source = (saved['config']['type'] == 'adatoken_refinement'
                          and self.config['training'].get('allow_refined_initialization', False))
        if (saved['config']['type'] != 'adatoken' and not refined_source) or protocol['source_identity'] != source_identity:
            raise ValueError('Initialization must be original AdaToken with the same frozen BrainVAE/data')
        if refined_source and (saved['config'].get('sample_condition', False)
                               or saved['config'].get('fusion', 'none') != 'none'):
            raise ValueError('Refined initialization must be a plain SiT without condition/fusion branches')
        if protocol.get('debug_training') or protocol.get('test_used_for_selection'):
            raise ValueError('Initialization must be a formal internally selected checkpoint')
        for key, value in split.items():
            if key not in saved['split'] or not torch.equal(torch.as_tensor(value), saved['split'][key]):
                raise ValueError('Initialization train/validation/test split mismatch')
        for key, value in saved['config'].items():
            if refined_source and key in {'train_backbone', 'sample_condition', 'fusion',
                                          'alpha_init', 'gate_width', 'condition_width'}:
                continue
            if key != 'type' and self.config['model'].get(key) != value:
                raise ValueError(f'Initialization AdaToken backbone option differs: {key}')
        self.initialization = {'path': str(path), 'sha256': file_sha256(path)}
        if refined_source:
            self.initialization['source_objective'] = protocol.get('objective', {'type': 'mse'})

    def build(self):
        model = build_clip_mapper(self.config['model'])
        if self.initialization:
            saved = torch.load(self.initialization['path'], map_location='cpu', weights_only=False, mmap=True)
            # Load only the original SiT, strictly. New branches retain zero initialization.
            if any(not k.startswith('sit.') for k in saved['model']):
                raise ValueError('Original AdaToken checkpoint contains unexpected branches')
            model.sit.load_state_dict({k[4:]: v for k, v in saved['model'].items()}, strict=True)
        return model

    def identity(self, max_steps, max_val_steps):
        result = head_training_identity(self.config, max_steps, max_val_steps)
        if 'objective' in result:
            result['objective'] = self.objective
        if self.initialization:
            result['initialization'] = self.initialization
        return result

    def backward(self, model, mean, target, batch, scaler, precision, micro):
        return backward_mapping_objective(model, mean, target, scaler, self.objective, precision, micro)

    def evaluate(self, model, loader, device, control, max_steps, precision, micro):
        return evaluate_mapper(model, loader, device, control, max_steps, precision, micro,
                               objective=self.objective if self.joint or self.contrastive else None)

    def score(self, validation):
        return validation['mapped']['token_mse']

    def checkpoint_fields(self):
        return {}

    def restore(self, saved):
        pass

    def protocol_fields(self):
        result = {'objective':self.objective} if self.joint or 'objective' in self.config else {}
        if self.initialization:
            result['initialization'] = self.initialization
        return result

    def epoch_fields(self, components, validation):
        result = {'train_token_mse':components['gen_mse']}
        if self.contrastive:
            result.update({'train_contrastive_loss': components['contrastive_loss'],
                           'val_contrastive_loss': validation['contrastive_loss'],
                           'train_total': components['total']})
        if self.joint:
            result.update({'train_gen_mse':components['gen_mse'], 'train_sem_loss':components['sem_loss'],
                           'val_gen_mse':validation['gen_mse'], 'val_sem_loss':validation['sem_loss'],
                           'semantic_weight':self.objective['semantic_weight'], 'temperature':self.objective['temperature']})
        return result


def validate_prior_config(config):
    if config['model']['type'] != 'clip_conditional_prior' or config.get('objective') != {'type':'epsilon'}:
        raise ValueError('Independent prior requires type clip_conditional_prior and objective epsilon')
    diffusion = dict(config['diffusion'])
    if diffusion.pop('schedule', None) != 'linear':
        raise ValueError('First prior implementation requires a linear DDPM schedule')
    scheduler = DDPMSchedule(**diffusion)
    normalization = config['normalization']
    if set(normalization) != {'min_std'} or not math.isfinite(float(normalization['min_std'])) or float(normalization['min_std']) <= 0:
        raise ValueError('Prior normalization accepts a positive min_std only; fitting is always training-only')
    sampling = config['sampling']
    if set(sampling) != {'steps','prior_seed','num_prior_samples','eta'} or sampling['eta'] != 0:
        raise ValueError('Prior sampling requires explicit eta=0 DDIM settings')
    scheduler.timesteps(int(sampling['steps']))
    if int(sampling['prior_seed']) < 0 or int(sampling['num_prior_samples']) < 1:
        raise ValueError('Invalid prior sampling seed/count')
    if config['validation']['selection_metric'] != 'denoising_mse' or int(config['validation']['noise_seed']) < 0:
        raise ValueError('Prior monitor uses fixed-ID-noise internal denoising_mse')
    candidates = config['validation'].get('candidate_epochs', [])
    if (len(set(candidates)) != len(candidates)
            or any(not isinstance(epoch, int) or epoch < 1 or epoch > int(config['training']['epochs']) for epoch in candidates)):
        raise ValueError('Predeclare unique candidate epochs inside the training budget')
    # Check geometry/keyword options without allocating the production model.
    with torch.device('meta'):
        build_clip_prior(config['model'])


def load_prior_config(path, root):
    config, brain_config, paths = load_head_source_config(path, root)
    validate_prior_config(config)
    return config, brain_config, paths


def prior_checkpoint_components(checkpoint, config, source_identity, device):
    saved = checkpoint['prior_statistics']
    identity = saved['identity']
    signature = dict(identity)
    recorded_hash = signature.pop('identity_sha256')
    stats = ClipChannelStats(saved['mean'], saved['std']).to(device)
    if (identity['source_identity'] != source_identity or identity['fit_split'] != 'internal_train_only'
            or identity['statistics_sha256'] != stats.fingerprint() or digest_json(signature) != recorded_hash
            or stats.mean.shape[-1] != config['model']['embed_dim']):
        raise ValueError('Prior training-statistics identity mismatch')
    training_ids = checkpoint['split']['source_image_ids'][checkpoint['split']['train_indices']].tolist()
    if training_ids != identity['train_image_ids']:
        raise ValueError('Prior statistics were not fitted to the checkpoint training split')
    options = dict(config['diffusion'])
    options.pop('schedule')
    scheduler = DDPMSchedule(**options).to(device)
    if checkpoint.get('prediction_type') != 'epsilon' or any(
            not torch.equal(value.cpu(), checkpoint['noise_schedule'][key].cpu()) for key,value in scheduler.state_dict().items()):
        raise ValueError('Prior checkpoint noise schedule mismatch')
    return scheduler, stats, identity


class PriorTrainingTask:
    protocol, stage = PRIOR_PROTOCOL, 'prior'
    drop_last = False
    loss_definition = 'MSE(epsilon_theta(noisy standardized raw_CLIP, integer t, frozen mu), epsilon)'
    def __init__(self, config):
        validate_prior_config(config)
        self.config = config
        options = dict(config['diffusion'])
        options.pop('schedule')
        self.scheduler = DDPMSchedule(**options)

    def prepare(self, clip, split, source_identity, device):
        self.stats, self.stats_identity = fit_clip_channel_stats(
            clip, split['train_indices'], split['source_image_ids'], source_identity,
            min_std=float(self.config['normalization']['min_std']))
        self.stats.to(device)
        self.scheduler.to(device)

    def build(self):
        return build_clip_prior(self.config['model'])

    def identity(self, max_steps, max_val_steps):
        return {**head_training_identity(self.config, max_steps, max_val_steps),
                'diffusion':self.config['diffusion'], 'normalization':self.config['normalization'],
                'statistics_identity':self.stats_identity}

    def backward(self, model, mean, target, batch, scaler, precision, micro):
        return backward_prior_batch(model, mean, target, self.scheduler, self.stats, scaler, precision, micro)

    @torch.no_grad()
    def evaluate(self, model, loader, device, control, max_steps, precision, micro):
        model.eval()
        total, count = 0.0, 0
        for step, batch in enumerate(loader, 1):
            mean, target = batch['mean'].to(device, non_blocking=True), batch['clip'].to(device, non_blocking=True)
            times, noise = validation_noise(mean, batch['image_id'].tolist(), self.scheduler,
                                            int(self.config['validation']['noise_seed']))
            for part in microbatch_slices(len(mean), micro):
                loss = prior_loss(model, mean[part], target[part], self.scheduler, self.stats,
                                  times[part], noise[part], precision)
                total += float(loss)*len(mean[part])
                count += len(mean[part])
            if max_steps and step >= max_steps:
                break
        if not count:
            raise ValueError('Empty prior validation loader')
        return {'n_items':count, 'denoising_mse':total/count,
                'validation_noise_rule':'stable_image_id_seed', 'free_sampling_evaluation':False}

    def score(self, validation):
        return validation['denoising_mse']

    def checkpoint_fields(self):
        return {'prediction_type':'epsilon', 'noise_schedule':self.scheduler.state_dict(),
                'prior_statistics':{'mean':self.stats.mean, 'std':self.stats.std, 'identity':self.stats_identity}}

    def restore(self, saved):
        if saved['prior_statistics']['identity'] != self.stats_identity:
            raise ValueError('Prior resume training-statistics mismatch')
        prior_checkpoint_components(saved, self.config, self.stats_identity['source_identity'], self.stats.mean.device)

    def protocol_fields(self):
        return {'objective':{'type':'epsilon'}, 'prediction_type':'epsilon', 'diffusion':self.config['diffusion'],
                'statistics_identity':self.stats_identity,
                'checkpoint_selection_role':'denoising_monitor_only; final selection requires nsd_val free generation'}

    def epoch_fields(self, components, validation):
        return {'train_denoising_mse':components['denoising_mse'], 'val_denoising_mse':validation['denoising_mse']}
