"""Explicit train/reconstruct/evaluate entry point for isolated AdaToken refinements."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import torch

from _bootstrap import add_project_root
ROOT = add_project_root()
from src.data.scheme1 import atomic_torch_save, resolve_path
from src.utils.clip_mapping import atomic_json, load_mapping_config, mapping_lock
from src.utils.clip_head_tasks import MappingTrainingTask
from src.utils.scheme1_runs import training_directory

VARIANTS = ('baseline', 'control', 'condition', 'gate', 'contrastive', 'gate_contrastive',
            'fixed025', 'fixed050', 'fixed075')


def export_fixed(config, run):
    """Export a self-contained frozen mapper without fitting or selecting on test."""
    path = resolve_path(config['training']['initialize_from'], ROOT)
    source = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    task = MappingTrainingTask(config)
    task.prepare(None, source['split'], source['protocol_metadata']['source_identity'], torch.device('cpu'))
    checkpoint_dir = run / 'checkpoints'
    if (checkpoint_dir/'best.pt').exists():
        raise FileExistsError(f'Export already exists: {run}')
    model = task.build()
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Export-only model must have no trainable parameters')
    protocol = copy.deepcopy(source['protocol_metadata'])
    protocol.update(config=config, run_name=run.name, training_identity=task.identity(0, 0),
                    export_only=True, loss_definition='No fitting: frozen original AdaToken and preset fusion',
                    **task.protocol_fields())
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({'model': model.state_dict(), 'config': config['model'],
                      'protocol_metadata': protocol, 'split': source['split'],
                      'preprocessing': source['preprocessing'],
                      'training_mean_clip': source['training_mean_clip'],
                      'checkpoint_kind': 'fixed_inference_export'}, checkpoint_dir/'best.pt')
    atomic_json(protocol, run/'protocol.json')
    atomic_json({'status':'complete', 'export_only':True, **task.protocol_fields()}, run/'training_complete.json')


def commands(args, config_path, config, run):
    dataset = 'nsd_val' if args.stage == 'validate' else 'nsd_test'
    count = args.max_items or (50 if dataset == 'nsd_val' else 250)
    tag = f'{dataset}_{count}_s1'
    common = ['--config', str(config_path), '--subject', str(args.subject), '--device', args.device]
    generate = [sys.executable, str(ROOT/'scripts/decode_clip_mapping.py'), *common,
                '--dataset', dataset, '--max-items', str(count), '--num-samples', '1',
                '--num-steps', '38', '--seed', '42', '--rgb-mode', 'standard',
                '--eval-tag', tag, '--token-source-tag', tag, '--resume']
    evaluate = [sys.executable, str(ROOT/'scripts/eval_scheme1_reconstruction.py'),
                '--dataset-dir', str(run/'evaluation'/tag/dataset),
                '--models', 'Alex2,Alex5,Incep,CLIP,Eff,SwAV', '--device', args.device]
    train = [sys.executable, str(ROOT/'scripts/train_clip_mapping.py'), *common]
    if (run/'checkpoints/last.pt').exists():
        train.append('--resume')
    return train, generate, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--stage', choices=('train','generate','evaluate','validate','all','check'), default='check')
    parser.add_argument('--subject', type=int, choices=(1,), default=1,
                        help='Presets refer to the existing subj01 source and cache')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-items', type=int, default=0, help='Default: validation 50, test 250')
    parser.add_argument('--print-only', action='store_true')
    args = parser.parse_args()
    if args.max_items < 0:
        parser.error('max-items must be nonnegative')
    path = ROOT/f'configs/clip_mapping_adatoken_refine_{args.variant}.yaml'
    config, _, _ = load_mapping_config(path, ROOT)
    run = training_directory(config, ROOT, args.subject)
    train, generate, evaluate = commands(args, path, config, run)
    export = args.variant == 'baseline' or args.variant.startswith('fixed')
    def execute(command):
        print(json.dumps(command, ensure_ascii=False), flush=True)
        if not args.print_only:
            subprocess.run(command, cwd=ROOT, check=True)
    if args.stage == 'check':
        if export:
            print('Frozen export preset; no optimization required.')
        else:
            execute([x for x in train if x != '--resume'] + ['--dry-run'])
    if args.stage in ('train', 'all'):
        if export:
            print(f'Export frozen preset to {run}', flush=True)
            if not args.print_only:
                with mapping_lock(run):
                    export_fixed(config, run)
        else:
            execute(train)
    if args.stage in ('generate','validate','all'):
        execute(generate)
    if args.stage in ('evaluate','validate','all'):
        execute(evaluate)


if __name__ == '__main__':
    main()
