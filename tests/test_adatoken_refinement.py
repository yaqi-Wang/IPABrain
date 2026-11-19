"""Warm-start parity, frozen adapters, full-batch contrastive gradients and pipeline."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from test_adatoken import tiny_adatoken, nonzero_sit, adatoken_fixture
from src.models.clip_mapping import build_clip_mapper
from src.utils.mapping_objective import backward_mapping_objective, generation_contrastive_losses
from src.utils.mapping_precision import mapping_scaler


def refined(**options):
    return build_clip_mapper(dict(type='adatoken_refinement', latent_tokens=4, embed_dim=6,
                                  depth=2, num_heads=2, mlp_ratio=2.,
                                  gradient_checkpointing=True, **options))


class RefinementTests(unittest.TestCase):
    def test_zero_condition_preserves_pretrained_function_and_learns(self):
        original = tiny_adatoken(True)
        nonzero_sit(original)
        model = refined(sample_condition=True, train_backbone=False)
        model.sit.load_state_dict(original.sit.state_dict())
        x = torch.randn(3,4,6)
        torch.testing.assert_close(model(x), original(x), rtol=0, atol=0)
        model(x).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.sit.parameters()))
        self.assertGreater(model.condition_adapter[-1].weight.grad.abs().sum().item(), 0)
        self.assertFalse(model.sit.pos_embed.requires_grad)

    def test_fusion_initialization_bounds_and_serialization(self):
        x = torch.randn(3,4,6)
        model = refined(fusion='learned', alpha_init=0.5, train_backbone=False)
        nonzero_sit(model)
        raw = model.sit(x, x.new_zeros(len(x)))
        torch.testing.assert_close(model(x), (raw+x)/2)
        model(x).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.sit.parameters()))
        self.assertGreater(model.fusion_gate[-1].weight.grad.abs().sum().item(), 0)
        clone = refined(fusion='learned', train_backbone=False)
        clone.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(model(x), clone(x))
        for alpha in (0.,1.):
            fixed = refined(fusion='fixed', alpha_init=alpha, train_backbone=False)
            fixed.sit.load_state_dict(model.sit.state_dict())
            torch.testing.assert_close(fixed(x), x if alpha == 0 else raw)

    def test_contrastive_microbatch_matches_full_batch_loss_and_gradients(self):
        torch.manual_seed(123)
        model = refined()
        nonzero_sit(model)
        replay = copy.deepcopy(model)
        mean, target = torch.randn(5,4,6), torch.randn(5,4,6)
        objective = {'type':'generation_contrastive','weight':0.01,'temperature':0.125}
        expected = generation_contrastive_losses(model(mean), target, objective)
        expected['total'].backward()
        actual = backward_mapping_objective(replay, mean, target,
                    mapping_scaler(torch.device('cpu'),'no'), objective, 'no', 2)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        for (name,left), right in zip(model.named_parameters(), replay.parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=5e-4, atol=2e-6, msg=name)
        with self.assertRaises(ValueError):
            generation_contrastive_losses(mean[:1], target[:1], objective)

    def test_train_resume_encode_and_fixed_export(self):
        from run_adatoken_refinement import export_fixed
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            fixture, original, original_path = adatoken_fixture(root)
            fixture.run('train_brainvae_scheme1.py', config=fixture.source_config)
            fixture.run('train_clip_mapping.py', config=original_path)
            init=root/'outputs/subj01/runs/adatoken/checkpoints/best.pt'
            # Test all three trainable changes, including contrastive gate training.
            for name,options,contrast in [('condition',{'sample_condition':True},False),
                                         ('gate',{'fusion':'learned'},False),
                                         ('contrastive',{'train_backbone':True},True)]:
                cfg=copy.deepcopy(original)
                cfg['run_name']='refine_'+name
                cfg['model'].update(type='adatoken_refinement',train_backbone=False)
                cfg['model'].update(options)
                cfg['training']['initialize_from']=str(init)
                if contrast:
                    cfg['objective']={'type':'generation_contrastive','weight':0.01,'temperature':0.125}
                path=root/f'{name}.yaml'
                path.write_text(yaml.safe_dump(cfg))
                fixture.run('train_clip_mapping.py',config=path)
                run=root/'outputs/subj01/runs'/cfg['run_name']
                saved=torch.load(run/'checkpoints/last.pt',weights_only=False,map_location='cpu')
                fixture.run('train_clip_mapping.py','--resume',config=path)
                resumed=torch.load(run/'checkpoints/last.pt',weights_only=False,map_location='cpu')
                self.assertEqual(saved['training_state']['history'],resumed['training_state']['history'])
                fixture.run('decode_clip_mapping.py','--dataset','nsd_val','--encode-only',config=path)
                self.assertTrue((run/'evaluation/mapped_mean_fmri/nsd_val/metadata.json').exists())
            cfg=copy.deepcopy(original)
            cfg['run_name']='fixed_export'
            cfg['model'].update(type='adatoken_refinement',train_backbone=False,fusion='fixed',alpha_init=0.5)
            cfg['training']['initialize_from']=str(init)
            path=root/'fixed.yaml'; path.write_text(yaml.safe_dump(cfg))
            run=root/'outputs/subj01/runs/fixed_export'
            export_fixed(cfg,run)
            (root/'nsd/subj01/nsd_sdxl_clip_test_sub1.npy').unlink()
            fixture.run('decode_clip_mapping.py','--dataset','nsd_test','--encode-only',config=path)
            self.assertTrue(json.loads((run/'training_complete.json').read_text())['export_only'])


if __name__ == '__main__':
    unittest.main()
