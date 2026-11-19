"""SiT parity, microbatch weighting, cache reuse and resumable mapping training."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.models.clip_mapping import AdaToken, build_clip_mapper
from src.utils.clip_mapping import load_mapping_config
from src.utils.mapping_precision import backward_mapping_batch, mapping_scaler, predict_mapping
from src.data.scheme1 import file_sha256
from test_clip_mapping import MappingFixture


def tiny_adatoken(checkpointing=True):
    return AdaToken(latent_tokens=4, embed_dim=6, depth=2, num_heads=2,
                    mlp_ratio=2.0, gradient_checkpointing=checkpointing)


def nonzero_sit(model):
    # The direct head begins at zero. Exercise all blocks beyond that first step.
    with torch.no_grad():
        for block in model.sit.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=0.02)
            block.adaLN_modulation[-1].bias.normal_(std=0.02)
        model.sit.final_layer.linear.weight.normal_(std=0.02)
        model.sit.final_layer.adaLN_modulation[-1].weight.normal_(std=0.02)


def adatoken_fixture(root):
    fixture = MappingFixture(root, "tokenformer")
    config = copy.deepcopy(fixture.mapping)
    config["run_name"] = "adatoken"
    config["mean_cache_run"] = "tokenformer"
    config["model"] = {"type": "adatoken", "latent_tokens": 4, "embed_dim": 6,
                       "depth": 2, "num_heads": 2, "mlp_ratio": 2.0,
                       "gradient_checkpointing": True, "fused_attn": True, "qk_norm": False}
    config["training"].update({"mixed_precision": "no", "micro_batch_size": 1})
    path = root / "adatoken.yaml"
    path.write_text(yaml.safe_dump(config))
    return fixture, config, path


class TestAdaToken(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_production_config_has_original_sit_size_and_fp16(self):
        config, _, _ = load_mapping_config("configs/clip_mapping_adatoken.yaml", ROOT)
        self.assertEqual(config["brainvae_config"], "configs/brainvae_scheme1_mean_nomse.yaml")
        self.assertEqual(config["mean_cache_run"], "tokenformer")
        self.assertEqual((config["model"]["embed_dim"], config["model"]["depth"],
                          config["model"]["num_heads"]), (1664, 8, 13))
        self.assertEqual(config["training"]["mixed_precision"], "fp16")
        self.assertEqual((config["training"]["batch_size"], config["training"]["micro_batch_size"]), (12, 3))
        with torch.device("meta"):
            model = build_clip_mapper(config["model"])
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 410430592)

    def test_adaln_zero_direct_prediction_and_timestep_zero(self):
        model = tiny_adatoken(False)
        observed = []
        hook = model.sit.t_embedder.register_forward_pre_hook(lambda _, inputs: observed.append(inputs[0]))
        mean = torch.randn(2, 4, 6)
        result = model(mean)
        hook.remove()
        self.assertTrue(torch.equal(result, torch.zeros_like(mean)))
        self.assertTrue(torch.equal(observed[0], torch.zeros(2)))
        self.assertFalse(model.sit.pos_embed.requires_grad)
        for block in model.sit.blocks:
            self.assertEqual(torch.count_nonzero(block.adaLN_modulation[-1].weight).item(), 0)
        self.assertEqual(torch.count_nonzero(model.sit.final_layer.linear.weight).item(), 0)
        for kwargs in ({"embed_dim": 7}, {"embed_dim": 6, "num_heads": 4}, {"mlp_ratio": 0}):
            with self.assertRaises(ValueError):
                AdaToken(**kwargs)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 6))

    def test_checkpointing_matches_original_sit_forward_and_gradients(self):
        torch.manual_seed(11)
        model = tiny_adatoken(True).train()
        nonzero_sit(model)
        original = copy.deepcopy(model.sit).train()
        mean, target = torch.randn(2, 4, 6), torch.randn(2, 4, 6)
        mapped = model(mean)
        reference = original(mean, mean.new_zeros(len(mean)))
        self.assertTrue(torch.equal(mapped, reference))
        F.mse_loss(mapped, target).backward()
        F.mse_loss(reference, target).backward()
        for name, parameter in original.named_parameters():
            actual = dict(model.sit.named_parameters())[name]
            if parameter.grad is not None:
                self.assertTrue(torch.equal(parameter.grad, actual.grad), name)

    def test_uneven_microbatches_preserve_mse_and_parameter_gradients(self):
        torch.manual_seed(13)
        full = tiny_adatoken(False)
        nonzero_sit(full)
        micro = copy.deepcopy(full)
        mean, target = torch.randn(5, 4, 6), torch.randn(5, 4, 6)
        scaler = mapping_scaler(torch.device("cpu"), "no")
        expected = F.mse_loss(full(mean), target)
        expected.backward()
        actual = backward_mapping_batch(micro, mean, target, scaler, micro_batch_size=2)
        torch.testing.assert_close(actual, expected.detach(), rtol=1e-6, atol=1e-7)
        for left, right in zip(full.parameters(), micro.parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-7)
        with torch.no_grad():
            torch.testing.assert_close(predict_mapping(micro, mean, micro_batch_size=2), micro(mean),
                                       rtol=1e-5, atol=1e-7)

    def test_fp16_requires_cuda_instead_of_silently_falling_back(self):
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            mapping_scaler(torch.device("cpu"), "fp16")
        with self.assertRaises(ValueError):
            mapping_scaler(torch.device("cpu"), "invalid")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_amp_scaler_can_resume_and_handles_gradient_overflow(self):
        model = tiny_adatoken(False).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", init_scale=8)
        mean, target = torch.randn(3, 4, 6, device="cuda"), torch.randn(3, 4, 6, device="cuda")
        loss = backward_mapping_batch(model, mean, target, scaler, "fp16", 1)
        self.assertEqual(loss.dtype, torch.float32)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        restored = mapping_scaler(torch.device("cuda"), "fp16")
        restored.load_state_dict(scaler.state_dict())
        self.assertEqual(restored.state_dict(), scaler.state_dict())
        optimizer.zero_grad(set_to_none=True)
        backward_mapping_batch(model, mean, target, restored, "fp16", 1)
        next(p for p in model.parameters() if p.grad is not None).grad.fill_(float("inf"))
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        old_scale = restored.get_scale()
        restored.unscale_(optimizer)
        restored.step(optimizer)
        restored.update()
        self.assertLess(restored.get_scale(), old_scale)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)


class TestAdaTokenPipeline(unittest.TestCase):
    def test_shared_frozen_means_train_and_encode_without_test_clip(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture, config, path = adatoken_fixture(Path(temporary))
            fixture.run("train_brainvae_scheme1.py", config=fixture.source_config)
            source = fixture.source_run / "checkpoints/best.pt"
            source_hash = file_sha256(source)
            fixture.run("train_clip_mapping.py", "--cache-only")
            cache = fixture.mapping_run / "features/posterior_means.npy"
            cache_stamp, cache_hash = cache.stat().st_mtime_ns, file_sha256(cache)
            smoke = fixture.run("train_clip_mapping.py", "--dry-run", config=path)
            self.assertIn("stage=mapping_smoke", smoke.stdout)
            result = fixture.run("train_clip_mapping.py", config=path)
            self.assertIn("复用冻结均值", result.stdout)
            run = fixture.root / "outputs/subj01/runs/adatoken"
            self.assertFalse((run / "features").exists())
            self.assertEqual((cache.stat().st_mtime_ns, file_sha256(cache)), (cache_stamp, cache_hash))
            self.assertEqual(file_sha256(source), source_hash)
            checkpoint = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["protocol_metadata"]["mean_cache"], str(cache))
            self.assertEqual(checkpoint["protocol_metadata"]["source_train_images"], 8)
            self.assertFalse(checkpoint["protocol_metadata"]["test_used_for_selection"])
            (fixture.root / "nsd/subj01/nsd_sdxl_clip_test_sub1.npy").unlink()
            fixture.run("decode_clip_mapping.py", "--dataset", "all", "--encode-only", config=path)
            for name in ("nsd_test", "vision", "imagery"):
                directory = run / f"evaluation/mapped_mean_fmri/{name}"
                metadata = json.loads((directory / "metadata.json").read_text())
                self.assertEqual(metadata["mapping_head"], "adatoken")
                self.assertEqual(metadata["mapping_precision"], "no")
                tokens = torch.load(directory / "tokens/chunk00000.pt", map_location="cpu", weights_only=False)
                self.assertEqual(tokens["mean"].dtype, torch.float32)
                self.assertEqual(tuple(tokens["mean"].shape), (2, 4, 6))

    def test_adatoken_microbatch_checkpoint_resumes_the_same_optimizer_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture, config, path = adatoken_fixture(Path(temporary))
            fixture.run("train_brainvae_scheme1.py", config=fixture.source_config)
            fixture.run("train_clip_mapping.py", config=path)
            run = fixture.root / "outputs/subj01/runs/adatoken"
            reference = torch.load(run / "checkpoints/last.pt", map_location="cpu", weights_only=False)
            self.assertIn("grad_scaler", reference)
            config["run_name"] = "adatoken_interrupted"
            interrupted = fixture.root / "interrupted.yaml"
            interrupted.write_text(yaml.safe_dump(config))
            injection = (
                "import sys; sys.path.insert(0, " + repr(str(ROOT / "scripts")) + ")\n"
                "import train_clip_mapping as trainer\noriginal = trainer.atomic_torch_save\n"
                "def interrupt(payload, path):\n"
                "    original(payload, path)\n"
                "    if path.name == 'last.pt' and payload['training_state']['global_step'] == 2:\n"
                "        raise RuntimeError('Simulated AdaToken interruption')\n"
                "trainer.atomic_torch_save = interrupt\ntrainer.main()\n"
            )
            result = subprocess.run([sys.executable, "-c", injection, "--config", str(interrupted),
                                     "--device", "cpu", "--threads", "1"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=90)
            self.assertIn("Simulated AdaToken interruption", result.stderr)
            fixture.run("train_clip_mapping.py", "--resume", config=interrupted)
            resumed = torch.load(fixture.root / "outputs/subj01/runs/adatoken_interrupted/checkpoints/last.pt",
                                 map_location="cpu", weights_only=False)
            self.assertEqual(reference["training_state"]["history"], resumed["training_state"]["history"])
            for key, value in reference["model"].items():
                self.assertTrue(torch.equal(value, resumed["model"][key]), key)
            for index, moments in reference["optimizer"]["state"].items():
                for key, value in moments.items():
                    self.assertTrue(torch.equal(value, resumed["optimizer"]["state"][index][key]))
            self.assertEqual(reference["grad_scaler"], resumed["grad_scaler"])


if __name__ == "__main__":
    unittest.main()
