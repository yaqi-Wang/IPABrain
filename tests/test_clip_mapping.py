"""Scientific routing, source-only data, cache recovery and optimizer resume."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.scheme1 import PROTOCOL, FMRIStandardizer, digest_json, file_sha256, file_signature, load_scheme1_config
from src.models.brainvae import BrainVAE
from src.models.clip_mapping import MeanBridge, TokenFormer, build_clip_mapper
from src.utils.clip_mapping import cache_source_means, load_mapping_config, training_clip_mean
from decode_clip_mapping import encode_mapped_predictions, validate_mapped_encoding
from decode_brainvae_scheme1 import generate_samples
from train_brainvae_scheme1 import kl_weight_for_epoch


class TestMappingModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_heads_begin_as_raw_identity_and_can_learn(self):
        for mapper in (MeanBridge(4, 6, 8), TokenFormer(4, 6, 8, 2, 2, 2, 0)):
            with self.subTest(head=type(mapper).__name__):
                mean = torch.randn(3, 4, 6) * 3
                target = 2 * mean + 1
                torch.testing.assert_close(mapper(mean), mean, rtol=0, atol=0)
                optimizer = torch.optim.AdamW(mapper.parameters(), lr=0.01)
                initial = float(F.mse_loss(mapper(mean), target))
                for _ in range(12):
                    optimizer.zero_grad()
                    loss = F.mse_loss(mapper(mean), target)
                    loss.backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in mapper.parameters()))
                    optimizer.step()
                self.assertLess(float(F.mse_loss(mapper(mean), target)), initial)
                with self.assertRaises(ValueError):
                    mapper(torch.randn(3, 1, 6))

    def test_transformer_exchanges_information_between_tokens(self):
        mapper = TokenFormer(4, 6, 8, 2, 2, 2, 0).eval()
        torch.nn.init.normal_(mapper.output_projection.weight, std=0.1)
        mean = torch.randn(1, 4, 6)
        changed = mean.clone()
        changed[:, 0, 0] += 5
        self.assertFalse(torch.allclose(mapper(mean)[:, 1], mapper(changed)[:, 1]))
        residual = MeanBridge(4, 6, 8)
        torch.nn.init.normal_(residual.residual[-1].weight, std=0.1)
        torch.testing.assert_close(residual(mean)[:, 1], residual(changed)[:, 1], rtol=0, atol=0)

    def test_mapping_backward_does_not_update_the_source_encoder(self):
        encoder = torch.nn.Linear(6, 24).eval().requires_grad_(False)
        before = copy.deepcopy(encoder.state_dict())
        with torch.no_grad():
            mean = encoder(torch.randn(3, 6)).reshape(3, 4, 6)
        mapper = MeanBridge(4, 6, 8)
        F.mse_loss(mapper(mean), torch.randn_like(mean)).backward()
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        self.assertGreater(float(mapper.residual[-1].weight.grad.abs().sum()), 0)
        for key in before:
            torch.testing.assert_close(before[key], encoder.state_dict()[key], rtol=0, atol=0)

    def test_nomse_config_only_changes_the_requested_coefficient_and_paths(self):
        reference, _ = load_scheme1_config("configs/brainvae_scheme1_mean.yaml", ROOT)
        candidate, _ = load_scheme1_config("configs/brainvae_scheme1_mean_nomse.yaml", ROOT)
        expected = {**reference, "run_name": "brainvae_mean_nomse", "log_root": "logs/scheme1_mean_nomse",
                    "model": {**reference["model"], "clip_reg_weight": 0.0}}
        self.assertEqual(candidate, expected)
        for epoch in range(31):
            self.assertEqual(kl_weight_for_epoch(epoch, candidate), kl_weight_for_epoch(epoch, reference))
        first, source1, _ = load_mapping_config("configs/clip_mapping_meanbridge.yaml", ROOT)
        second, source2, _ = load_mapping_config("configs/clip_mapping_tokenformer.yaml", ROOT)
        self.assertEqual(source1["model"]["clip_reg_weight"], 100)
        self.assertEqual(source2["model"]["clip_reg_weight"], 0)
        self.assertEqual(second["model"]["depth"], 8)
        self.assertNotEqual(first["run_name"], second["run_name"])
        self.assertEqual(source1["preprocessing"], source2["preprocessing"])
        self.assertEqual(source1["validation"], source2["validation"])

    def test_training_mean_control_excludes_validation_images(self):
        clip = np.stack([np.ones((4, 6)), np.full((4, 6), 10000), np.full((4, 6), 3)]).astype(np.float32)
        control = training_clip_mean(clip, np.array([0, 2]), chunk_size=1)
        torch.testing.assert_close(control, torch.full((4, 6), 2.0))

    def test_invalid_heads_and_geometry_are_rejected(self):
        for options in ({"type": "s2n"}, {"type": "meanbridge", "hidden_dim": 0},
                        {"type": "tokenformer", "hidden_dim": 7, "num_heads": 2}):
            with self.assertRaises(ValueError):
                build_clip_mapper(options)


class TestMappingCache(unittest.TestCase):
    def test_interrupted_cache_resumes_and_rejects_a_different_brainvae(self):
        class Encoder(torch.nn.Module):
            def __init__(self, fail=False):
                super().__init__()
                self.calls, self.fail = 0, fail
            def encode(self, value):
                self.calls += 1
                if self.fail and self.calls == 2:
                    raise RuntimeError("interrupted encoder")
                return SimpleNamespace(mode=lambda: value[:, None, :].square())
        data = [{"fmri": torch.tensor([float(i), float(i + 1)])} for i in range(5)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            identity = {"brainvae_checkpoint_sha256": "first"}
            with self.assertRaisesRegex(RuntimeError, "interrupted encoder"):
                cache_source_means(lambda: Encoder(True), data, directory, identity, (1, 2), 2, torch.device("cpu"))
            self.assertEqual(json.loads((directory / "cache_state.json").read_text())["next_index"], 2)
            path = cache_source_means(lambda: Encoder(), data, directory, identity, (1, 2), 2, torch.device("cpu"))
            expected = np.array([[[i * i, (i + 1) ** 2]] for i in range(5)], dtype=np.float32)
            np.testing.assert_array_equal(np.load(path), expected)
            def forbidden():
                raise AssertionError("Completed cache should not instantiate the encoder")
            self.assertEqual(cache_source_means(forbidden, data, directory, identity, (1, 2), 2, torch.device("cpu")), path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "different encoder"):
                cache_source_means(forbidden, data, directory, {"brainvae_checkpoint_sha256": "second"}, (1, 2), 2, torch.device("cpu"))
            self.assertEqual(path.read_bytes(), before)

    def test_generator_receives_mapped_tokens_for_each_target(self):
        class Dataset:
            def __len__(self):
                return 3
            def __getitem__(self, index):
                return {"fmri": torch.full((6,), float(index + 1)), "index": index}
            def record(self, index):
                return {"stimulus_id": f"target{index}", "task_type": "vision", "stimulus_type": "simple"}
        class Brain:
            def encode(self, value):
                return SimpleNamespace(mode=lambda: value[:, None, :].expand(-1, 4, -1))
        class Mapper:
            latent_tokens, embed_dim = 4, 6
            def __call__(self, mean):
                return 2 * mean + 7
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            chunks = encode_mapped_predictions(Brain(), Mapper(), Dataset(), directory, 2, "cpu", "head1")
            received = []
            def decode(tokens, *args, **kwargs):
                received.append(tokens.clone())
                return torch.zeros(1, 3, 2, 2)
            def save(value, path):
                Path(path).write_bytes(b"mock generated image")
            generate_samples(chunks, directory, {"aggregation": "mean_fmri"}, None, None, 0,
                             decode, save, 2, 42, {}, False)
            self.assertEqual(len(received), 6)
            for i, value in enumerate(received):
                torch.testing.assert_close(value, torch.full((1, 4, 6), float(2 * (i % 3 + 1) + 7)))
            with self.assertRaisesRegex(ValueError, "cache mismatch"):
                encode_mapped_predictions(Brain(), Mapper(), Dataset(), directory, 2, "cpu", "head2", resume=True)

    def test_source_metadata_protects_both_heads_and_encoders(self):
        metadata = {"checkpoint_sha256": "head1", "brainvae_checkpoint_sha256": "brain1",
                    "mapping_head": "meanbridge", "token_representation": "mapped_raw_clip_tokens"}
        validate_mapped_encoding(metadata, metadata)
        for key, value in (("checkpoint_sha256", "head2"), ("brainvae_checkpoint_sha256", "brain2"),
                           ("mapping_head", "tokenformer"), ("token_representation", "posterior_mean")):
            with self.assertRaises(ValueError):
                validate_mapped_encoding(metadata, {**metadata, key: value})


class MappingFixture:
    def __init__(self, root: Path, kind: str):
        self.root, self.kind = root, kind
        rng = np.random.default_rng(7)
        nsd = root / "nsd/subj01"
        nsd.mkdir(parents=True)
        self.fmri = rng.normal(size=(10, 3, 6)).astype(np.float32)
        self.clip = rng.normal(size=(10, 4, 6)).astype(np.float16)
        for name, ids, fmri, clip in (("train", np.arange(100, 110), self.fmri, self.clip),
                                     ("test", np.arange(200, 203), self.fmri[:3], self.clip[:3])):
            np.save(nsd / f"nsd_{name}_fmri_scale_sub1.npy", fmri)
            np.save(nsd / f"nsd_{name}_image_id_sub1.npy", ids)
            np.save(nsd / f"nsd_sdxl_clip_{name}_sub1.npy", clip)
        paths_file = root / "paths.yaml"
        paths_file.write_text(yaml.safe_dump({"nsd_root": str(root / "nsd"), "data_root": str(root),
                                              "raw_nsd_root": str(root / "raw"), "output_root": str(root / "outputs"),
                                              "checkpoint_root": str(root / "unused_checkpoints"),
                                              "imagery_root": str(root / "unused_imagery")}))
        source_name = "brainvae_scheme1_mean" if kind == "meanbridge" else "brainvae_scheme1_mean_nomse"
        config = yaml.safe_load((ROOT / f"configs/{source_name}.yaml").read_text())
        config.update({"base_config": str(ROOT / "configs/brainvae.yaml"), "paths": str(paths_file),
                       "output_root": str(root / "outputs"), "log_root": str(root / "logs"),
                       "training": {"batch_size": 2, "epochs": 1, "seed": 42, "checkpoint_every_steps": 1},
                       "model": {**config["model"], "latent_tokens": 4, "embed_dim": 6, "hidden_dim": 8,
                                 "linear_dim": 8, "hidden_channels": 4, "free_bits": 0,
                                 "ddconfig": {"z_channels": 4, "in_channels": 1, "out_ch": 1, "ch": 4,
                                              "ch_mult": [1], "ada_length": 8, "num_res_blocks": 1,
                                              "num_down_blocks": 0, "num_up_blocks": 0,
                                              "mid_attn_type": "none", "block_attn_type": "none"}}})
        config["validation"]["fraction"] = 0.2
        self.source_config = root / "brain.yaml"
        self.source_config.write_text(yaml.safe_dump(config))
        loaded, _ = load_scheme1_config(self.source_config, ROOT)
        prepared = root / "outputs/subj01/data_v1"
        prepared.mkdir(parents=True)
        self.train_indices, self.val_indices = np.arange(8), np.arange(8, 10)
        normalizer = FMRIStandardizer.fit(self.fmri, self.train_indices, 2000 / 300, 300)
        np.savez(prepared / "preprocessing.npz", **normalizer.arrays())
        np.savez(prepared / "split.npz", source_image_ids=np.arange(100, 110), test_image_ids=np.arange(200, 203),
                 imagery_nsd_ids=np.array([300, 301]), train_indices=self.train_indices, val_indices=self.val_indices)
        np.savez(prepared / "imagery_trials.npz", subject_id=1, common_divisor=300.0,
                 fmri_scaled=rng.normal(size=(8, 6)).astype(np.float32),
                 task_type=np.array(["vision"] * 4 + ["imagery"] * 4),
                 stimulus_type=np.array(["simple", "simple", "complex", "complex"] * 2),
                 stimulus_id=np.array(["a", "a", "b", "b"] * 2), global_trial_idx=np.arange(8),
                 run=np.ones(8, dtype=int), condition_index=np.zeros(8, dtype=int))
        names = {"train_fmri": "nsd_train_fmri_scale_sub1.npy", "train_ids": "nsd_train_image_id_sub1.npy",
                 "train_clip": "nsd_sdxl_clip_train_sub1.npy", "test_fmri": "nsd_test_fmri_scale_sub1.npy",
                 "test_ids": "nsd_test_image_id_sub1.npy"}
        spec = {"files": {key: file_signature(nsd / name) for key, name in names.items()},
                "preprocessing": loaded["preprocessing"], "val_fraction": 0.2, "split_seed": 42}
        manifest = {"protocol": PROTOCOL, "subject": 1, "input_spec": spec, "data_fingerprint": digest_json(spec),
                    "artifact_sha256": {name: file_sha256(prepared / name)
                                        for name in ("split.npz", "preprocessing.npz", "imagery_trials.npz")}}
        (prepared / "manifest.json").write_text(json.dumps(manifest))
        mapping = yaml.safe_load((ROOT / f"configs/clip_mapping_{kind}.yaml").read_text())
        model = {"type": kind, "latent_tokens": 4, "embed_dim": 6, "hidden_dim": 8}
        if kind == "tokenformer":
            model.update({"depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "dropout": 0.2})
        mapping.update({"brainvae_config": str(self.source_config), "output_root": str(root / "outputs"),
                        "log_root": str(root / "mapping_logs"), "model": model})
        mapping["training"].update({"epochs": 2, "batch_size": 2, "encode_batch_size": 2, "checkpoint_every_steps": 1})
        mapping["validation"].update({"patience": 0, "min_delta": 0.0, "batch_size": 2})
        mapping["evaluation"]["encode_batch_size"] = 2
        self.mapping = mapping
        self.mapping_config = root / "mapping.yaml"
        self.mapping_config.write_text(yaml.safe_dump(mapping))
        self.source_run = root / f"outputs/subj01/runs/{loaded['run_name']}"
        self.mapping_run = root / f"outputs/subj01/runs/{kind}"

    def run(self, script, *args, config=None, expect_success=True):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--config",
                                 str(config or self.mapping_config), "--device", "cpu", "--threads", "1", *args],
                                cwd=ROOT, text=True, capture_output=True, timeout=90)
        if expect_success and result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return result


class TestMappingPipeline(unittest.TestCase):
    def test_both_methods_train_and_encode_three_branches_without_target_features(self):
        for kind in ("meanbridge", "tokenformer"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                fixture = MappingFixture(Path(temporary), kind)
                fixture.run("train_brainvae_scheme1.py", config=fixture.source_config)
                source = fixture.source_run / "checkpoints/best.pt"
                source_digest = file_sha256(source)
                smoke = fixture.run("train_clip_mapping.py", "--dry-run")
                self.assertIn("stage=mapping_smoke", smoke.stdout)
                self.assertFalse(fixture.mapping_run.exists())
                fixture.run("train_clip_mapping.py")
                self.assertEqual(file_sha256(source), source_digest)
                head = torch.load(fixture.mapping_run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
                self.assertEqual(head["protocol_metadata"]["source_identity"]["brainvae_checkpoint_sha256"], source_digest)
                self.assertFalse(head["protocol_metadata"]["test_used_for_selection"])
                self.assertEqual(head["protocol_metadata"]["source_train_images"], 8)
                self.assertEqual(head["protocol_metadata"]["source_val_images"], 2)
                control = fixture.clip[fixture.train_indices].astype(np.float64).mean(axis=0).astype(np.float32)
                np.testing.assert_allclose(head["training_mean_clip"].numpy(), control, rtol=0, atol=1e-7)
                brain_payload = torch.load(source, map_location="cpu", weights_only=False)
                brain = BrainVAE(**brain_payload["config"]).eval()
                brain.load_state_dict(brain_payload["model"])
                norm = FMRIStandardizer.from_checkpoint(brain_payload["preprocessing"])
                means = np.load(fixture.mapping_run / "features/posterior_means.npy")
                with torch.no_grad():
                    expected = brain.encode(torch.from_numpy(norm.transform(fixture.fmri.mean(axis=1), "nsd"))).mode().numpy()
                np.testing.assert_allclose(means, expected, rtol=1e-5, atol=1e-6)
                # Inference remains possible when true test CLIP features do not exist.
                (fixture.root / "nsd/subj01/nsd_sdxl_clip_test_sub1.npy").unlink()
                fixture.run("decode_clip_mapping.py", "--dataset", "all", "--encode-only", "--resume")
                for name, count in (("nsd_test", 3), ("vision", 2), ("imagery", 2)):
                    directory = fixture.mapping_run / f"evaluation/mapped_mean_fmri/{name}"
                    metadata = json.loads((directory / "metadata.json").read_text())
                    self.assertEqual(metadata["n_items"], count)
                    self.assertEqual(metadata["token_representation"], "mapped_raw_clip_tokens")
                    payload = torch.load(directory / "tokens/chunk00000.pt", map_location="cpu", weights_only=False)
                    self.assertEqual(tuple(payload["mean"].shape), (2, 4, 6))
                    self.assertEqual(payload["representation"], "mapped_raw_clip_tokens")
                fixture.run("decode_clip_mapping.py", "--dataset", "all", "--encode-only", "--resume")
                repeated = fixture.run("train_clip_mapping.py", expect_success=False)
                self.assertNotEqual(repeated.returncode, 0)
                self.assertIn("checkpoints already exist", repeated.stderr)

    def test_transformer_exact_mid_epoch_resume_and_changed_learning_rate_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MappingFixture(Path(temporary), "tokenformer")
            fixture.run("train_brainvae_scheme1.py", config=fixture.source_config)
            fixture.run("train_clip_mapping.py")
            reference = torch.load(fixture.mapping_run / "checkpoints/last.pt", map_location="cpu", weights_only=False)
            interrupted_config = copy.deepcopy(fixture.mapping)
            interrupted_config["run_name"] = "tokenformer_interrupted"
            config_path = fixture.root / "interrupted.yaml"
            config_path.write_text(yaml.safe_dump(interrupted_config))
            injection = (
                "import sys; sys.path.insert(0, " + repr(str(ROOT / "scripts")) + ")\n"
                "import train_clip_mapping as trainer\n"
                "original = trainer.atomic_torch_save\n"
                "def interrupt(payload, path):\n"
                "    original(payload, path)\n"
                "    if path.name == 'last.pt' and payload['training_state']['global_step'] == 2:\n"
                "        raise RuntimeError('Simulated mapping interruption')\n"
                "trainer.atomic_torch_save = interrupt\ntrainer.main()\n"
            )
            result = subprocess.run([sys.executable, "-c", injection, "--config", str(config_path),
                                     "--device", "cpu", "--threads", "1"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=90)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Simulated mapping interruption", result.stderr)
            fixture.run("train_clip_mapping.py", "--resume", config=config_path)
            run_dir = fixture.root / "outputs/subj01/runs/tokenformer_interrupted"
            resumed = torch.load(run_dir / "checkpoints/last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(reference["training_state"]["history"], resumed["training_state"]["history"])
            for key in reference["model"]:
                self.assertTrue(torch.equal(reference["model"][key], resumed["model"][key]), key)
            for index, moments in reference["optimizer"]["state"].items():
                for key, expected in moments.items():
                    self.assertTrue(torch.equal(expected, resumed["optimizer"]["state"][index][key]))
            completion = run_dir / "training_complete.json"
            completion.unlink()
            fixture.run("train_clip_mapping.py", "--resume", config=config_path)
            self.assertTrue(completion.exists())
            changed = copy.deepcopy(interrupted_config)
            changed["training"]["learning_rate"] *= 2
            changed_path = fixture.root / "changed.yaml"
            changed_path.write_text(yaml.safe_dump(changed))
            stamp = (run_dir / "checkpoints/last.pt").stat().st_mtime_ns
            rejected = fixture.run("train_clip_mapping.py", "--resume", config=changed_path, expect_success=False)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("resume config", rejected.stderr)
            self.assertEqual((run_dir / "checkpoints/last.pt").stat().st_mtime_ns, stamp)


if __name__ == "__main__":
    unittest.main()
