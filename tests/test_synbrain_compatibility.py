"""Numerical parity with the existing upstream copy and strict source loading."""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from einops import rearrange
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.scheme1 import (
    PROTOCOL, FMRIStandardizer, data_directory, file_sha256, file_signature, load_scheme1_config,
)
from src.ldm.modules.distributions.distributions import DiagonalGaussianDistribution_2D
from src.models.brainvae_factory import build_brainvae
from src.utils.clip_mapping import frozen_brain, load_brain_source, load_mapping_config
from src.utils.synbrain_checkpoint import (
    external_model_config, read_synbrain_checkpoint, validate_synbrain_weights,
)
from train_brainvae_scheme1 import kl_weight_for_epoch


def tiny_config(checkpointing=False):
    return {"architecture": "synbrain", "latent_tokens": 2, "embed_dim": 3,
            "hidden_channels": 4, "hidden_dim": 8, "linear_dim": 8,
            "clip_weight": 10.0, "kl_weight": 0.001, "activation_checkpointing": checkpointing,
            "semantic_source": "sample", "free_bits": 0.0, "clip_reg_weight": 0.0, "recon_weight": 1.0,
            "ddconfig": {"double_z": False, "z_channels": 2, "in_channels": 1, "out_ch": 1,
                         "ch": 4, "ch_mult": [1, 2], "ada_length": 16, "resolution": 16,
                         "num_res_blocks": 1, "num_down_blocks": 1, "num_up_blocks": 1,
                         "attn_resolutions": [], "dropout": 0.0, "attn_type": "vanilla"}}


def upstream_class():
    """Execute upstream definitions after removing only unused import statements."""
    namespace = {"torch": torch, "nn": nn, "F": torch.nn.functional, "np": np,
                 "rearrange": rearrange, "DiagonalGaussianDistribution_2D": DiagonalGaussianDistribution_2D}
    names = {"nonlinearity", "Normalize", "Downsample", "Upsample", "ResnetBlock", "LinAttnBlock",
             "AttnBlock", "make_attn", "NeuroEncoder", "NeuroDecoder"}
    for path, selected in ((ROOT / "src/vae/module.py", names),
                           (ROOT / "src/vae/brainvae.py", {"MLP", "BrainVAE"})):
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                     and node.name in selected]
        exec(compile(tree, str(path), "exec"), namespace)
    return namespace["BrainVAE"]


class TestSynBrainParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_upstream_state_keys_mean_sample_losses_and_gradients(self):
        config = tiny_config()
        original = upstream_class()(ddconfig=config["ddconfig"], hidden_dim=8, linear_dim=8,
                                    embed_dim=3, clip_weight=10, kl_weight=0.001)
        current = build_brainvae(config)
        current.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(list(current.state_dict()), list(original.state_dict()))
        self.assertIn("encoder.down.1.block.0.nin_shortcut.weight", current.state_dict())
        self.assertIn("decoder.up.1.upsample.trans_conv.weight", current.state_dict())
        fmri, target = torch.randn(3, 1, 10), torch.randn(3, 2, 3)
        for sampled in (False, True):
            original.zero_grad(set_to_none=True)
            current.zero_grad(set_to_none=True)
            torch.manual_seed(101)
            z1, recon1, rec1, kl1, clip1, loss1 = original(fmri, target, sampled)
            torch.manual_seed(101)
            z2, recon2, losses = current(fmri, target, sample_posterior=sampled)
            for first, second in ((z1, z2), (recon1, recon2), (rec1, losses["recon_loss"]),
                                  (kl1, losses["kl_loss"]), (clip1, losses["clip_loss"]), (loss1, losses["loss"])):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            loss1.backward()
            losses["loss"].backward()
            for first, second in zip(original.parameters(), current.parameters()):
                if first.grad is None:
                    self.assertIsNone(second.grad)
                else:
                    torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)

    def test_checkpointing_preserves_values_and_gradients(self):
        baseline = build_brainvae(tiny_config())
        checkpointed = build_brainvae(tiny_config(True))
        checkpointed.load_state_dict(baseline.state_dict(), strict=True)
        fmri, target = torch.randn(3, 10), torch.randn(3, 2, 3)
        torch.manual_seed(102)
        first = baseline(fmri, target)[2]["loss"]
        torch.manual_seed(102)
        second = checkpointed(fmri, target)[2]["loss"]
        first.backward()
        second.backward()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for a, b in zip(baseline.parameters(), checkpointed.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_semantic_loss_reaches_variance_and_logvar_is_not_clipped(self):
        model = build_brainvae(tiny_config())
        _, _, losses = model(torch.randn(3, 10), torch.randn(3, 2, 3))
        losses["clip_loss"].backward()
        self.assertGreater(float(model.pre_projector_logvar[-1].weight.grad.abs().sum()), 0)
        with torch.no_grad():
            model.pre_projector_logvar[-1].weight.zero_()
            model.pre_projector_logvar[-1].bias.fill_(21)
        self.assertEqual(float(model.encode(torch.randn(2, 10)).logvar.min()), 21)

    def test_project_configs_and_constant_original_kl_are_preserved(self):
        original, _ = load_scheme1_config("configs/brainvae_scheme1_synbrain.yaml", ROOT)
        current, _ = load_scheme1_config("configs/brainvae_scheme1_mean_nomse.yaml", ROOT)
        self.assertEqual(current["model"]["ddconfig"]["mid_attn_type"], "linear")
        self.assertEqual(current["model"]["semantic_source"], "mean")
        self.assertEqual(current["preprocessing"]["normalization"], "source_train")
        for epoch in (0, 1, 5, 29):
            self.assertEqual(kl_weight_for_epoch(epoch, original), 0.001)
        local, _, _ = load_mapping_config("configs/clip_mapping_synbrain_local.yaml", ROOT)
        author, _, _ = load_mapping_config("configs/clip_mapping_synbrain_author.yaml", ROOT)
        for key in ("model", "training", "validation", "evaluation"):
            self.assertEqual(local[key], author[key])
        self.assertNotEqual(local["run_name"], author["run_name"])

    def test_beta_scaling_roundtrip_keeps_raw_nsd_values(self):
        normalizer = FMRIStandardizer.scaled_beta(3, 1, 2000)
        values = np.array([1, -2, 3], dtype=np.float32)
        np.testing.assert_array_equal(normalizer.transform(values, "nsd"), values)
        np.testing.assert_array_equal(normalizer.transform(values, "imagery_scaled"), values)
        restored = FMRIStandardizer.from_checkpoint(normalizer.checkpoint_state())
        self.assertEqual(restored.normalization, "beta_scale")
        self.assertEqual(restored.description, "beta_divisor_2000_no_zscore")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preprocessing.npz"
            np.savez(path, **normalizer.arrays())
            loaded = FMRIStandardizer.load(path)
            np.testing.assert_array_equal(loaded.transform(values, "nsd"), values)
        legacy = FMRIStandardizer(np.zeros(3), np.ones(3), 1, 300)
        self.assertNotIn("normalization", legacy.checkpoint_state())
        self.assertNotIn("normalization", legacy.arrays())

    def test_checkpoint_formats_prefixes_and_strict_failures(self):
        model = build_brainvae(tiny_config())
        state = model.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "author.pth"
            prefixed = {"module." + key: value for key, value in state.items()}
            for payload in ({"model": prefixed, "epoch": 3}, {"state_dict": state},
                            {"model_state_dict": state}, state):
                torch.save(payload, path)
                weights, _ = read_synbrain_checkpoint(path)
                validate_synbrain_weights(weights, tiny_config())
                model.load_state_dict(weights, strict=True)
            bad = dict(state)
            bad.pop("encoder.mid.attn_1.q.weight")
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_synbrain_weights(bad, tiny_config())
            bad = {**state, "decoder.extra.weight": torch.ones(1)}
            with self.assertRaisesRegex(ValueError, "unexpected"):
                validate_synbrain_weights(bad, tiny_config())
            incompatible = copy.deepcopy(tiny_config())
            incompatible["ddconfig"]["ada_length"] = 8
            with self.assertRaisesRegex(ValueError, "matching ddconfig"):
                validate_synbrain_weights(state, incompatible)
            supplied = Path(directory) / "author.yaml"
            supplied.write_text(yaml.safe_dump({"model": {"params": {"ddconfig": tiny_config()["ddconfig"]}}}))
            validate_synbrain_weights(state, external_model_config(state, incompatible, supplied))
            torch.save({"model": {"input_projection.weight": torch.ones(3, 3)}}, path)
            with self.assertRaisesRegex(ValueError, "Stage1"):
                read_synbrain_checkpoint(path)


class TestSynBrainPipeline(unittest.TestCase):
    def test_real_training_resume_and_external_frozen_mapping(self):
        """Small CPU fixture uses actual scripts/data; no experiment files touched."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nsd = root / "nsd/subj01"
            nsd.mkdir(parents=True)
            rng = np.random.default_rng(5)
            file_spec = {}
            source_ids, test_ids = np.arange(10, 16), np.arange(20, 22)
            for name, ids in (("train", source_ids), ("test", test_ids)):
                for kind, filename, value in (
                    ("fmri", f"nsd_{name}_fmri_scale_sub1.npy", rng.normal(size=(len(ids), 3, 10)).astype(np.float32)),
                    ("ids", f"nsd_{name}_image_id_sub1.npy", ids),
                    ("clip", f"nsd_sdxl_clip_{name}_sub1.npy", rng.normal(size=(len(ids), 2, 3)).astype(np.float32)),
                ):
                    path = nsd / filename
                    np.save(path, value)
                    file_spec[name + "_" + kind] = file_signature(path)
            paths = root / "paths.yaml"
            paths.write_text(yaml.safe_dump({"nsd_root": str(root / "nsd"), "data_root": str(root),
                                            "imagery_root": str(root), "checkpoint_root": str(root),
                                            "output_root": str(root / "outputs")}))
            base, _ = load_scheme1_config("configs/brainvae_scheme1_synbrain.yaml", ROOT)
            base["model"] = tiny_config(True)
            base.update({"paths": str(paths), "output_root": str(root / "outputs"), "run_name": "original"})
            base.pop("paths_config", None)
            base["training"].update({"epochs": 1, "batch_size": 3, "checkpoint_every_steps": 1})
            base["validation"].update({"fraction": 1 / 3, "patience": 0})
            config_path = root / "brain.yaml"
            # A self-contained base config avoids inheritance from the current linear VAE.
            base_path = root / "base.yaml"
            base_path.write_text(yaml.safe_dump(base))
            config_path.write_text(yaml.safe_dump({"base_config": str(base_path)}))
            config, _ = load_scheme1_config(config_path, ROOT)
            prepared = data_directory(config, ROOT, 1)
            prepared.mkdir(parents=True)
            split = {"source_image_ids": source_ids, "test_image_ids": test_ids,
                     "train_indices": np.arange(4), "val_indices": np.arange(4, 6),
                     "imagery_nsd_ids": np.empty(0, dtype=np.int64)}
            np.savez(prepared / "split.npz", **split)
            normalizer = FMRIStandardizer.scaled_beta(10, 1, 2000)
            np.savez(prepared / "preprocessing.npz", **normalizer.arrays())
            manifest = {"protocol": PROTOCOL, "subject": 1, "data_fingerprint": "synthetic-original",
                        "voxel_order": "native_XYZ_mask_C_order", "source_sample_check": {"status": "verified"},
                        "input_spec": {"files": file_spec, "preprocessing": config["preprocessing"],
                                       "val_fraction": config["validation"]["fraction"], "split_seed": 42},
                        "artifact_sha256": {name: file_sha256(prepared / name)
                                            for name in ("split.npz", "preprocessing.npz")}}
            (prepared / "manifest.json").write_text(json.dumps(manifest))

            def run(script, config_file, *args):
                result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--config", str(config_file),
                                         "--device", "cpu", "--threads", "1", *args],
                                        cwd=ROOT, capture_output=True, text=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result

            run("train_brainvae_scheme1.py", config_path)
            local_dir = root / "outputs/subj01/runs/original"
            before = file_sha256(local_dir / "checkpoints/best.pt")
            resumed = run("train_brainvae_scheme1.py", config_path, "--resume")
            self.assertIn("不再训练", resumed.stdout)
            self.assertEqual(before, file_sha256(local_dir / "checkpoints/best.pt"))
            saved = torch.load(local_dir / "checkpoints/best.pt", weights_only=False)
            self.assertEqual(saved["protocol_metadata"]["loss_definition"]["semantic_source"], "sample")
            self.assertEqual(saved["preprocessing"]["normalization"], "beta_scale")
            author = root / "author.pth"
            torch.save({"model": {"module." + k: v for k, v in saved["model"].items()}, "epoch": 4}, author)
            # Direct decoding must work before any prediction head exists.
            run("decode_synbrain_direct.py", config_path, "--source", "local", "--max-items", "2",
                "--encode-only")
            run("decode_synbrain_direct.py", config_path, "--source", "author", "--checkpoint", str(author),
                "--max-items", "2", "--encode-only")
            direct_local = local_dir / "evaluation/direct_mean_nsd2_s1/nsd_test"
            direct_author = root / "outputs/subj01/runs/brainvae_synbrain_author/evaluation/direct_mean_nsd2_s1/nsd_test"
            local_metadata = json.loads((direct_local / "metadata.json").read_text())
            author_metadata = json.loads((direct_author / "metadata.json").read_text())
            self.assertIsNone(local_metadata["mapping_head"])
            self.assertIsNone(author_metadata["mapping_head"])
            self.assertEqual(local_metadata["num_samples"], 1)
            self.assertEqual(local_metadata["selection_manifest_sha256"], author_metadata["selection_manifest_sha256"])
            local_tokens = torch.load(direct_local / "tokens/chunk00000.pt", weights_only=False)
            author_tokens = torch.load(direct_author / "tokens/chunk00000.pt", weights_only=False)
            self.assertEqual(local_tokens["records"], author_tokens["records"])
            torch.testing.assert_close(local_tokens["mean"], author_tokens["mean"], rtol=0, atol=0)
            before_tokens = file_sha256(direct_local / "tokens/chunk00000.pt")
            run("decode_synbrain_direct.py", config_path, "--source", "local", "--max-items", "2",
                "--encode-only", "--resume")
            self.assertEqual(file_sha256(direct_local / "tokens/chunk00000.pt"), before_tokens)
            mapping, _, _ = load_mapping_config("configs/clip_mapping_synbrain_author.yaml", ROOT)
            mapping.update({"brainvae_config": str(config_path), "brainvae_checkpoint": str(author),
                            "run_name": "author_head", "output_root": str(root / "outputs")})
            mapping["model"] = {"type": "meanbridge", "latent_tokens": 2, "embed_dim": 3, "hidden_dim": 4}
            mapping["training"].update({"epochs": 1, "batch_size": 3, "encode_batch_size": 3})
            mapping_path = root / "mapping.yaml"
            mapping_path.write_text(yaml.safe_dump(mapping))
            source = load_brain_source(mapping, config, ROOT, 1)
            brain = frozen_brain(source, torch.device("cpu"))
            self.assertTrue(all(not p.requires_grad for p in brain.parameters()))
            self.assertFalse(brain.training)
            with self.assertRaisesRegex(ValueError, "matching subject"):
                load_brain_source(mapping, config, ROOT, 2)
            wrong = {**mapping, "external_source": {"subject": 1}}
            config_wrong = copy.deepcopy(config)
            config_wrong["preprocessing"]["common_divisor"] = 300
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                load_brain_source(wrong, config_wrong, ROOT, 1)
            run("train_clip_mapping.py", mapping_path)
            head_dir = root / "outputs/subj01/runs/author_head"
            head = torch.load(head_dir / "checkpoints/best.pt", weights_only=False)
            self.assertTrue(head["protocol_metadata"]["brainvae_frozen"])
            self.assertEqual(head["protocol_metadata"]["source_identity"]["source_kind"], "synbrain_external")
            self.assertTrue((head_dir / "features/posterior_means.npy").is_file())
            self.assertEqual(file_sha256(local_dir / "checkpoints/best.pt"), before)
            run("train_clip_mapping.py", mapping_path, "--resume")
            run("decode_clip_mapping.py", mapping_path, "--dataset", "nsd_test", "--encode-only")
            metadata = json.loads((head_dir / "evaluation/mapped_mean_fmri/nsd_test/metadata.json").read_text())
            self.assertEqual(metadata["normalization"], "beta_divisor_2000_no_zscore")
            self.assertEqual(metadata["n_items"], 2)
            # A different subset gets its own token cache, preserving existing predictions.
            original_metadata = (head_dir / "evaluation/mapped_mean_fmri/nsd_test/metadata.json").read_bytes()
            run("decode_clip_mapping.py", mapping_path, "--dataset", "nsd_test", "--max-items", "1",
                "--token-source-tag", "mapped_nsd_first1", "--encode-only")
            selected = head_dir / "evaluation/mapped_nsd_first1/nsd_test"
            self.assertEqual(json.loads((selected / "metadata.json").read_text())["n_items"], 1)
            self.assertTrue((selected / "tokens/chunk00000.pt").is_file())
            self.assertEqual((head_dir / "evaluation/mapped_mean_fmri/nsd_test/metadata.json").read_bytes(), original_metadata)


class TestSynBrainRunner(unittest.TestCase):
    def test_local_source_head_sequence_and_author_never_trains_encoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            runner = root / "scripts/run_synbrain_comparison.sh"
            shutil.copyfile(ROOT / "scripts/run_synbrain_comparison.sh", runner)
            trace = root / "trace.txt"
            executable = root / "python-mock"
            executable.write_text("#!/usr/bin/env bash\n"
                                  "printf '%s\\n' \"$*\" >> \"$SYNBRAIN_TEST_TRACE\"\n"
                                  "if [[ ${SYNBRAIN_FAIL_PREP:-0} == 1 && $* == *prepare_scheme1_data.py* ]]; then exit 7; fi\n")
            executable.chmod(0o755)
            environment = {**os.environ, "SYNBRAIN_PYTHON": str(executable), "SYNBRAIN_TEST_TRACE": str(trace)}
            def run(source, stage="train", env=None):
                return subprocess.run(["bash", str(runner), source, stage, "1", "cpu"],
                                      env=env or environment, capture_output=True, text=True, timeout=10)
            result = run("local")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = trace.read_text().splitlines()
            self.assertEqual(len(calls), 3)
            for call, script in zip(calls, ("prepare_scheme1_data.py", "train_brainvae_scheme1.py", "train_clip_mapping.py")):
                self.assertIn(script, call)
            trace.unlink()
            custom = {**environment, "SYNBRAIN_AUTHOR_CHECKPOINT": "/tmp/downloaded BrainVAE.pth"}
            result = run("author", env=custom)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = trace.read_text().splitlines()
            self.assertEqual(len(calls), 2)
            self.assertNotIn("train_brainvae_scheme1.py", "\n".join(calls))
            self.assertIn("--brainvae-checkpoint /tmp/downloaded BrainVAE.pth", calls[1])
            trace.unlink()
            result = run("local", env={**environment, "SYNBRAIN_FAIL_PREP": "1"})
            self.assertEqual(result.returncode, 7)
            self.assertEqual(len(trace.read_text().splitlines()), 1)

    def test_author_brain_stage_is_rejected(self):
        result = subprocess.run(["bash", str(ROOT / "scripts/run_synbrain_comparison.sh"), "author", "brain"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
