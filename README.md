# IPABrain

IPABrain is a perception-anchored probabilistic alignment framework for reconstructing images from visual perception and mental imagery fMRI. It learns subject-specific brain representations from perception data, adapts imagery responses to the frozen perception encoder, and reconstructs images with visual and text conditioning.

## Method

| Module | Implementation |
| --- | --- |
| **PPBE** · Perceptual Probabilistic Brain Encoder | [SynBrain BrainVAE](src/models/brainvae_synbrain.py), [perception training](scripts/train_brainvae_scheme1.py), [training configuration](configs/brainvae_synbrain.yaml) |
| **IVA** · Imagery Voxel Adapter | [Affine residual adapter](src/models/imagery_residual_comparison.py), [perception anchor aggregation](src/utils/perception_anchor_prior.py), [subject-level training and inference](output_1104/scripts/run_multisubject_linear_imagery.py) |
| **VSIR** · Visual-Semantic Initial Reconstruction | [AdaToken mapper](src/models/clip_mapping.py), [mapping training](scripts/train_clip_mapping.py), [SDXL-UnCLIP reconstruction](scripts/decode_clip_mapping.py) |
| **TGIR** · Text-Guided Image Refinement | [Three-branch Ridge fitting](output_1104/scripts/ridge_multisubject_three_text.py), [text prediction](output_1104/scripts/predict_multisubject_ridge.py), [SDXL image-to-image refinement](output_1104/scripts/refine_multisubject_vision_imagery_three.py) |

**Visual path.** PPBE estimates a diagonal Gaussian posterior from fMRI. For imagery, IVA applies a subject-specific affine residual transformation to the averaged response. The frozen PPBE posterior mean is mapped by the w010 visual head and passed to SDXL-UnCLIP. The main adapter variant is `mu_mse`. B0 uses the frozen perception backbone directly, and `w010_mse` applies the main regression loss at the visual head output.

**Text path.** Three Ridge regressors predict CLIP-L sequence features, OpenCLIP bigG sequence features, and the bigG pooled feature from the PPBE posterior mean of the unadapted response. The predicted text features condition ZavyChromaXL image-to-image refinement of the initial reconstruction. The text extraction and Ridge training code uses NSD COCO captions.

## Environment and local inputs

Use Linux with Python 3.10 or newer and a CUDA-enabled PyTorch and torchvision installation matched to your GPU. For the SDXL generation configurations, install an xFormers build compatible with that PyTorch and CUDA environment. The training queues use POSIX file locks. Install the remaining Python dependencies from [requirements.txt](requirements.txt). Run the examples from the repository root.

```bash
python -m pip install -r requirements.txt
```

Set the raw and processed NSD and NSD-Imagery paths in [configs/paths.example.yaml](configs/paths.example.yaml). Prepare the local NSD fMRI arrays and CLIP features using [prepare_nsd_perception_fmri.py](scripts/prepare_nsd_perception_fmri.py) and [extract_nsd_sdxl_clip_vision.py](scripts/extract_nsd_sdxl_clip_vision.py). The NSD preprocessing entry point takes the COCO annotation array through `--annots`. Set `raw_nsd_root` to the directory containing `nsddata/` and `nsddata_betas/`.

Put the needed local checkpoints under `huggingface/` or use the path overrides supported by the corresponding entry points. The visual path requires MindEye2 SDXL-UnCLIP and OpenCLIP ViT-bigG/14. TGIR additionally uses a local `clip-vit-large/` tokenizer and text model directory and the `zavychromaxl_v30.safetensors` weights. Use the checkpoint versions expected by the protocol checks in the scripts.

## Training and reconstruction entry points

### Perception backbone

With the local perception arrays and feature caches in place, prepare the subject data and train PPBE. The same configuration controls both steps.

```bash
python scripts/prepare_scheme1_data.py --config configs/brainvae_scheme1_synbrain.yaml --subject 1 --reuse
python scripts/train_brainvae_scheme1.py --config configs/brainvae_scheme1_synbrain.yaml --subject 1 --device cuda
```

[run_w010_multisubject.py](scripts/run_w010_multisubject.py) records the subject-specific AdaToken base training, w010 fine-tuning, checkpoint selection and NSD reconstruction workflow. Its [suite configuration](configs/w010_multisubject_suite.yaml) references the subject-specific perception configurations and checkpoints. For an individual run, use [train_clip_mapping.py](scripts/train_clip_mapping.py) and [decode_clip_mapping.py](scripts/decode_clip_mapping.py) with the corresponding mapping configuration and checkpoint.

### Imagery adaptation and text refinement

1. The [subject worker](output_1104/scripts/run_multisubject_linear_imagery.py) implements `prepare`, `retrieval`, `teachers`, `prior`, `train`, `predict` and `generate`. Use `--method mu_mse` for the main method's training, prediction and generation. The imagery evaluation averages four held-out fMRI trials per stimulus. The target stimuli are represented in the adapter training partition.
2. [extract_sdxl_clip_l11.py](output_1104/scripts/extract_sdxl_clip_l11.py) and [extract_multisubject_nsd_text_resumable.py](output_1104/scripts/extract_multisubject_nsd_text_resumable.py) prepare text supervision. [ridge_three_text.py](output_1104/scripts/ridge_three_text.py) fits the subject 01 text model. [ridge_multisubject_three_text.py](output_1104/scripts/ridge_multisubject_three_text.py) fits the models for subjects 02–08. Run [predict_multisubject_ridge.py](output_1104/scripts/predict_multisubject_ridge.py) and [refine_multisubject_vision_imagery_three.py](output_1104/scripts/refine_multisubject_vision_imagery_three.py) with the predicted text features and initial images.
3. The [adapter queue](output_1104/scripts/queue_multisubject_linear_imagery.py) and [B0 queue](output_1104/scripts/queue_multisubject_nsd_imagery.py) use the original experiment directory layout. Their preflight checks expect the locally prepared trial manifests, generated per-subject YAML files, compatible checkpoint and model directories, and provenance records. Generate these artifacts on your system. A fresh clone provides the implementation and configuration templates.

```bash
python output_1104/scripts/run_multisubject_linear_imagery.py --help
python output_1104/scripts/refine_multisubject_vision_imagery_three.py --help
```

## Image reconstruction evaluation

Use [eval_scheme1_reconstruction.py](scripts/eval_scheme1_reconstruction.py) on a reconstruction directory containing `metadata.json` and one `sampleXX/manifest.json` per diffusion sample. The evaluator matches the target identities across samples, scores each sample, and averages the per-sample scores. It writes `image_metrics.json` with the individual and aggregate scores.

```bash
python scripts/eval_scheme1_reconstruction.py --dataset-dir path/to/reconstruction --device cpu
python scripts/eval_scheme1_reconstruction.py --dataset-dir path/to/reconstruction --device cuda \
  --models Alex2,Alex5,Incep,CLIP,Eff,SwAV \
  --clip-ckpt huggingface/ViT-L-14.pt \
  --swav-ckpt huggingface/swav_800ep_pretrain.pth.tar
```

The first command computes PixCorr and SSIM. The full evaluation adds two-way identification with AlexNet layers 2 and 5, Inception and CLIP, plus EfficientNet-B1 and SwAV correlation distance. The torchvision models use their standard pretrained weights.

## Upstream code

The code uses components from [SynBrain](https://github.com/MichaelMaiii/SynBrain), [SiT](https://github.com/willisma/SiT), [Latent Diffusion](https://github.com/CompVis/latent-diffusion), [Stability AI generative-models](https://github.com/Stability-AI/generative-models), and [NeuroFlow](https://github.com/MichaelMaiii/NeuroFlow). Their license notices accompany the corresponding source files. The SDXL-UnCLIP generation path follows [MindEye2](https://github.com/MedARC-AI/MindEyeV2).
