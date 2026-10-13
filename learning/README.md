# Learned contour guide training

This is a separate Python 3.12+ environment, outside QGIS. Importing `learning`
loads neither PyTorch nor an inference session and never downloads weights.
The model is a ResNet18 encoder and a small SMP U-Net decoder with two logits:
`contour_support` and `other_ink_risk`. RGB input is divided by 255 by the caller;
ImageNet mean/std normalization belongs to the model and exported ONNX graph.

## Prepare and review the pilot dataset

The prepared public pilot contains 16 frozen USGS sheets and 144 lossless native
512px PNG crops. Their initial annotations are drafts and cannot start training.
Use an existing prepared folder, or explicitly create a new one with the
versioned source-selection plan:

```bash
python -m learning prepare --output /absolute/new-pilot-data
python -m learning validate /absolute/new-pilot-data/manifest.json
python -m learning verify-pixels /absolute/new-pilot-data/manifest.json
```

Preparation and pixel verification need the separate data/evaluation environment
(`requirements-dev.txt`, including rasterio). `verify-pixels` proves decoded
PNG pixels equal their declared native source windows; it does not approve
annotations or establish contour accuracy. Split/family assignment stays frozen.

In QGIS, open **Advanced / Legacy methods → Annotate training crop** and select
one of the 144 PNGs. The annotation project creates `contours`, `other_ink`,
`other_ink_regions`, `reviewed` and `ignore` layers. Correct the initial Ink draft,
draw every visible contour within the reviewed areas, identify noncontour lines
or regions, and exclude uncertain pixels with ignore polygons. Save these layer
edits before **Review / export labels**. Approval requires a real reviewer ID and
an explicit confirmation of all visible contours; the annotator and reviewer
may be the same person at the declared single-reviewer pilot level. The draft
algorithm never supplies reviewer approval.

Opening a prepared crop restores its manifest association; export can update
that association with a new annotation revision. To attach an independently
exported revision explicitly, create a new manifest beside the existing one:

```bash
python -m learning attach /absolute/pilot-data/manifest.json \
  --sample SAMPLE_ID --annotation /absolute/reviewed-labels.json \
  --output /absolute/pilot-data/manifest-reviewed.json
python -m learning difficulty /absolute/pilot-data/manifest-reviewed.json \
  SAMPLE_ID hard --output /absolute/pilot-data/manifest-classified.json
python -m learning validate /absolute/pilot-data/manifest-classified.json --require-approved
python -m learning archive /absolute/pilot-data/manifest-classified.json \
  --output /absolute/approved-dataset.zip
```

Replace `SAMPLE_ID` with the selected PNG's manifest sample ID. Approval is
required for every declared pilot sample. Explicitly classify the 48 calibration
and holdout crops as `standard` or `hard` before training. Training and validation
crops may remain `unassessed`. The difficulty command changes a
metadata/annotation revision and never supplies approval.
Keep native rasters in
the default archive for later source-context evaluation. A `--crops-only` archive
can train but cannot perform that evaluation. For calibration/holdout reference
routes, explicitly select one reviewed open contour as the evaluation target;
its two endpoints must fit one 320px Live-Wire window. No longest-path or target
selection is inferred automatically.

Before pilot training, freeze all 144 annotation revisions, including
calibration/holdout labels, their explicit reference targets and predeclared
difficulty flags, in that same final manifest. Their protected files stay unopened by the
train/validation loader. Attaching later revisions changes the full manifest
SHA and therefore requires a new experiment/model lineage; a trained bundle
cannot silently adopt newly edited evaluation labels or difficulty categories.

## Train and resume

Install `requirements-learning.txt` in an isolated environment. Prepare the
manifest and human-approved annotations using the dataset tools, then run:

```bash
python -m learning.train /absolute/data/manifest.json /absolute/runs/seed-3407 \
  --seed 3407 --device cuda --initialize-pretrained
```

The explicit initialization flag obtains the official torchvision ImageNet
ResNet18 weights and records their SHA-256. `--encoder-weights /absolute/weights.pth`
uses an explicitly selected local encoder checkpoint with the same flag.
Training and validation must be approved, rights-checked selections from one
manifest. Training never constructs a calibration or holdout dataset. One real
reviewer is accepted as `single_reviewer_pilot`, with publication ranking
eligibility false; generated labels cannot approve themselves.

Defaults are seeds 3407/3408/3409 when running the notebook, AdamW 1e-4,
60 epochs maximum, patience 8, batch 2 and accumulation 8. Masks exclude ignored
pixels from BCE/Dice. The centreline head also uses 0.2 soft-clDice, with an
ignore-adjacent margin excluded from topology scoring. Augmentation affects
training only and uses sample/epoch/seed-specific RNG for rotations, flips,
brightness, contrast and noise.

`latest.pt` is atomically replaced every epoch and at the first complete optimizer
boundary after five minutes. It stores model, optimizer, AMP scaler, exact RNG,
epoch permutation, sampler position and partial epoch metrics. No pending
gradients are published. Resume requires the same training bytes, runtime,
manifest, sample selection and configuration:

```bash
python -m learning.train /absolute/data/manifest.json /absolute/runs/seed-3407 \
  --device cuda --resume /absolute/runs/seed-3407/latest.pt
```

If CUDA runs out of memory, the last durable checkpoint remains usable. A
deliberate `--retry-oom` resume changes only batch to 1 and accumulation to 16;
work since that checkpoint is repeated. It does not relabel a partial failed
step as saved progress. CPU diagnostics and CUDA/AMP runs have separate recorded
environments; bitwise resume is not promised across different environments.

Validation alone selects each seed's `best.pt`. Freeze the three-seed comparison
in `selected-model.json` using the notebook, then export its selected checkpoint:

```bash
python -m learning.export /absolute/runs/seed-3407/best.pt \
  /absolute/new-bundle /absolute/data/manifest.json --license GPL-2.0-only \
  --selection-file /absolute/runs/selected-model.json
```

Export creates one FP32 `model.onnx` and `manifest.json` in an absent directory.
The fixed interface is `rgb: [1,3,512,512]` to `logits: [1,2,512,512]`. CPU
PyTorch/ONNX probability parity must stay within 1e-4 on deterministic diagnostic
inputs. The manifest binds checkpoint, config, dataset, code, environment,
initialization and model hashes. Export checks all three actual checkpoint files
against the frozen validation results and binds the selection content/file hashes
into provenance. Real pilot export requires that proof. Fixed source-aligned 384 cores plus 64 halos
provide consistent pan context, not whole-image equivalence.

The Colab notebook mounts the user's Drive, verifies a full Git commit, safely
extracts an explicitly selected approved dataset ZIP and resumes three seeds.
It freezes exactly one candidate in `selected-model.json` by the lowest best
validation loss (ties use ascending seed), then exports only that candidate.
The selection binds checkpoint, config, dataset and code before calibration or
holdout access. It needs an already available CUDA GPU
runtime. The notebook neither purchases GPU access nor executes a remote
inference service. Actual Colab GPU execution remains a separate user run.

## QGIS runtime and isolated evaluation

QGIS needs only the optional local ONNX runtime, not PyTorch or SMP. Install
`requirements-learned-guide.txt` using the chosen QGIS Python when explicitly
enabling learned guidance. Older QGIS Python keeps the default Ink route.
Import the selected local bundle with **Import Learned Guide**, then explicitly
enable the learned-guidance checkbox. The local bundle is hash-verified; no
map crops or geometries are sent to an inference server. Inspect every green
preview and accept its points yourself. A missing or failed runtime keeps Ink.

After validation-only model selection, run the separate CPU evaluation protocol.
It uses final ordered, smoothed routes and source-anchored context, with one
thread per inference runtime. Freeze calibration before opening holdout results:

```bash
python -m learning evaluate calibrate /absolute/data/manifest.json \
  /absolute/selected-bundle /absolute/new-calibration --repetitions 3
python -m learning evaluate holdout /absolute/data/manifest.json \
  /absolute/selected-bundle /absolute/new-holdout \
  --calibration /absolute/new-calibration/calibration.json \
  --recovery-cache /absolute/verified-recovery-cache --repetitions 3 \
  --seal-output /absolute/new-sealed-bundle
```

Holdout compares Ink, Ink+Recovery and learned guidance under the same route
configuration. Synthetic models are refused. Sealing publishes a new lineage
bundle only when the real reviewed holdout gates pass; failed or incomplete
gates remain failed. A USGS pilot result does not establish Korean historical-map
generalization or independent publication review. Existing evaluation directories
and frozen calibration are not overwritten.

Synthetic training/export diagnostics use the explicit `synthetic_diagnostic`
API option and require `--synthetic-diagnostic` on export. Such bundles carry
`research_ready=false`, are not domain-trained models, and cannot support
accuracy or publication claims. Calibration, route evaluation and holdout
decisions belong to the separate evaluation workflow.
