# CS 771 Fall 2026 — Homework 2: Image Classification & Object Detection

PyTorch implementation for **BMI/CS 771 HW2**. Two parts: (I) image classification on
MiniPlaces, and (II) object detection (FCOS) on PASCAL VOC 2007. The starter scaffolding
(data pipelines, training loops, configs, eval scripts) is provided; our job is to fill in
the missing model code, train, record results, and write the report.

- **Points:** 12 + 4 bonus
- **Team:** Ashvin, Akshat, Shubhangi
- **Plan & task assignment:** see [PLAN.md](PLAN.md)

---

## 1. What needs to be done

All implementation happens in **two files**. Everything else is provided.

### Part I — Image Classification (`code/cls_libs/modeling.py`)

| # | Where | Pts | Task |
|---|-------|-----|------|
| I.2 | `SimpleNet` / `default_cnn_model` | 2 | Design an improved CNN (BatchNorm, skip connections, more depth) to beat the ~40% `SimpleNet` baseline |
| I.3a | `Attention.forward` | 1 | Multi-head self-attention: `softmax(QKᵀ/√d)·V`, reshape, apply `self.proj` |
| I.3b | `TransformerBlock.forward` | 1 | Pre-LN block with **local window attention** (`window_partition`/`window_unpartition`) + residual / drop-path / MLP |
| I.4 | `SimpleViT.__init__` / `forward` | 2 | Stack transformer blocks (`window_block_indexes`), pos-embed, patch-embed, final norm + classifier |
| — | Writeup | 1 | Compare best CNN vs `SimpleNet` vs fine-tuned ResNet-18 |

### Part II — Object Detection (`code/det_libs/modeling.py`)

| # | Where | Pts | Task |
|---|-------|-----|------|
| II.1 | `FCOS.inference` | 2 | Per-level decode → score threshold → top-K → decode boxes (**× stride**) → clip → per-class NMS → keep top-K; labels **+1**. Target ≈ **60.9% mAP@0.5** on the provided pretrained model |
| II.2 | `FCOS.compute_loss` | 3 | Center-sampling label assignment, reg targets **÷ stride**, centerness, 3 losses (focal / GIoU / BCE) normalized by #positives. Must run **< 30 min / 10 epochs on a T4** |

### Bonus (optional, +4)
- FCOS with a **Swin Transformer** backbone (+2)
- Training & inference on **COCO** (+2)

> ⚠️ The starter code **will not run until the stubs above are filled in** — each stubbed
> method currently returns an undefined/unmodified value.

---

## 2. Repository structure

```
.
├── code/
│   ├── cls_libs/        # classification: dataset, engine, blocks, modeling (EDIT modeling.py)
│   ├── det_libs/        # detection: dataset, engine, losses, point_generator, modeling (EDIT modeling.py)
│   ├── configs/         # YAML configs (cls_miniplaces_*.yaml, det_voc_fcos.yaml)
│   ├── train_cls.py  eval_cls.py
│   └── train_det.py  eval_det.py
├── data/                # datasets live here (downloaded; NOT committed except starter JSON)
├── pretrained/          # det_voc_res18.yaml committed; .pth.tar is gitignored (see §3.4)
├── results/             # final configs/logs/checkpoints to submit (copy here at the end)
├── writeup/             # project report PDF
├── download_cls_dataset.sh  download_det_dataset.sh
├── zip_submission.py    # builds the Canvas submission archive
├── readme.pdf           # the original assignment handout
├── PLAN.md              # 3-day team plan + task assignment
└── README.md            # this file
```

---

## 3. Setup (detailed)

### 3.1 Clone

```bash
git clone https://github.com/akshatsinghal92/cs771-fall2026-hw2.git
cd cs771-fall2026-hw2
```

### 3.2 Python environment

Use any of `venv` / `conda` / `uv`. Example with `venv` (Python 3.10+ recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### 3.3 Install dependencies

Install a **GPU build of PyTorch** matching your CUDA version (see https://pytorch.org for
the exact command for your machine), then the rest:

```bash
# PyTorch >= 2.0 with GPU support — pick the right index for your CUDA, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# remaining requirements
pip install "opencv-python>=4" numpy gdown pycocotools pyyaml pillow tensorboard
```

Required packages (from the handout): **PyTorch ≥ 2.0 (GPU), OpenCV ≥ 3, NumPy, gdown,
torchvision, pycocotools, PyYAML, Pillow, TensorBoard**. Record any *extra* packages you add
— they must be listed in the writeup.

> macOS: you can develop/debug on CPU or the **MPS** backend (set `devices: ["mps"]` in the
> config). GPU is required for real training.

### 3.4 Datasets & pretrained model

Download datasets into `data/` using the provided scripts (they need `gdown` etc. installed):

```bash
sh download_cls_dataset.sh     # MiniPlaces  -> data/images, data/objects, train.txt, val.txt
sh download_det_dataset.sh     # PASCAL VOC  -> data/VOCdevkit, VOC*tar
```

Datasets are **gitignored** — each teammate downloads their own; do not commit them.

**Pretrained FCOS model (needed for the II.1 inference test):**
`pretrained/det_voc_res18.pth.tar` is **not** stored in git (it's a 52 MB binary, gitignored).
Get it from the original assignment release (`assignment2_release/pretrained/`) or ask Akshat
to share it, and place it at `pretrained/det_voc_res18.pth.tar`.

---

## 4. Running the code

Run all commands from the **`code/`** directory.

```bash
cd code
```

### 4.1 Classification

```bash
# usage
python train_cls.py --help
python eval_cls.py  --help

# train (config selects the model via model_type / its own file)
python train_cls.py configs/cls_miniplaces_cnn.yaml          # provided SimpleNet baseline (~40%)
python train_cls.py configs/cls_miniplaces_resnet18.yaml     # fine-tune ImageNet ResNet-18
python train_cls.py configs/cls_miniplaces_vit.yaml          # SimpleViT (AdamW)

# resume from an epoch checkpoint
python train_cls.py configs/cls_miniplaces_cnn.yaml --resume ../logs/<exp>/epoch_xxx.pth.tar

# evaluate a trained model (pass the experiment folder or a checkpoint)
python eval_cls.py configs/cls_miniplaces_cnn.yaml ../logs/<exp>/
```

### 4.2 Detection

```bash
# usage
python train_det.py --help
python eval_det.py  --help

# verify FCOS.inference against the provided pretrained model (no training needed)
python eval_det.py ../pretrained/det_voc_res18.yaml ../pretrained/det_voc_res18.pth.tar
#   -> expect ~60.9% mAP@0.5

# train your own FCOS
python train_det.py configs/det_voc_fcos.yaml

# evaluate (add -v to save detection visualizations in <exp>/viz/)
python eval_det.py configs/det_voc_fcos.yaml ../logs/<exp>/ -v
```

### 4.3 Config overrides

Defaults live in `code/{cls,det}_libs/config.py`; override them in the YAML without editing
source. Example overrides for a classification config:

```yaml
opt:
  epochs: 80          # change number of epochs
devices:
  - "cpu"             # or "mps" on mac, or "cuda:0" (default)
```

### 4.4 Monitoring

```bash
watch -n 0.1 nvidia-smi                 # GPU utilization/memory
tensorboard --logdir ../logs/<exp>      # training curves
```

Run long jobs inside **`tmux`** so they survive SSH drops. Copy `../logs/<exp>` to your laptop
to view TensorBoard locally. **Shut down the cloud instance when idle.**

---

## 5. Git workflow (branch protection is ON)

`main` is protected — **no direct pushes**. Every change goes through a PR:

1. Branch off `main`: `git checkout -b feat/<thing>` (e.g. `feat/fcos-loss`, `feat/cnn`, `feat/vit`).
2. Commit and push the branch: `git push -u origin feat/<thing>`.
3. Open a PR: `gh pr create --fill` (or via the GitHub UI).
4. **Get one of the other two members to approve.** You cannot approve your own PR, and the
   rule applies to everyone (admins included). New commits after approval dismiss the approval.
5. Resolve conversations, then merge.

Do **not** commit `data/`, `logs/`, `models/`, or loose `*.pth`/`*.pth.tar` (all gitignored).
**Exception:** everything under `results/` *is* tracked — that's the submission payload
(configs, training logs, and **final** `model_final.*` checkpoints). Only place final
checkpoints there; intermediate `epoch_*` checkpoints are auto-ignored even inside `results/`.

---

## 6. Submission

1. Copy required **configs, training logs, and final `model_final.pth.tar` checkpoints** into
   `results/` (not `data/`, `logs/`, `models/`, or intermediate epoch checkpoints).
2. Put the report PDF in `writeup/`.
3. Make sure all code uses **relative paths**.
4. Build the archive and submit on Canvas:
   ```bash
   python zip_submission.py
   ```
5. **All three members submit the same PDF.** The writeup must clearly identify each member's
   contributions.

See [readme.pdf](readme.pdf) for the full official handout and grading criteria.
