# Assignment 2 — 3-Day Team Plan

**Course:** BMI/CS 771 Fall 2026, Homework 2 — Image Classification + Object Detection (PyTorch)
**Team size:** 3 — **Deadline:** 3 days
**Total:** 12 pts + 4 bonus

Starter scaffolding (data pipelines, training loops, configs, eval scripts) is provided. The work is to fill in missing model code in two files, train, record results, and write the PDF report.

- `code/cls_libs/modeling.py` — image classification (MiniPlaces, 100 classes)
- `code/det_libs/modeling.py` — object detection (PASCAL VOC 2007, FCOS)

## What needs to be implemented

| # | File / location | Pts | Task |
|---|-----------------|-----|------|
| I.2 | `cls_libs/modeling.py` `SimpleNet`/`default_cnn_model` | 2 | Design an improved CNN (BatchNorm, skip connections, more depth) to beat the ~40% SimpleNet baseline |
| I.3a | `cls_libs/modeling.py` `Attention.forward` | 1 | Multi-head self-attention: `softmax(QKᵀ/√d)V`, reshape, `self.proj` |
| I.3b | `cls_libs/modeling.py` `TransformerBlock.forward` | 1 | Pre-LN block with local window attention (`window_partition`/`window_unpartition`) + residual/drop-path/MLP |
| I.4 | `cls_libs/modeling.py` `SimpleViT.__init__`/`forward` | 2 | Stack transformer blocks (`window_block_indexes`), pos-embed, patch-embed, final norm + classifier |
| — | Compare & Discuss (writeup) | 1 | Best CNN vs SimpleNet vs fine-tuned ResNet-18 |
| II.1 | `det_libs/modeling.py` `FCOS.inference` | 2 | Per-level decode, score threshold, top-K, decode boxes (×stride), clip, per-class NMS, keep top-K; labels +1. Target ≈60.9% mAP@0.5 on provided pretrained model |
| II.2 | `det_libs/modeling.py` `FCOS.compute_loss` | 3 | Center-sampling label assignment, reg targets ÷stride, centerness, 3 losses (focal/GIoU/BCE) normalized by #positives; must run <30 min / 10 epochs on T4 |
| Bonus | FCOS + Swin backbone | +2 | Only after core 12 pts are locked |
| Bonus | COCO training/inference | +2 | Only after core 12 pts are locked |

## Role split

- **Ashvin — Classification/CNN:** custom CNN (I.2) + ResNet-18 fine-tune runs + classification writeup.
- **Shubhangi — Classification/ViT:** Attention (I.3a) → TransformerBlock (I.3b) → SimpleViT (I.4) + ViT writeup.
- **Akshat — Detection:** FCOS inference (II.1) → compute_loss (II.2) + detection writeup. Also owns repo/infra setup.

Tracks are independent code-wise. Shubhangi's three pieces are strictly sequential. Akshat's `inference` is self-verifiable against the provided pretrained model (early correctness signal). Owners can swap tracks if preferred — update this section if you do.

## Task dependencies

**All three tracks are independent — nobody is blocked waiting on another person's code.** Everyone can start and finish their implementation in parallel from day one.

| Track | File edited | Depends on another member? |
|-------|-------------|----------------------------|
| Ashvin (CNN) | `code/cls_libs/modeling.py` — `SimpleNet`/`default_cnn_model` | No |
| Shubhangi (ViT) | `code/cls_libs/modeling.py` — `Attention`, `TransformerBlock`, `SimpleViT` | No |
| Akshat (Detection) | `code/det_libs/modeling.py` — `inference`, `compute_loss` | No |

- Detection uses a **ResNet** backbone, not the ViT `Attention` class — Akshat does **not** depend on Shubhangi. (That link only appears in the *bonus* Swin backbone, which has its own attention.)
- Ashvin's CNN stands alone.

**Within-person dependency (not cross-person):** Shubhangi's track is strictly sequential — `Attention` → `TransformerBlock` → `SimpleViT` (each uses the previous). This chain lives entirely inside her tasks. Combined with ViT being the slowest training run, she is the **longest critical path** and most likely schedule bottleneck — protect her timeline.

**Couplings that need coordination (not task dependencies):**
1. **Ashvin and Shubhangi edit the same file** (`cls_libs/modeling.py`) but different classes — no functional dependency, just a git merge concern. Use separate branches (`feat/cnn`, `feat/vit`) and separate PRs; non-overlapping classes mean conflicts should be rare/trivial. Akshat's file is entirely separate.
2. **GPU sharing** (if fewer than 3 instances) and the **Day-3 writeup assembly** are the only real convergence points — scheduling, not code.

---

## DAY 1 — Setup + core implementation

**Morning (all together)**
1. Set up cloud GPU instances; install deps (PyTorch ≥2.0+GPU, OpenCV, torchvision, pycocotools, PyYAML, gdown, Pillow, TensorBoard). Record extra packages for writeup.
2. `sh download_cls_dataset.sh` and `sh download_det_dataset.sh` → `./data/`. Verify contents.
3. **Ashvin:** run provided SimpleNet baseline to confirm pipeline (~40%).
4. All: read helper code for your track.

**Afternoon/evening (parallel)**
- **Akshat (priority):** implement `FCOS.inference`; verify `eval_det.py ../pretrained/det_voc_res18.yaml ../pretrained/det_voc_res18.pth.tar` → ≈60.9% mAP. Correctness gate.
- **Shubhangi:** implement `Attention.forward` + `TransformerBlock.forward`; shape-test on dummy tensor.
- **Ashvin:** design improved CNN; smoke-test 1 epoch.

**End-of-Day-1 checkpoint — launch overnight runs:**
- **Ashvin:** full custom CNN training.
- **Ashvin:** ResNet-18 fine-tune (`cls_miniplaces_resnet18.yaml`).
- Use tmux; confirm GPU active with `watch -n 0.1 nvidia-smi`.

## DAY 2 — Finish implementation, launch slow runs

**Morning (parallel)**
- **Akshat:** implement `FCOS.compute_loss`; verify loss decreases + 10 epochs <30 min on T4.
- **Shubhangi:** implement `SimpleViT`; smoke-test 1 epoch.
- **Ashvin:** review overnight CNN + ResNet-18 in TensorBoard; iterate/relaunch if CNN underperforms; copy logs locally.

**Afternoon → overnight**
- **Akshat:** full FCOS training (`det_voc_fcos.yaml`); eval with `-v` for detection visualizations.
- **Shubhangi:** full ViT training (`cls_miniplaces_vit.yaml`) — slowest run, let it run overnight.
- **Ashvin:** relaunch better CNN config if needed.

**End-of-Day-2 checkpoint:** all 4 code pieces done + verified; all training done or running. Start writeup drafts.

## DAY 3 — Results, writeup, submit

**Morning**
- All: pull final checkpoints + logs; generate TensorBoard curves.
- **Akshat:** final `eval_det.py -v`; record mAP + inference time + sample detections.
- **Ashvin:** record top-1/top-5 for custom CNN, SimpleNet, ResNet-18 → comparison table.
- **Shubhangi:** record ViT top-1/top-5.
- Last window to relaunch a short run if anything is broken.

**Afternoon — writeup (each owns a section; one assembles)**
Required: CNN design/scheme/results + config/logs/checkpoint; self-attention explanation; ViT design/scheme/results + config/logs/checkpoint; CNN vs SimpleNet vs ResNet-18 comparison with curves; FCOS mAP/inference time/sample detections/curves/implementation details + config/logs/checkpoint; list of extra packages; **clearly identify each member's contributions**.

**Late afternoon — package & submit**
- Copy required logs + final `model_final.pth.tar` checkpoints into `results/` (NOT `data/`, `logs/`, `models/`, or intermediate checkpoints).
- Verify relative paths everywhere.
- `python zip_submission.py`.
- **All 3 members submit the same PDF** to Canvas.

## Risk notes
- Classification training is the long pole — protect both overnight windows.
- Do `FCOS.inference` first (only piece verifiable without training).
- Shut down idle GPU instances between sessions.
- Target all code done by end of Day 2; keep Day 3 for results + writeup. `compute_loss` is the task most likely to slip.

## Git workflow
- `main` is protected: all work via feature branches + PR with ≥1 review.
- Suggested branch names: `feat/cnn`, `feat/vit`, `feat/attention`, `feat/fcos-inference`, `feat/fcos-loss`, `docs/writeup`.
