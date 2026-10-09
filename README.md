# Collaborative Knowledge Distillation (CKD) — PyTorch Reproduction

Unofficial PyTorch reproduction of:

> Weiwei Zhang, Yufeng Guo, Junhuang Wang, Jianqing Zhu, Huanqiang Zeng,
> "Collaborative Knowledge Distillation",
> *IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)*,
> vol. 34, no. 8, pp. 7601–7613, 2024. DOI: 10.1109/TCSVT.2024.3377251

The paper does not release official code. This repository reimplements the
method from the paper text; every setting that the paper leaves unspecified
is documented as a *documented inference* in the "Reproduction decisions"
section below. The model zoo, data pipeline and teacher checkpoints are
compatible with RepDistiller (HobbitLong/RepDistiller), which is the
benchmark the paper's Table IX/X directly builds on (its baseline and KD
columns match RepDistiller's published numbers exactly).

## Method

CKD trains a frozen teacher `T`, a student `S`, and a small trainable
**feature fusion module** `F` together. `F` splices the teacher and student
penultimate-layer features and predicts through its own classifier:

```
Z^F = F([F_L^T ; F_L^S]),   P^F = softmax(Z^F / tau)      (Eq. 5, tau omitted in paper notation)
```

Training losses (Eqs. 6–8):

```
Loss_F = alpha * KL(P^T || P^F) + beta * KL(P^S || P^F) + lambda * CE(Y, P^F)   (Eq. 6)
Loss_S = gamma * KL(P^F || P^S) + eta * CE(Y, P^S)                             (Eq. 7)
Loss   = Loss_F + Loss_S                                                       (Eq. 8)
```

The student is updated by both losses jointly (the `KL(P^S||P^F)` term
propagates gradients back into `S` through `P^S`); at inference the fusion
module is discarded and the student architecture is unchanged.

The paper states `tau = 1` for the cross-entropy terms; the temperature used
for the softmax/KL terms is not given numerically (see decisions below).

## Repository layout

```
models/                 RepDistiller-compatible CIFAR/ImageNet model zoo
  resnet.py             ResNetV1 (resnet8..resnet110, resnet32x4, resnet8x4)
  resnetv2.py           ResNet50
  wrn.py                Wide ResNet (wrn_40_2, wrn_40_1, wrn_16_2, ...)
  vgg.py                VGG8/11/13/16/19 (BN variants)
  imagenet_resnet.py    torchvision ResNet34/18 wrappers (Table X)
dataset/                CIFAR-10/100 + ImageNet loaders (paper transforms)
fuse.py                 FeatureFusionModule (Eq. 5)
distiller_zoo/
  KD.py                 classic KD baseline (DistillKL, T^2-scaled, T=4)
  ckd.py                CKDLoss (Eqs. 6-8), Exp1Loss (Eqs. 9-10), Exp2Loss (Eqs. 11-12)
helper/                 training loops, meters, LR schedule
train_teacher.py        vanilla teacher training (paper protocol)
train_student.py        distillation entry point (default --method ckd)
scripts/                experiment launchers (Table IX pairs, ImageNet, smoke test)
```

## Requirements

```
pip install torch torchvision
```

(torch >= 2.0 tested; CPU-only machines work for smoke tests.)

## Data & pretrained teachers

```
# CIFAR-100/10 download automatically on first run into ./data/
sh scripts/fetch_pretrained_teachers.sh   # official RepDistiller vanilla ckpts
```

Teacher accuracies of the official checkpoints match the paper's Table IX
teacher row: wrn_40_2 75.61, resnet56 72.34, resnet110 74.31, resnet32x4
79.42, vgg13 74.64.

## Usage

Vanilla teacher training (if you prefer training teachers yourself):

```
python train_teacher.py --model wrn_40_2 --dataset cifar100 --trial 1
```

Protocol: SGD momentum 0.9, weight decay 5e-4, lr 0.1 divided by 10 at
epochs 150/180/210, 240 epochs, batch 64 (paper Sec. IV). Note the official
RepDistiller ckpts were trained with lr 0.05; pass `--learning_rate 0.05` to
match that recipe exactly.

Distillation (CKD is the default method):

```
python train_student.py \
    --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth \
    --model_s wrn_16_2 --method ckd --trial 1
```

Key flags of `train_student.py`:

| flag | default | meaning |
| --- | --- | --- |
| `--method` | `ckd` | `ckd` (Eqs. 6-8), `kd` (classic KD baseline), `exp1` / `exp2` (Appendix A) |
| `--alpha --beta --lam --gamma --eta` | 1.0 each | loss weights of Eqs. 6-7 |
| `--kd_T` | 4.0 | temperature `tau` for softmax/KL terms |
| `--kd_t2` | 1 | scale KL terms by `tau^2` (set 0 to disable) |
| `--fusion_dim` | `teacher` | fusion hidden width: `teacher` / `student` / integer |
| `--fusion_arch` | `linear` | fusion head: `linear` / `mlp2` / `vggcls` |
| `--val_fusion` | 1 | log per-epoch test accuracy of the fusion module (diagnostic: is P^F better than the teacher?) |
| `--alpha1 --beta` | 1.0 | Exp1 loss weights (Eq. 10) |
| `--gamma --delta` | 1.0 | Exp2 loss weights (Eq. 12) |
| `--exp1_literal` | 0 | 0: FitNet-style `||F^S - F^T||_2`; 1: literal printed Eq. 10 |
| `--subset` | 1.0 | fraction of the train set (for smoke tests) |

Per-epoch logs include the individual loss components (`loss_f`, `loss_s`,
`kl_tf`, `kl_sf`, `kl_fs`, `ce_f`, `ce_s`) and training accuracies of both
the student (`train_acc_s`) and the fusion module (`train_acc_f`).

Quick smoke test:

```
sh scripts/run_smoke_test.sh
```

## Expected results (paper)

CIFAR-100 test accuracy (%) — Table IX, teacher / baseline / KD / DKD / CKD:

| teacher → student | baseline | KD | DKD | CKD (paper) |
| --- | --- | --- | --- | --- |
| WRN-40-2 → WRN-16-2 | 73.26 | 74.92 | 76.23 | **76.29** |
| WRN-40-2 → WRN-40-1 | 71.98 | 73.54 | 74.81 | **74.85** |
| ResNet56 → ResNet20 | 69.06 | 70.66 | 71.97 | **72.03** |
| ResNet110 → ResNet32 | 71.14 | 73.08 | 74.11 | **74.14** |
| ResNet32x4 → ResNet8x4 | 72.50 | 73.33 | 76.32 | **76.62** |
| VGG13 → VGG8 | 70.36 | 72.98 | 74.68 | **74.86** |

ImageNet top-1/top-5 (%) — Table X (ResNet34 → ResNet18): baseline
69.75/89.07, KD 70.66/89.88, DKD 71.70/90.41, **CKD 71.79/90.58**.

The average CIFAR-100 gain reported in the abstract is +3.42% over the
baseline and +1.71% over classic KD; ImageNet top-1 +2.04%.

<<<<<<< HEAD
## Reproduction audit & troubleshooting

Findings from a line-by-line audit against the paper full text plus local
verification runs:

1. **Everything the paper specifies is implemented as specified**: the loss
   decomposition of Eqs. 6-8 (KL directions, CE at `tau = 1`), penultimate
   feature splicing, the fusion module's own classifier, a single optimizer
   over student + fusion, and the CIFAR/ImageNet protocols.
2. **Device handling fixed (2026-10)**: `helper/loops.py` moved
   input/target to CUDA only; on machines without CUDA (CPU / Apple MPS)
   validation crashed with a device mismatch (`Mismatched Tensor types in
   NNPack convolutionOutput`). All loops now follow the model's device, and
   `train_student.py` / `train_teacher.py` select cuda/mps/cpu
   automatically.
3. **Learning-rate tension — check this first if results are low.** The
   paper text says lr 0.1 for CIFAR, but Table IX's baseline and KD columns
   match RepDistiller's published numbers *exactly* (e.g. 73.26/74.92 for
   WRN-40-2 -> WRN-16-2), and RepDistiller trains students with lr 0.05.
   The authors most likely ran on RepDistiller with its default lr. If your
   CKD runs land systematically 1-3 points below Table IX, rerun with
   `scripts/run_cifar_student_ckd_lr005.sh` (lr 0.05, everything else
   identical).
4. **Verify your teacher checkpoint before distilling.** The official
   RepDistiller wrn_40_2 checkpoint scores 75.61 on CIFAR-100 (verified
   locally with `scripts/verify_teachers.py`). A teacher below the paper's
   teacher row propagates directly into the student;
   `train_student.py` also prints the teacher accuracy at startup.

Remaining paper-unspecified choices (loss weights, KL temperature, fusion
internals, feature normalization) are documented under "Reproduction
decisions" and stay overridable via CLI flags.
=======
## Troubleshooting the gap vs. the paper (loss-balance sweep)

First-round reproductions land at classic-KD level (e.g. VGG13 -> VGG8
~71.9 vs. paper CKD 74.86). With tau^2-scaled KL and unit weights the
fusion loss is dominated by the teacher-mimicry term, so P^F degenerates
toward P^T and the student gains collapse to plain-KD level. The paper
prints the losses WITHOUT tau^2 and never gives numeric weights, so the
loss balance is the main free variable. `scripts/sweep_ckd_weights.sh`
sweeps the balance on VGG13 -> VGG8 and re-runs the two best configs on
WRN-40-2 -> WRN-16-2:

| config | flags | rationale |
| --- | --- | --- |
| `t2off` | `--kd_t2 0` | literal printed equations; CE dominates fusion training |
| `lam10` | `--lam 10` | CE-dominant fusion while keeping the tau^2-scaled student pull |
| `ab01_g10` | `--alpha 0.1 --beta 0.1 --gamma 10` | fusion trained mainly on labels of joint features; stronger pull to P^F |
| `t2off_g10` | `--kd_t2 0 --gamma 10` | literal balance + stronger student pull |
| `mlp2` | `--fusion_arch mlp2` | deeper fusion head (capacity axis) |

Watch `fusion test acc` in the logs: the CKD mechanism requires P^F to
exceed the teacher for the student to gain beyond KD.
>>>>>>> 3768baf (Add fusion-arch options, per-epoch fusion test accuracy diagnostic, and loss-balance sweep script)

## Reproduction decisions (documented inferences)

The paper omits several implementation details. The following defaults are
documented inferences, all overridable via CLI flags:

1. **Loss weights** `alpha, beta, lam, gamma, eta`: the paper gives only the
   symbolic weights (Eqs. 6-7), no numeric values. Default 1.0 each.
2. **Temperature**: the paper states only that `tau = 1` for the
   cross-entropy parts. For the KL parts we follow the convention of the
   benchmark the paper compares on (RepDistiller/DKD): `tau = 4` with
   `tau^2`-scaled KL terms (CE unscaled, `tau = 1`). The paper prints KL
   without `tau^2`; `--kd_t2 0` reproduces the literal printed form.
3. **Fusion module architecture**: the paper specifies "a trainable feature
   fusion module with a classifier" whose input is the spliced teacher +
   student features (Eq. 5 / Algorithm 1), but not its internal design.
   We use `Linear(d_T + d_S -> d_T) -> BatchNorm1d -> ReLU -> Linear(d_T ->
   n_cls)`, `d_T` being the teacher's penultimate width (configurable with
   `--fusion_dim`). P^F = softmax(Z^F / tau) matches the softmax in Eq. (5).
4. **Feature source**: "penultimate layer feature representations ... directly
   connected to the classifier" (paper Sec. III-A/III-B) = the flattened
   pre-classifier feature, i.e. `feat[-1]` of the RepDistiller model zoo
   forward output.
5. **Optimization**: a single SGD optimizer over student + fusion module
   parameters, with the paper's protocol (CIFAR: lr 0.1, 240 epochs,
   /10 at 150/180/210, batch 64; ImageNet: lr 0.2, 90 epochs, /10 every 30,
   batch 256, wd 1e-4). Note the lr 0.1 vs 0.05 tension discussed in
   "Reproduction audit & troubleshooting".
6. **Exp1 (Appendix A)**: Eq. (10) as printed,
   `alpha_1/2 * ||F^S - F^A||_2` with `F^A = F^T + F^S` (Eq. 9), reduces to
   `alpha_1/2 * ||F^T||_2` — a constant with zero gradient, since
   `F^S - F^A = -F^T`. We therefore default to the FitNet-style matching
   term `alpha_1/2 * ||F^S - F^T||_2` (Appendix A states Exp1/Exp2 are
   FitNet-based), and keep the literal printed equation behind
   `--exp1_literal 1`. This degeneracy is a likely typo in the paper.
7. **Exp2 (Appendix A)**: `Z^A = sigma(F^A)` uses the teacher's frozen
   classifier `sigma`; it requires equal teacher/student feature widths
   (e.g. ResNet110 → ResNet32, both 64-d).
8. **Table X teacher typo**: the caption says "teacher ResNet32", but the
   teacher accuracy 73.31 equals torchvision ResNet34 (and 69.75 equals
   torchvision ResNet18); we use ResNet34 → ResNet18.
9. **CIFAR-10 rows** (Table III/IV): no official CIFAR-10 teachers exist in
   RepDistiller; train them with `scripts/run_cifar_teacher_vanilla.sh`.

## Upstream codebase

The model zoo, data pipeline, teacher checkpoints and vanilla training
recipe are taken from the RepDistiller benchmark (MIT license):
https://github.com/HobbitLong/RepDistiller . The paper's Table IX baseline
and KD columns match that benchmark exactly, so all comparisons run under
the same protocol. The CKD-specific components (fusion module, Eqs. 6-8,
Exp1/Exp2) are newly implemented here.

## Disclaimer

This is an unofficial reproduction. Where the paper is silent, choices are
explicitly listed above. If the official code is later released, prefer it
for exact-numeral comparisons.
