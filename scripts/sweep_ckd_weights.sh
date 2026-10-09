#!/usr/bin/env bash
# CKD loss-balance sweep, phase 1 on VGG13 -> VGG8 (largest gap vs. paper
# in first-round reproduction), phase 2 automatically re-runs the two best
# configs on WRN-40-2 -> WRN-16-2.
#
# Motivation: with tau^2-scaled KL and unit weights, Loss_F is dominated by
# the teacher-mimicry term, so the fusion module degenerates into a copy of
# the teacher and the student gain collapses to classic-KD level. These
# configs shift the balance toward (a) label-driven fusion training and
# (b) a stronger pull of the student toward the fusion output.
set -u
PYTHON=${PYTHON:-python3}
T_VGG=./save/models/vgg13_vanilla/ckpt_epoch_240.pth
T_WRN=./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth
mkdir -p save/sweep

NAMES=(t2off lam10 ab01_g10 t2off_g10 mlp2)
FLAGS=(
  "--kd_t2 0"
  "--lam 10"
  "--alpha 0.1 --beta 0.1 --gamma 10"
  "--kd_t2 0 --gamma 10"
  "--fusion_arch mlp2"
)

for i in "${!NAMES[@]}"; do
  n=${NAMES[$i]}
  f=${FLAGS[$i]}
  echo "=== [phase1] vgg13->vgg8 config $n : $f"
  $PYTHON train_student.py --path_t "$T_VGG" --model_s vgg8 --method ckd \
      --trial "sw_$n" $f > "save/sweep/vgg_$n.log" 2>&1
  acc=$(grep -o 'best_acc [0-9.]*' save/student_data/*_sw_${n}/result.txt 2>/dev/null | awk '{print $2}' | tail -1)
  echo "=== [phase1] $n done, best_acc=$acc (paper CKD: 74.86, KD: 72.98, baseline: 70.36)"
done

echo "=== ranking phase-1 results"
accs=$(for i in "${!NAMES[@]}"; do
  a=$(grep -o 'best_acc [0-9.]*' save/student_data/*_sw_${NAMES[$i]}/result.txt 2>/dev/null | awk '{print $2}' | tail -1)
  echo "${a:-0} $i"
done | sort -rn | head -2 | awk '{print $2}')

for i in $accs; do
  n=${NAMES[$i]}
  f=${FLAGS[$i]}
  echo "=== [phase2] wrn_40_2->wrn_16_2 config $n : $f"
  $PYTHON train_student.py --path_t "$T_WRN" --model_s wrn_16_2 --method ckd \
      --trial "sw2_$n" $f > "save/sweep/wrn_$n.log" 2>&1
  acc=$(grep -o 'best_acc [0-9.]*' save/student_data/*_sw2_${n}/result.txt 2>/dev/null | awk '{print $2}' | tail -1)
  echo "=== [phase2] $n done, best_acc=$acc (paper CKD: 76.29, KD: 74.92, baseline: 73.26)"
done

echo "=== sweep complete"
for i in "${!NAMES[@]}"; do
  n=${NAMES[$i]}
  a1=$(grep -o 'best_acc [0-9.]*' save/student_data/*_sw_${n}/result.txt 2>/dev/null | awk '{print $2}' | tail -1)
  a2=$(grep -o 'best_acc [0-9.]*' save/student_data/*_sw2_${n}/result.txt 2>/dev/null | awk '{print $2}' | tail -1)
  echo "config $n : vgg8=${a1:-NA} wrn16_2=${a2:-NA}"
done
