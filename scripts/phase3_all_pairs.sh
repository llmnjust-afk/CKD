#!/usr/bin/env bash
# Phase 3: after sweep_ckd_weights.sh finishes, take the best phase-1
# config and run the remaining Table IX pairs with it, then emit a
# consolidated parameter/accuracy table for every pair.
set -u
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}

echo "waiting for the phase-1/2 sweep to finish..."
while ! grep -q "sweep complete" logs_sweep.txt 2>/dev/null; do
  sleep 120
done
echo "sweep finished; ranking phase-1 configs"

NAMES=(t2off lam10 ab01_g10 t2off_g10 mlp2)
FLAGS=(
  "--kd_t2 0"
  "--lam 10"
  "--alpha 0.1 --beta 0.1 --gamma 10"
  "--kd_t2 0 --gamma 10"
  "--fusion_arch mlp2"
)

best_i=""
best_acc=0
for i in "${!NAMES[@]}"; do
  a=$(grep -o 'best_acc [0-9.]*' save/student_data/*_sw_${NAMES[$i]}/result.txt 2>/dev/null | awk '{print $2}' | tail -1)
  echo "phase1 ${NAMES[$i]}: ${a:-NA}"
  if [ -n "${a:-}" ]; then
    cmp=$(python3 -c "print(1 if $a > $best_acc else 0)")
    if [ "$cmp" = "1" ]; then best_acc=$a; best_i=$i; fi
  fi
done
if [ -z "$best_i" ]; then
  echo "no phase-1 results found; aborting"; exit 1
fi
n=${NAMES[$best_i]}
f=${FLAGS[$best_i]}
echo "best config: $n ($f), vgg8 best_acc=$best_acc"

mkdir -p save/models
for m in resnet56_vanilla resnet110_vanilla resnet32x4_vanilla; do
  if [ ! -f save/models/$m/ckpt_epoch_240.pth ]; then
    mkdir -p save/models/$m
    curl -sL --max-time 600 -o save/models/$m/ckpt_epoch_240.pth \
        http://shape2prog.csail.mit.edu/repo/$m/ckpt_epoch_240.pth
    echo "downloaded $m"
  fi
done

run_one() { # teacher_dir student
  td=$1; s=$2
  echo "=== [phase3] $td -> $s with config $n : $f"
  $PYTHON train_student.py --path_t ./save/models/${td}/ckpt_epoch_240.pth \
      --model_s $s --method ckd --trial "p3_$n" $f \
      > "save/sweep/p3_${s}_${n}.log" 2>&1
  a=$(grep -o 'best_acc [0-9.]*' "save/student_data/cifar100_${s}_ckd_T${td}_trial_p3_${n}/result.txt" 2>/dev/null | awk '{print $2}' | tail -1)
  echo "=== [phase3] $td -> $s done, best_acc=${a:-NA}"
}

run_one wrn_40_2_vanilla wrn_40_1
run_one resnet56_vanilla resnet20
run_one resnet110_vanilla resnet32
run_one resnet32x4_vanilla resnet8x4

python3 scripts/collect_results.py > save/sweep/final_summary.txt 2>&1
echo "=== phase3 complete; summary:"
cat save/sweep/final_summary.txt
