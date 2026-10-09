#!/usr/bin/env bash
# Round-2 sweep: combine the two confirmed ingredients from round 1 --
# (a) fusion trained with CE-dominant balance (lam10) which pushed the
# fusion module ~2% above the teacher, and (b) a strong student-side pull
# (gamma) toward the fusion output. Also tests the literal-autograd KL
# variant and the deeper fusion head.
set -u
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
T_VGG=./save/models/vgg13_vanilla/ckpt_epoch_240.pth
T_WRN=./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth
mkdir -p save/sweep

echo "waiting for phase-3 to finish..."
while ! grep -q "phase3 complete" logs_phase3.txt 2>/dev/null; do
  sleep 180
done

NAMES=(lam10_g10 lam10_g4 t2off_g16 fullgrad fullgrad_lam10)
FLAGS=(
  "--lam 10 --gamma 10"
  "--lam 10 --gamma 4"
  "--kd_t2 0 --gamma 16"
  "--full_grad_kl 1"
  "--full_grad_kl 1 --lam 10"
)

for i in "${!NAMES[@]}"; do
  n=${NAMES[$i]}
  f=${FLAGS[$i]}
  echo "=== [round2] vgg13->vgg8 config $n : $f"
  $PYTHON train_student.py --path_t "$T_VGG" --model_s vgg8 --method ckd \
      --trial "r2_$n" $f > "save/sweep/r2_vgg_$n.log" 2>&1
  a=$(grep -o 'best_acc [0-9.]*' "save/student_data/cifar100_vgg8_ckd_Tvgg13_vanilla_trial_r2_${n}/result.txt" 2>/dev/null | awk '{print $2}' | tail -1)
  echo "=== [round2] $n done, best_acc=${a:-NA} (paper CKD: 74.86, KD: 72.98)"
done

echo "=== round2 ranking"
best_n=""
best_acc=0
for i in "${!NAMES[@]}"; do
  a=$(grep -o 'best_acc [0-9.]*' "save/student_data/cifar100_vgg8_ckd_Tvgg13_vanilla_trial_r2_${NAMES[$i]}/result.txt" 2>/dev/null | awk '{print $2}' | tail -1)
  echo "round2 ${NAMES[$i]}: ${a:-NA}"
  if [ -n "${a:-}" ]; then
    cmp=$(python3 -c "print(1 if $a > $best_acc else 0)")
    if [ "$cmp" = "1" ]; then best_acc=$a; best_n=${NAMES[$i]}; best_i=$i; fi
  fi
done
if [ -z "$best_n" ]; then echo "no round2 results"; exit 1; fi

f=${FLAGS[$best_i]}
echo "=== [round2-final] wrn_40_2->wrn_16_2 with best config $best_n : $f"
$PYTHON train_student.py --path_t "$T_WRN" --model_s wrn_16_2 --method ckd \
    --trial "r2f_$best_n" $f > "save/sweep/r2_wrn_$best_n.log" 2>&1
a=$(grep -o 'best_acc [0-9.]*' "save/student_data/cifar100_wrn_16_2_ckd_Twrn_40_2_vanilla_trial_r2f_${best_n}/result.txt" 2>/dev/null | awk '{print $2}' | tail -1)
echo "=== [round2-final] done, best_acc=${a:-NA} (paper CKD: 76.29, KD: 74.92)"

python3 scripts/collect_results.py > save/sweep/final_summary_r2.txt 2>&1
echo "=== round2 complete"
