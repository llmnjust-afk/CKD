# Same Table IX pairs as run_cifar_student_ckd.sh but with lr 0.05
# (the RepDistiller default student LR) instead of the paper-text lr 0.1.
# Rationale: the paper's Table IX baseline/KD columns match RepDistiller's
# published numbers exactly, and RepDistiller trains students with
# lr 0.05 (/10 at 150,180,210) — see README "Reproduction audit".
# Use this recipe if runs with the default lr 0.1 land systematically
# 1-3 points below Table IX.

T=wrn_40_2_vanilla;   S=wrn_16_2;   ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method ckd --learning_rate 0.05 --trial 1
T=wrn_40_2_vanilla;   S=wrn_40_1;   ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method ckd --learning_rate 0.05 --trial 1
T=resnet56_vanilla;   S=resnet20;   ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method ckd --learning_rate 0.05 --trial 1
T=resnet110_vanilla;  S=resnet32;   ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method ckd --learning_rate 0.05 --trial 1
T=resnet32x4_vanilla; S=resnet8x4;  ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method ckd --learning_rate 0.05 --trial 1
T=vgg13_vanilla;      S=vgg8;       ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method ckd --learning_rate 0.05 --trial 1

# KD reference row under the same recipe
T=wrn_40_2_vanilla;   S=wrn_16_2;   ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method kd  --learning_rate 0.05 --trial 1
