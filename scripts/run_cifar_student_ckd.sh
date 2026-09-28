# CKD distillation runs on CIFAR-100 (Table IX of the paper) and the
# extra pairs of Table III / Table IV.
# Protocol: SGD momentum 0.9, wd 5e-4, lr 0.1, /10 at 150,180,210, 240 epochs, batch 64.
# Loss weights alpha/beta/lam/gamma/eta default to 1.0 and T=4 with tau^2
# scaling (see README for the documented-inference discussion).

T=wrn_40_2_vanilla;   S=wrn_16_2;   ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method ckd --trial 1
T=wrn_40_2_vanilla;   S=wrn_40_1;   ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method ckd --trial 1
T=resnet56_vanilla;   S=resnet20;   ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method ckd --trial 1
T=resnet110_vanilla;  S=resnet32;   ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method ckd --trial 1
T=resnet32x4_vanilla; S=resnet8x4;  ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method ckd --trial 1
T=vgg13_vanilla;      S=vgg8;       ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method ckd --trial 1

# classic KD baseline rows of Table IX
T=wrn_40_2_vanilla;   S=wrn_16_2;   ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method kd  --trial 1
T=vgg13_vanilla;      S=vgg8;       ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method kd  --trial 1

# Table IV: different teachers, same student
T=vgg11_vanilla;      S=vgg8;       ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method ckd --trial 1
T=resnet110_vanilla;  S=resnet20;   ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method ckd --trial 1
T=resnet32_vanilla;   S=resnet20;   ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method ckd --trial 1

# Appendix Exp1 / Exp2 (exploratory; FitNet-based setups of Appendix A)
T=resnet110_vanilla;  S=resnet32;   ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method exp1 --trial 1
T=resnet110_vanilla;  S=resnet32;   ${PYTHON:-python3} train_student.py --path_t ./save/models/${T}/ckpt_epoch_240.pth --model_s ${S} --method exp2 --trial 1
