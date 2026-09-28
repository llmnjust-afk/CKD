# quick end-to-end smoke test (small subset, few epochs; verifies the
# pipeline runs and the loss decreases, NOT full accuracy)
# CPU-only machines: set num_workers 0-2, expect several minutes per epoch.

PYTHON=${PYTHON:-python3} train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth \
    --model_s vgg8 --method ckd --dataset cifar100 \
    --subset 0.02 --epochs 2 --batch_size 64 --num_workers 2 --trial smoke

PYTHON=${PYTHON:-python3} train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth \
    --model_s vgg8 --method kd --dataset cifar100 \
    --subset 0.02 --epochs 2 --batch_size 64 --num_workers 2 --trial smoke
