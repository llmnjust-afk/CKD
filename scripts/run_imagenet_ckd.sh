# ImageNet distillation of Table X.
# Paper protocol: SGD momentum 0.9, wd 1e-4, lr 0.2, /10 every 30 epochs, batch 256.
# NOTE: the caption of Table X says "teacher ResNet32", but the teacher
# accuracy 73.31 equals torchvision ResNet34 (and the 69.75 baseline equals
# torchvision ResNet18), so ResNet34 is almost surely the intended teacher.
# Point --data_root at the ImageNet root (must contain imagenet/train and
# imagenet/val; adjust dataset/imagenet.py get_data_folder if needed).

python train_student.py --dataset imagenet \
    --model_t imagenet_resnet34 --model_s imagenet_resnet18 --method ckd --trial 1
