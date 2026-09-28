# train vanilla teachers from scratch (protocol of the CKD paper:
# SGD momentum 0.9, wd 5e-4, lr 0.1, /10 at 150,180,210, 240 epochs, batch 64)
# NOTE: the official RepDistiller teacher ckpts were trained with lr 0.05;
# pass `--learning_rate 0.05` if you want to match that recipe exactly.

# CIFAR-100 teachers (Table IX)
python train_teacher.py --model wrn_40_2   --dataset cifar100 --trial 1
python train_teacher.py --model resnet56   --dataset cifar100 --trial 1
python train_teacher.py --model resnet110  --dataset cifar100 --trial 1
python train_teacher.py --model resnet32x4 --dataset cifar100 --trial 1
python train_teacher.py --model vgg13      --dataset cifar100 --trial 1
python train_teacher.py --model vgg11      --dataset cifar100 --trial 1   # Table IV teacher (79.06)
python train_teacher.py --model resnet32   --dataset cifar100 --trial 1   # Table IV teacher (71.37)

# CIFAR-10 teachers (Table III/IV)
python train_teacher.py --model wrn_40_2   --dataset cifar10  --trial 1
python train_teacher.py --model vgg13      --dataset cifar10  --trial 1
