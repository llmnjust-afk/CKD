# fetch the official pretrained vanilla teachers used by Table IX of the
# CKD paper (weights are from RepDistiller; the paper's teacher accuracies
# match this release exactly)
mkdir -p save/models/
cd save/models

for m in wrn_40_2_vanilla resnet56_vanilla resnet110_vanilla resnet32x4_vanilla vgg13_vanilla; do
    mkdir -p $m
    wget -c http://shape2prog.csail.mit.edu/repo/$m/ckpt_epoch_240.pth -O $m/ckpt_epoch_240.pth
done

cd ../..
