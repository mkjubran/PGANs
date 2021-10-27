python3 train_OGAN_OL_batch.py \
          --dataset cifar10 \
          --ckptG1            ../../../PresGANs/cifar10imageSize32/SaveS2019/presgan_lambda_0.0_GS2019/netG_presgan_cifar10_epoch_180.pth \
          --logsigma_file_G1  ../../../PresGANs/cifar10imageSize32/SaveS2019/presgan_lambda_0.0_GS2019/log_sigma_cifar10_180.pth \
          --ckptD1            ../../../PresGANs/cifar10imageSize32/SaveS2019/presgan_lambda_0.0_GS2019/netD_presgan_cifar10_epoch_180.pth \
          --ckptE1            ../../../PresGANs/cifar10imageSize32/SaveS2019/VAEncoderType2_lambda0.0_GS2019epoch180/netE_presgan_CIFAR10_epoch_180.pth\
          --ckptG2            ../../../PresGANs/cifar10imageSize32/SaveS2020/presgan_lambda_0.0_GS2020/netG_presgan_cifar10_epoch_180.pth \
          --logsigma_file_G2  ../../../PresGANs/cifar10imageSize32/SaveS2020/presgan_lambda_0.0_GS2020/log_sigma_cifar10_180.pth \
          --ckptD2            ../../../PresGANs/cifar10imageSize32/SaveS2020/presgan_lambda_0.0_GS2020/netD_presgan_cifar10_epoch_180.pth \
          --ckptE2            ../../../PresGANs/cifar10imageSize32/SaveS2020/VAEncoderType2_lambda0.0_GS2020epoch180/netE_presgan_CIFAR10_epoch_180.pth\
          --ckptOL_E1         ../../../PresGANs/testcifar10OLoss32/vGAN_lambda0.0_lr0.0002_W10.005_W20.005_valbatches100_S2000_GS2019_GS2020/E1Type2 \
          --save_OL_E1        ../../../PresGANs/testcifar10OLoss32/vGAN_lambda0.0_lr0.0002_W10.005_W20.005_valbatches100_S2000_GS2019_GS2020/E1Type2_Images \
          --ckptOL_E2         ../../../PresGANs/testcifar10OLoss32/vGAN_lambda0.0_lr0.0002_W10.005_W20.005_valbatches100_S2000_GS2019_GS2020/E2Type2 \
          --save_OL_E2        ../../../PresGANs/testcifar10OLoss32/vGAN_lambda0.0_lr0.0002_W10.005_W20.005_valbatches100_S2000_GS2019_GS2020/E2Type2_Images \
          --ckptOL            ../../../PresGANs/testcifar10OLoss32/vGAN_lambda0.0_lr0.0002_W10.005_W20.005_valbatches100_S2000_GS2019_GS2020/OLoss \
          --ckptOL_G          ../../../PresGANs/testcifar10OLoss32/vGAN_lambda0.0_lr0.0002_W10.005_W20.005_valbatches100_S2000_GS2019_GS2020/G \
          --ckptOL_G1         ../../../PresGANs/testcifar10OLoss32/vGAN_lambda0.0_lr0.0002_W10.005_W20.005_valbatches100_S2000_GS2019_GS2020/G1_Images \
          --ckptOL_G2         ../../../PresGANs/testcifar10OLoss32/vGAN_lambda0.0_lr0.0002_W10.005_W20.005_valbatches100_S2000_GS2019_GS2020/G2_Images \
          --S 2000 \
          --seed_G1 2019 \
          --seed_G2 2020 \
          --lambda_ 0.0 \
          --W1 0.005 \
          --W2 0.005 \
          --lrOL 0.0002 \
          --beta  10 \
          --nz 100 \
          --OLepochs 50000 \
          --epochs 50 \
          --batchSize 100\
          --OLbatchSize 100\
          --num_gen_images 100\
          --GPU 0 \
          --overlap_loss_min -10000000000 \
          --valevery 2 \
          --valbatches 5 \
          --mode validate\
          --imageSize 32 --ngf 32 --ndf 32 --ngfg 32 --ndfg 32 --ncg 3 --nc 3 --lrE1 0.0001 --lrE2 0.0001
