\python3 train_OGAN.py \
          --dataset mnist \
          --ckptG1            ../../../PresGANs/S2019/presgan_lambda_0.0_GS2019/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file_G1  ../../../PresGANs/S2019/presgan_lambda_0.0_GS2019/log_sigma_mnist_20.pth \
          --ckptD1            ../../../PresGANs/S2019/presgan_lambda_0.0_GS2019/netD_presgan_mnist_epoch_20.pth \
          --ckptE1            ../../../PresGANs/S2019/VAEncoder_lambda0.0_GS2019epoch20/netE_presgan_MNIST_epoch_50.pth\
          --ckptG2            ../../../PresGANs/S2020/presgan_lambda_0.0_GS2020/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file_G2  ../../../PresGANs/S2020/presgan_lambda_0.0_GS2020/log_sigma_mnist_20.pth \
          --ckptD2            ../../../PresGANs/S2020/presgan_lambda_0.0_GS2020/netD_presgan_mnist_epoch_20.pth \
          --ckptE2            ../../../PresGANs/S2020/VAEncoder_lambda0.0_GS2020epoch20/netE_presgan_MNIST_epoch_50.pth\
          --ckptOL_E1         ../../../PresGANs/OLoss/OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_E1 \
          --save_OL_E1        ../../../PresGANs/OLoss/OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_E1_Images \
          --ckptOL_E2         ../../../PresGANs/OLoss/OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_E2 \
          --save_OL_E2        ../../../PresGANs/OLoss/OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_E2_Images \
          --ckptOL            ../../../PresGANs/OLoss/OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0 \
          --lrOL 0.0001 \
          --beta  10 \
          --nz 64 \
          --epochs 1000 \
          --OLbatchSize 40


