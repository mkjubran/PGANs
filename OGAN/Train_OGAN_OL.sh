python3 train_OGAN_OL.py \
          --dataset mnist \
          --ckptG1            ../../../PresGANs/S2020/presgan_lambda_0.0_GS2020/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file_G1  ../../../PresGANs/S2020/presgan_lambda_0.0_GS2020/log_sigma_mnist_20.pth \
          --ckptD1            ../../../PresGANs/S2020/presgan_lambda_0.0_GS2020/netD_presgan_mnist_epoch_20.pth \
          --ckptE1            ../../../PresGANs/S2020/VAEncoder_lambda0.0_GS2020epoch20/netE_presgan_MNIST_epoch_50.pth\
          --ckptG2            ../../../PresGANs/S2021/presgan_lambda_0.0_GS2021/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file_G2  ../../../PresGANs/S2021/presgan_lambda_0.0_GS2021/log_sigma_mnist_20.pth \
          --ckptD2            ../../../PresGANs/S2021/presgan_lambda_0.0_GS2021/netD_presgan_mnist_epoch_20.pth \
          --ckptE2            ../../../PresGANs/S2021/VAEncoder_lambda0.0_GS2021epoch20/netE_presgan_MNIST_epoch_50.pth\
          --ckptOL_E1         ../../../PresGANs/OLoss/tOLoss_lambda_0.0_GS2020_GS2021/OLoss_lambda0.0_E1 \
          --save_OL_E1        ../../../PresGANs/OLoss/tOLoss_lambda_0.0_GS2020_GS2021/OLoss_lambda0.0_E1_Images \
          --ckptOL_E2         ../../../PresGANs/OLoss/tOLoss_lambda_0.0_GS2020_GS2021/OLoss_lambda0.0_E2 \
          --save_OL_E2        ../../../PresGANs/OLoss/tOLoss_lambda_0.0_GS2020_GS2021/OLoss_lambda0.0_E2_Images \
          --ckptOL            ../../../PresGANs/OLoss/tOLoss_lambda_0.0_GS2020_GS2021/OLoss_lambda0.0 \
          --ckptOL_G         ../../../PresGANs/OLoss/tOLoss_lambda_0.0_GS2020_GS2021/OLoss_lambda0.0_G \
          --ckptOL_G1         ../../../PresGANs/OLoss/tOLoss_lambda_0.0_GS2020_GS2021/OLoss_lambda0.0_G1_Images \
          --ckptOL_G2         ../../../PresGANs/OLoss/tOLoss_lambda_0.0_GS2020_GS2021/OLoss_lambda0.0_G2_Images \
          --W1 0.000001 \
          --W2 0.000001 \
          --lrOL 0.0001 \
          --beta  10 \
          --nz 64 \
          --OLepochs 20 \
          --epochs 500 \
          --batchSize 100\
          --OLbatchSize 10\
          --num_gen_images 100


