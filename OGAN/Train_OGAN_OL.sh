python3 train_OGAN_OL.py \
          --dataset mnist \
          --ckptG1            ../../../PresGANs/SaveS2019/presgan_lambda_0.0_GS2019/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file_G1  ../../../PresGANs/SaveS2019/presgan_lambda_0.0_GS2019/log_sigma_mnist_20.pth \
          --ckptD1            ../../../PresGANs/SaveS2019/presgan_lambda_0.0_GS2019/netD_presgan_mnist_epoch_20.pth \
          --ckptE1            ../../../PresGANs/SaveS2019/VAEncoderType2_lambda0.0_GS2019epoch20/netE_presgan_MNIST_epoch_19.pth\
          --ckptG2            ../../../PresGANs/SaveS2020/presgan_lambda_0.0_GS2020/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file_G2  ../../../PresGANs/SaveS2020/presgan_lambda_0.0_GS2020/log_sigma_mnist_20.pth \
          --ckptD2            ../../../PresGANs/SaveS2020/presgan_lambda_0.0_GS2020/netD_presgan_mnist_epoch_20.pth \
          --ckptE2            ../../../PresGANs/SaveS2020/VAEncoderType2_lambda0.0_GS2020epoch20/netE_presgan_MNIST_epoch_19.pth\
          --ckptOL_E1         ../../../PresGANs/OLoss/True0.01_OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_E1Type2 \
          --save_OL_E1        ../../../PresGANs/OLoss/True0.01_OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_E1Type2_Images \
          --ckptOL_E2         ../../../PresGANs/OLoss/True0.01_OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_E2Type2 \
          --save_OL_E2        ../../../PresGANs/OLoss/True0.01_OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_E2Type2_Images \
          --ckptOL            ../../../PresGANs/OLoss/True0.01_OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0 \
          --ckptOL_G          ../../../PresGANs/OLoss/True0.01_OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_G \
          --ckptOL_G1         ../../../PresGANs/OLoss/True0.01_OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_G1_Images \
          --ckptOL_G2         ../../../PresGANs/OLoss/True0.01_OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_G2_Images \
          --lambda_ 0 \
          --W1 0.0 \
          --W2 0.01 \
          --lrOL 0.0001 \
          --beta  10 \
          --nz 100 \
          --OLepochs 600 \
          --epochs 50 \
          --batchSize 100\
          --OLbatchSize 10\
          --num_gen_images 100


