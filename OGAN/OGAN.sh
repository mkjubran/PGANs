\python3 train_OGAN.py \
          --dataset mnist \
          --ckptG1            ../../presgan_lambda_G_0.0_OLoss/netG_presgan_mnist_epoch_46.pth \
          --logsigma_file_G1    ../../presgan_lambda_G_0.0_OLoss/log_sigma_mnist_46.pth \
          --ckptD1            ../../presgan_lambda_G_0.0_OLoss/netD_presgan_mnist_epoch_46.pth \
          --ckptE1            ../../VAEncoder_E_lambda0.0_beta5_lr0.0002_OLoss/netE_presgan_MNIST_epoch_99.pth\
          --ckptG2            ../../presgan_lambda_G_0.0_OLoss/netG_presgan_mnist_epoch_46.pth \
          --logsigma_file_G2    ../../presgan_lambda_G_0.0_OLoss/log_sigma_mnist_46.pth \
          --ckptD2            ../../presgan_lambda_G_0.0_OLoss/netD_presgan_mnist_epoch_46.pth \
          --ckptE2            ../../VAEncoder_E_lambda0.0_beta5_lr0.0002_OLoss/netE_presgan_MNIST_epoch_99.pth\
          --ckptOL_E1            ../../Likelihood_lambda0.0_beta10_lr0.0002_E1_test \
          --save_OL_E1 ../../Likelihood_lambda0.0_beta10_lr0.0002_Images_E1_test \
          --ckptOL_E2            ../../Likelihood_lambda0.0_beta10_lr0.0002_E2_test \
          --save_OL_E2 ../../Likelihood_lambda0.0_beta10_lr0.0002_Images_E2_test \
          --beta  10 \
          --nz 64 \
          --epochs 1000 \
          --OLbatchSize 50


