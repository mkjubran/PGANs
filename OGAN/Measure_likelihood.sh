python3 measure_likelihood.py \
          --dataset mnist \
          --ckptG1                    ../../../PresGANs/S2020/presgan_lambda_0.0_GS2020/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file_G1          ../../../PresGANs/S2020/presgan_lambda_0.0_GS2020/log_sigma_mnist_20.pth \
          --ckptD1                    ../../../PresGANs/S2020/presgan_lambda_0.0_GS2020/netD_presgan_mnist_epoch_20.pth \
          --ckptE1                    ../../../PresGANs/S2020/VAEncoder_lambda0.0_GS2020epoch20/netE_presgan_MNIST_epoch_50.pth\
          --ckptG2                    ../../../PresGANs/S2021/presgan_lambda_0.0_GS2021/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file_G2          ../../../PresGANs/S2021/presgan_lambda_0.0_GS2021/log_sigma_mnist_20.pth \
          --ckptD2                    ../../../PresGANs/S2021/presgan_lambda_0.0_GS2021/netD_presgan_mnist_epoch_20.pth \
          --ckptE2                    ../../../PresGANs/S2021/VAEncoder_lambda0.0_GS2021epoch20/netE_presgan_MNIST_epoch_50.pth\
          --save_likelihood_folder    ../../../PresGANs/Likelihood/True0.0_OLoss_lambda_0.0_GS2020_GS2021/Likelihood_E1 \
          --lambda_ 0 \
          --W1 0.0 \
          --W2 0.0 \
          --lrOL 0.0001 \
          --beta  10 \
          --nz 64 \
          --OLepochs 100 \
          --epochs 500 \
          --batchSize 100\
          --OLbatchSize 1\
          --num_gen_images 100


