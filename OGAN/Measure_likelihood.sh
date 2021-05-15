python3 measure_likelihood.py \
          --dataset mnist \
          --ckptG1                    ../../../PresGANs/S2019/presgan_lambda_0.0_GS2019/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file_G1          ../../../PresGANs/S2019/presgan_lambda_0.0_GS2019/log_sigma_mnist_20.pth \
          --ckptD1                    ../../../PresGANs/S2019/presgan_lambda_0.0_GS2019/netD_presgan_mnist_epoch_20.pth \
          --ckptE1                    ../../../PresGANs/S2019/VAEncoder_lambda0.0_GS2019epoch20/netE_presgan_MNIST_epoch_19.pth\
          --ckptG2                    ../../../PresGANs/S2019/presgan_lambda_0.0_GS2019/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file_G2          ../../../PresGANs/S2019/presgan_lambda_0.0_GS2019/log_sigma_mnist_20.pth \
          --ckptD2                    ../../../PresGANs/S2019/presgan_lambda_0.0_GS2019/netD_presgan_mnist_epoch_20.pth \
          --ckptE2                    ../../../PresGANs/S2019/VAEncoder_lambda0.0_GS2019epoch20/netE_presgan_MNIST_epoch_19.pth\
          --save_likelihood_folder    ../../../PresGANs/Likelihood/True0.0_OLoss_lambda_0.0_GS2019_GS2019/Likelihood_E1 \
          --lambda_ 0 \
          --W1 0.0 \
          --W2 0.0 \
          --lrOL 0.0001 \
          --beta  10 \
          --nz 100 \
          --OLepochs 1000 \
          --epochs 500 \
          --batchSize 100\
          --OLbatchSize 1\
          --num_gen_images 100


