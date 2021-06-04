python3 measure_likelihood.py \
          --dataset mnist \
          --sample_from generator \
          --ckptG1                    ../../../PresGANs/SaveOLoss/True0.1_OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_G/netG1_presgan_mnist_step_2000.pth \
          --logsigma_file_G1          ../../../PresGANs/SaveOLoss/True0.1_OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_G/log_sigma_G1_mnist_step_2000.pth \
          --ckptD1                    ../../../PresGANs/SaveOLoss/True0.1_OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_G/netD1_presgan_mnist_step_2000.pth \
          --ckptE1                    ../../../PresGANs/SaveS2019/VAEncoderType2_lambda0.0_GS2019epoch20/netE_presgan_MNIST_epoch_19.pth\
          --ckptG2                    ../../../PresGANs/SaveOLoss/True0.1_OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_G/netG2_presgan_mnist_step_2000.pth \
          --logsigma_file_G2          ../../../PresGANs/SaveOLoss/True0.1_OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_G/log_sigma_G2_mnist_step_2000.pth \
          --ckptD2                    ../../../PresGANs/SaveOLoss/True0.1_OLoss_lambda_0.0_GS2019_GS2020/OLoss_lambda0.0_G/netD2_presgan_mnist_step_2000.pth \
          --ckptE2                    ../../../PresGANs/SaveS2020/VAEncoderType2_lambda0.0_GS2020epoch20/netE_presgan_MNIST_epoch_19.pth\
          --save_likelihood_folder    ../../../PresGANs/Likelihood/True0.1_OLoss_lambda_0.0_GS2019_GS2020_AfterOL/Likelihood_EType2 \
          --number_samples_likelihood 1000\
          --lambda_ 0 \
          --W1 0.0 \
          --W2 0.0 \
          --lrOL 0.0001 \
          --beta  10 \
          --nz 100 \
          --OLepochs 2000 \
          --epochs 500 \
          --batchSize 100\
          --OLbatchSize 4\
          --num_gen_images 100


