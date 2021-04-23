python3 train_PGANs_v1.py \
          --ckptG         ../../presgan_lambda_0.01_G_sigma0.001_0.3_lrD0.0002_lrG0.0002/netG_presgan_mnist_epoch_80.pth \
          --logsigma_file ../../presgan_lambda_0.01_G_sigma0.001_0.3_lrD0.0002_lrG0.0002/log_sigma_mnist_80.pth \
          --ckptE       ../../VAEncoder_lambda_0.01_G_sigma0.001_0.3_lrD0.0002_lrG0.0002
