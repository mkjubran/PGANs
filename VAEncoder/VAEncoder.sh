\python3 train_PGANs_encoder.py \
          --dataset mnist \
          --ckptG            ../../presgan_lambda_G_0.0/netG_presgan_mnist_epoch_46.pth \
          --logsigma_file    ../../presgan_lambda_G_0.0/log_sigma_mnist_46.pth \
          --ckptE            ../../VAEncoder_E_lambda0.0_beta10_lr0.0002_test \
          --save_imgs_folder ../../VAEncoder_E_lambda0.0_beta10_lr0.0002_Images_test \
          --beta  10 \
          --nz 64 \
          --epochs 2


