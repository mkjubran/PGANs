\python3 train_PGANs_encoder.py \
          --dataset mnist \
          --ckptG            ../../../PresGANs/presgan_lambda_0.0_GS2020/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file    ../../../PresGANs/presgan_lambda_0.0_GS2020/log_sigma_mnist_20.pth \
          --ckptE            ../../../PresGANs/VAEncoder_lambda0.0_GS2020epoch20 \
          --save_imgs_folder ../../../PresGANs/VAEncoder_lambda0.0_GS2020epoch20_Images \
          --beta  10 \
          --nz 64 \
          --epochs 100


