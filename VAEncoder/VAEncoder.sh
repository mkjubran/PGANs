\python3 train_PGANs_encoder.py \
          --dataset mnist \
          --ckptG            ../../../PresGANs/S2020/presgan_lambda_0.0_GS2020/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file    ../../../PresGANs/S2020/presgan_lambda_0.0_GS2020/log_sigma_mnist_20.pth \
          --ckptE            ../../../PresGANs/S2020/VAEncoder_lambda0.0_GS2020epoch20 \
          --save_imgs_folder ../../../PresGANs/S2020/VAEncoder_lambda0.0_GS2020epoch20_Images \
          --beta  10 \
          --nz 100 \
          --epochs 20


