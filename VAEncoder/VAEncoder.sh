\python3 train_PGANs_encoder.py \
          --dataset mnist \
          --ckptG            ../../../PresGANs/S2021/presgan_lambda_0.0_GS2021/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file    ../../../PresGANs/S2021/presgan_lambda_0.0_GS2021/log_sigma_mnist_20.pth \
          --ckptE            ../../../PresGANs/S2021/VAEncoderReduced_lambda0.0_GS2021epoch20 \
          --save_imgs_folder ../../../PresGANs/S2021/VAEncoderReduced_lambda0.0_GS2021epoch20_Images \
          --beta  10 \
          --nz 64 \
          --epochs 100


