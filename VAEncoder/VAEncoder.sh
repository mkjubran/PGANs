\python3 train_PGANs_encoder.py \
          --dataset mnist \
          --ckptG            ../../../PresGANs/S2018/presgan_lambda_0.0_GS2018/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file    ../../../PresGANs/S2018/presgan_lambda_0.0_GS2018/log_sigma_mnist_20.pth \
          --ckptE            ../../../PresGANs/S2018/VAEncoderType2_lambda0.0_GS2018epoch20 \
          --save_imgs_folder ../../../PresGANs/S2018/VAEncoderType2_lambda0.0_GS2018epoch20_Images \
          --beta  10 \
          --nz 100 \
          --epochs 20


