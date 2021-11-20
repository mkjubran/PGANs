python3 train_PGANs_encoder.py \
          --dataset celeba \
          --ckptG            ../../../PresGANs/celebaimageSize32/SaveS2019/presgan_lambda_0.0_GS2019/netG_presgan_celeba_epoch_180.pth \
          --logsigma_file    ../../../PresGANs/celebaimageSize32/SaveS2019/presgan_lambda_0.0_GS2019/log_sigma_celeba_180.pth \
          --ckptE            ../../../PresGANs/S2019/VAEncoderType2_lambda0.0_GS2019epoch180 \
          --save_imgs_folder ../../../PresGANs/S2019/VAEncoderType2_lambda0.0_GS2019epoch180_Images \
          --beta  10 \
          --nz 100 \
          --epochs 250 \
          --imageSize 32 \
          --ngf 32 --ndf 32 --ngfg 32 --ncg 3 --nc 3 --lrE 0.0002
