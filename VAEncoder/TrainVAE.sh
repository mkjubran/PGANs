\python3 train_VAE.py \
          --dataset mnist \
          --ckptE            ../../../PresGANs/S2020/VAEncoderDCGANDecoderBCE2020 \
          --save_imgs_folder ../../../PresGANs/S2020/VAEncoderDCGANDecoderBCE2020_Images \
          --beta  10 \
          --nz 100 \
          --lrE 0.0002 \
          --epochs 50 \
          --imageSize 32 \
          --ngf 32 --ndf 32 --ngfg 32 --ncg 1 --nc 1 --lrE 0.0002
