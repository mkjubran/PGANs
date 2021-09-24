\python3 train_VAE_encoder_decoder.py \
          --dataset mnist \
          --ckptE            ../../../PresGANs/S2020/VAEncoderDCGANDecoder2020 \
          --save_imgs_folder ../../../PresGANs/S2020/VAEncoderDCGANDecoder2020_Images \
          --beta  10 \
          --nz 100 \
          --lrE 0.0002 \
          --epochs 50 \
          --imageSize 32 \
          --ngf 32 --ndf 32 --ngfg 32
