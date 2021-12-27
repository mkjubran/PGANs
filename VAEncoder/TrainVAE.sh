\python3 train_VAE.py \
          --dataset cifar10 \
          --ckptE            ../../../PresGANs/S2019/VAEncoderDCGANDecoderBCE2019 \
          --save_imgs_folder ../../../PresGANs/S2019/VAEncoderDCGANDecoderBCE2019_Images \
          --beta  10 \
          --nz 100 \
          --lrE 0.0002 \
          --epochs 500 \
          --imageSize 32 \
          --ngf 32 --ndf 32 --ngfg 32 --ncg 3 --nc 3 --lrE 0.0002
