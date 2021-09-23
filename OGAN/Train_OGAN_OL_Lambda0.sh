python3 train_OGAN_OL_batch.py \
          --dataset mnist \
          --ckptG1            ../../../PresGANs/SaveS2019/presgan_lambda_0.0001_GS2019/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file_G1  ../../../PresGANs/SaveS2019/presgan_lambda_0.0001_GS2019/log_sigma_mnist_20.pth \
          --ckptD1            ../../../PresGANs/SaveS2019/presgan_lambda_0.0001_GS2019/netD_presgan_mnist_epoch_20.pth \
          --ckptE1            ../../../PresGANs/SaveS2019/VAEncoderType2_lambda0.0001_GS2019epoch20/netE_presgan_MNIST_epoch_19.pth\
          --ckptG2            ../../../PresGANs/SaveS2020/presgan_lambda_0.0001_GS2020/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file_G2  ../../../PresGANs/SaveS2020/presgan_lambda_0.0001_GS2020/log_sigma_mnist_20.pth \
          --ckptD2            ../../../PresGANs/SaveS2020/presgan_lambda_0.0001_GS2020/netD_presgan_mnist_epoch_20.pth \
          --ckptE2            ../../../PresGANs/SaveS2020/VAEncoderType2_lambda0.0001_GS2020epoch20/netE_presgan_MNIST_epoch_19.pth\
          --ckptOL_E1         ../../../PresGANs/OLoss/GAN_lambda0.0_lr0.0002_W10.0005_W20.0005_valbatches100_S2000_GS2019_GS2020/E1Type2 \
          --save_OL_E1        ../../../PresGANs/OLoss/GAN_lambda0.0_lr0.0002_W10.0005_W20.000_valbatches100_S20005_GS2019_GS2020/E1Type2_Images \
          --ckptOL_E2         ../../../PresGANs/OLoss/GAN_lambda0.0_lr0.0002_W10.0005_W20.0005_valbatches100_S2000_GS2019_GS2020/E2Type2 \
          --save_OL_E2        ../../../PresGANs/OLoss/GAN_lambda0.0_lr0.0002_W10.0005_W20.0005_valbatches100_S2000_GS2019_GS2020/E2Type2_Images \
          --ckptOL            ../../../PresGANs/OLoss/GAN_lambda0.0_lr0.0002_W10.0005_W20.0005_valbatches100_S2000_GS2019_GS2020/OLoss \
          --ckptOL_G          ../../../PresGANs/OLoss/GAN_lambda0.0_lr0.0002_W10.0005_W20.0005_valbatches100_S2000_GS2019_GS2020/G \
          --ckptOL_G1         ../../../PresGANs/OLoss/GAN_lambda0.0_lr0.0002_W10.0005_W20.0005_valbatches100_S2000_GS2019_GS2020/G1_Images \
          --ckptOL_G2         ../../../PresGANs/OLoss/GAN_lambda0.0_lr0.0002_W10.0005_W20.0005_valbatches100_S2000_GS2019_GS2020/G2_Images \
          --S 2000 \
          --seed_G1 2019 \
          --seed_G2 2020 \
          --lambda_ 0.0 \
          --W1 0.0005 \
          --W2 0.0005 \
          --lrOL 0.0002 \
          --beta  10 \
          --nz 100 \
          --OLepochs 5000 \
          --epochs 50 \
          --batchSize 100\
          --OLbatchSize 100\
          --num_gen_images 100\
          --GPU 0 \
          --overlap_loss_min 0 \
          --valevery 200 \
          --valbatches 100 \
          --mode train_validate

