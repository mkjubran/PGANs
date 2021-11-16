Path1='mnistimageSize32'
Path2='tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.0_W20.0_valbatches100_S2000_GS2019_GS2020'
Step='4000'

Path='../../../PresGANs/'$Path1'/'$Path2'/G'
python3 measure_likelihood_batch.py \
          --dataset mnist \
          --sample_from dataset \
          --ckptG1            $Path'/netG1_presgan_mnist_step_'$Step'.pth' \
          --logsigma_file_G1  $Path'/log_sigma_G1_mnist_step_'$Step'.pth' \
          --ckptD1            $Path'/netD1_presgan_mnist_step_'$Step'.pth' \
          --ckptE1            ../../../PresGANs/mnistimageSize32/SaveS2019/VAEncoderType2_lambda0.0_GS2019epoch20/netE_presgan_MNIST_epoch_19.pth\
          --ckptG2            $Path'/netG2_presgan_mnist_step_'$Step'.pth' \
          --logsigma_file_G2  $Path'/log_sigma_G2_mnist_step_'$Step'.pth' \
          --ckptD2            $Path'/netD2_presgan_mnist_step_'$Step'.pth' \
          --ckptE2            ../../../PresGANs/mnistimageSize32/SaveS2020/VAEncoderType2_lambda0.0_GS2020epoch20/netE_presgan_MNIST_epoch_19.pth\
          --save_likelihood_folder    '../../../PresGANs/MeasureLL_'$Path2'_OGAN/Step'$Step\
          --number_samples_likelihood 10000\
          --S 2000 \
          --seed_G1 2019 \
          --seed_G2 2020 \
          --lambda_ 0.0 \
          --W1 0.0 \
          --W2 0.0 \
          --lrOL 0.0002 \
          --beta  10 \
          --nz 100 \
          --OLepochs 50000 \
          --epochs 50 \
          --batchSize 1\
          --OLbatchSize 1\
          --num_gen_images 20\
          --GPU 0 \
          --overlap_loss_min -10000000000 \
          --imageSize 32 --ngf 32 --ndf 32 --ngfg 32 --ndfg 32 --ncg 1 --nc 1 --lrE1 0.0002 --lrE2 0.0002

