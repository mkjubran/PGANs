python3 measure_likelihood_batch_VAE.py \
          --dataset mnist \
          --sample_from generator \
          --ckptDec                  ../../../PresGANs/SaveS2019/VAEncoderDCGANDecoder_2019epoch20/VAEncoderDCGANDecoder2019/netVADec_MNIST_epoch_20.pth \
          --ckptE                  ../../../PresGANs/SaveS2019/VAEncoderDCGANDecoder_2019epoch20/VAEncoderDCGANDecoder2019/netVAEnc_MNIST_epoch_20.pth \
          --save_likelihood_folder    ../../../PresGANs/Likelihood/tTrue0.0_OLb4_OLe2000_S2000_OLoss_VADCGANDec2019/Likelihood_EType2 \
          --number_samples_likelihood 1000\
          --lambda_ 0 \
          --lrOL 0.0001 \
          --beta  10 \
          --nz 100 \
          --OLepochs 100 \
          --epochs 500 \
          --batchSize 100\
          --OLbatchSize 100\
          --num_gen_images 100\
          --seed_VAE 2019 \
          --S 1000


