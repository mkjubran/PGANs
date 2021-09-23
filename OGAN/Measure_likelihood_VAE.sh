python3 measure_likelihood_batch_VAE.py \
          --dataset mnist \
          --sample_from generator \
          --ckptDec                  ../../../PresGANs/SaveS2019/VAEncoderDCGANDecoder_2019epoch20/VAEncoderDCGANDecoder2019/netVADec_MNIST_epoch_20.pth \
          --ckptE                  ../../../PresGANs/SaveS2019/VAEncoderDCGANDecoder_2019epoch20/VAEncoderDCGANDecoder2019/netVAEnc_MNIST_epoch_20.pth \
          --save_likelihood_folder    ../../../PresGANs/OLoss/LikelihoodVAE_lr0.0002_valbatches100_S2000_VADCGANDec2019/Likelihood_EType2 \
          --number_samples_likelihood 1000\
          --lambda_ 0 \
          --lrOL 0.0002 \
          --beta  10 \
          --nz 100 \
          --OLepochs 5000 \
          --epochs 500 \
          --batchSize 100\
          --OLbatchSize 100\
          --num_gen_images 100\
          --seed_VAE 2019 \
          --S 2000 \
          --GPU 0 \
          --overlap_loss_min 0 \
          --valevery 100 \
          --valbatches 100


