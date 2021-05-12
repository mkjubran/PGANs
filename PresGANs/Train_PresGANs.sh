#python3 main_TB.py --dataset mnist --model presgan --lambda_ 0.0 --gpu 1 --save_imgs_every 2 \
#                            --restrict_sigma 1 --sigma_min 0.001 --sigma_max 0.2 --epochs 50 --seed 2021

python3 main_TB.py --dataset mnist --model presgan --lambda_ 0.0 --gpu 1 --save_imgs_every 2 \
                            --restrict_sigma 1 --sigma_min 0.001 --sigma_max 0.2 --epochs 50 --seed 2018

#          --ckptG            ../../../PresGANs/S2018/presgan_lambda_0.0_GS2018/netG_presgan_mnist_epoch_20.pth \
#          --logsigma_file  ../../../PresGANs/S2018/presgan_lambda_0.0_GS2018/log_sigma_mnist_20.pth \
#         --ckptD            ../../../PresGANs/S2018/presgan_lambda_0.0_GS2018/netD_presgan_mnist_epoch_20.pth
