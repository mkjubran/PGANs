#python3 main_TB.py --dataset mnist --model presgan --lambda_ 0.0 --gpu 1 --save_imgs_every 2 \
#                            --restrict_sigma 1 --sigma_min 0.001 --sigma_max 0.2 --epochs 50 --seed 2021

python3 main_TB.py --dataset mnist --model presgan --lambda_ 0.0001 --gpu 0 --save_imgs_every 2 \
                            --restrict_sigma 1 --sigma_min 0.001 --sigma_max 0.2 --epochs 50 --seed 2019 
#          --ckptG            ../../../PresGANs/S2019/presgan_lambda_0.0001_GS2019/netG_presgan_mnist_epoch_20.pth \
#          --logsigma_file  ../../../PresGANs/S2019/presgan_lambda_0.0001_GS2019/log_sigma_mnist_20.pth \
#         --ckptD            ../../../PresGANs/S2019/presgan_lambda_0.0001_GS2019/netD_presgan_mnist_epoch_20.pth
