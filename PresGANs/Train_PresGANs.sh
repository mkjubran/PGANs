#python3 main_TB.py --dataset mnist --model presgan --lambda_ 0.01 --gpu 1 --restrict_sigma 1 --sigma_min 0.001 --sigma_max 0.3
#python3 main_TB.py --dataset mnist --model presgan --lambda_ 0.01 --gpu 1
#python3 main_TB.py --dataset mnist --model presgan --lambda_ 0.01 --gpu 0 --lrD 0.001 --lrG 0.0001 \
#                            --restrict_sigma 1 --sigma_min 0.001 --sigma_max 0.3 --logsigma_init 0 --sigma_lr 0.0001
python3 main_TB.py --dataset mnist --model presgan --lambda_ 0.01 --gpu 1 \
                            --restrict_sigma 1 --sigma_min 0.001 --sigma_max 0.3
