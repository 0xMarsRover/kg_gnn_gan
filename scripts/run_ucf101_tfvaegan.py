import os
# Will do:
# 1. Implement gzsl classifier without using OD (--gzsl)    --done
# 2. argparse: --zsl; --gzsl; --gzsl_od                     -- done

# Need to check: zsl/gzsl, nz/attSize, nclass_all, nepoch
# Also consider --syn_num: how many number of features should be generated per class

# Conducting different experiments
# when --class_embedding att
#       --nz 115 --attSize 115

# when --class_embedding wv
#       --nz 300 --attSize 300

# when --class_embedding img_avg
#       --nz 2048 --attSize 2048


# Number of splits range(30)
# when using --class_embedding img_avg, set "CUDA_LAUNCH_BLOCKING=1 python train_vaegan.py ...."
for n in range(1):
    n = n + 1
    os.system('''CUDA_LAUNCH_BLOCKING=1 python /content/kg_gnn_gan/train_tfvaegan.py \
    --nclass_all 101 --dataset ucf101 --zsl \
    --dataroot /content/drive/MyDrive/colab_data/action_datasets \
    --splits_path ucf101_semantics --split {split} \
    --action_embedding i3d --class_embedding img_avg \
    --nepoch 50 --batch_size 64 \
    --syn_num 600 --preprocessing --cuda --gammaD 10 --gammaG 10 \
    --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
    --nz 2048 --attSize 2048 --resSize 8192 --lr 0.0001 \
    --encoded_noise --workers 8 \
    --recons_weight 0.1 --feedback_loop 2 --a2 1 --a1 1 \
    --feed_lr 0.00001 --dec_lr 0.0001'''.format(split=n))
