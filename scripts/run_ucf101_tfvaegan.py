import os
# Need to check: zsl/gzsl, nz/attSize, nclass_all, nepoch
# Also consider --syn_num: how many number of features should be generated per class

# Conducting different experiments
# when --class_embedding attribute
#       --nz 115 --attSize 115

# when --class_embedding action_class_w2v
#       --nz 300 --attSize 300

# when --class_embedding avg_img_resnet101
#       --nz 2048 --attSize 2048

# when --class_embedding avg_img_googlenet
#       --nz 1024 --attSize 1024

# when --class_embedding avg_desc_w2v
#       --nz 300 --attSize 300

# when --class_embedding fwv_k1_desc
#       --nz 600 --attSize 600

# Number of splits range(30)
# when using --class_embedding img_avg, set "CUDA_LAUNCH_BLOCKING=1 python train_vaegan.py ...."

for n in range(1, 2):
    # n = n + 1
    os.system('''CUDA_LAUNCH_BLOCKING=1 python /content/kg_gnn_gan/train_tfvaegan.py \
    --nclass_all 101 --dataset ucf101 --zsl \
    --dataroot /content/drive/MyDrive/colab_data/action_datasets \
    --splits_path ucf101_semantics --split {split} \
    --action_embedding i3d --class_embedding avg_img_googlenet \
    --nepoch 50 --batch_size 64 \
    --syn_num 600 --preprocessing --cuda --gammaD 10 --gammaG 10 \
    --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
    --nz 1024 --attSize 1024 --resSize 8192 --lr 0.0001 \
    --encoded_noise --workers 8 \
    --recons_weight 0.1 --feedback_loop 2 --a2 1 --a1 1 \
    --feed_lr 0.00001 --dec_lr 0.0001'''.format(split=n))
