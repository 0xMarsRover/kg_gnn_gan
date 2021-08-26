import os
# Need to check: zsl/gzsl, nz/attSize, nclass_all, nepoch
# Also consider --syn_num: how many number of features should be generated per class

# Conducting different experiments accordign to different semantics
# when --class_embedding action_class_w2v
#       --nz 300 --attSize 300

# when --class_embedding avg_img_resnet18
#       --nz 512 --attSize 512

# when --class_embedding avg_img_resnet50
#       --nz 2048 --attSize 2048

# when --class_embedding avg_img_resnet101
#       --nz 2048 --attSize 2048

# when --class_embedding avg_img_googlenet
#       --nz 1024 --attSize 1024

# when --class_embedding avg_img_googlenet_me
#       --nz 1024 --attSize 1024

# when --class_embedding avg_desc_w2v
#       --nz 300 --attSize 300

# when --class_embedding fwv_k1_desc
#       --nz 600 --attSize 600

# Number of splits range(30)
# when using --class_embedding img_avg, set "CUDA_LAUNCH_BLOCKING=1 python train_vaegan.py ...."

#TODO: Conducting 30-split experiment in one run (by for-loop)
# 1. recording results into a local file for each split:
#       a. The best Epoch
#       b. The best average accuracy over test classes
#       c. In the best epoch:
#            i. accuracy per class
#           ii. confusion matrix


for n in range(1, 2):
    # n = n + 1
    os.system('''CUDA_LAUNCH_BLOCKING=1 python /content/kg_gnn_gan/train_tfvaegan.py \
    --dataset ucf101 --nclass_all 101 --zsl \
    --dataroot /content/drive/MyDrive/colab_data/action_datasets \
    --splits_path ucf101_semantics --split {split} \
    --action_embedding i3d --resSize 8192 \
    --class_embedding avg_img_googlenet_me --nz 1024 --attSize 1024 \
    --nepoch 50 --batch_size 64 --syn_num 600 \
    --preprocessing --cuda --gammaD 10 --gammaG 10 \
    --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
    --lr 0.0001 --workers 8 --encoded_noise  \
    --recons_weight 0.1 --feedback_loop 2 --a2 1 --a1 1 \
    --feed_lr 0.00001 --dec_lr 0.0001'''.format(split=n))
