import os
# Need to check: zsl/gzsl, nz/attSize, nclass_all, nepoch
# Also consider --syn_num: how many number of features should be generated per class

# Conducting different experiments accordign to different semantics
# when --class_embedding attribute
#       --nz 300 --attSize 115

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

'''
class_embedding = {'attribute': 115, 'action_class_w2v': 300, 'avg_desc_w2v': 300, 'fwv_k1_desc': 600,
                   'avg_img_googlenet': 1024, 'avg_img_googlenet_me': 1024,
                   'bert_embedding_768': 768, 'bert_embedding_1024': 1024,
                   'avg_img_resnet18': 512, 'avg_img_resnet50': 2048, 'avg_img_resnet101': 2048}

class_embedding = {'action_class_w2v': 300, 'avg_desc_w2v': 300,
                   'avg_img_googlenet': 1024, 'avg_img_googlenet_me': 1024, 'avg_img_resnet101': 2048}
'''

# avg_img_googlenet_me; avg_img_resnet101; 21-31
class_embedding = {'avg_desc_w2v': 300}
for c, dim in class_embedding.items():
    for n in range(1, 6):
        # n = n + 1
        os.system('''CUDA_LAUNCH_BLOCKING=1 python /ichec/work/tucom002c/single_free/train_tfvaegan.py \
        --dataset ucf101 --nclass_all 101 --zsl --manualSeed 806 \
        --dataroot /ichec/work/tud01/kaiqiang/action_datasets \
        --resultroot /ichec/work/tucom002c/single_free \
        --splits_path ucf101_semantics --split {split} \
        --action_embedding i3d --resSize 8192 \
        --class_embedding {semantics} --nz {semantics_dimension} --attSize {semantics_dimension} \
        --nepoch 100 --batch_size 64 --syn_num 400 \
        --preprocessing --cuda --gammaD 10 --gammaG 10 \
        --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
        --lr 0.0001 --workers 8 --encoded_noise  \
        --recons_weight 0.1 --feedback_loop 2 --a2 1 --a1 1 \
        --feed_lr 0.00001 --dec_lr 0.0001'''.format(split=n, semantics=c, semantics_dimension=dim))
