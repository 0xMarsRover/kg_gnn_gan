import os
# Need to check: zsl/gzsl, nz/attSize, nclass_all, nepoch
# Also consider --syn_num: how many number of features should be generated per class
# --combined_syn avg/sum/concat_pca
# Number of splits range(30)

'''
class_embedding = {'action_class_w2v': 300, 'avg_desc_w2v': 300, 'fwv_k1_desc': 600,
                   'avg_img_googlenet': 1024, 'avg_img_googlenet_me': 1024, 'avg_img_resnet101': 2048}
'''

# Colab:
#       python /content/kg_gnn_gan/dual/train_tfvaegan_dual.py
#       --dataroot /content/drive/MyDrive/colab_data/action_datasets
#       --resultroot /content/drive/MyDrive/colab_data/KG_GCN_GAN

# Kay:
#       python /ichec/home/users/kaiqiang/kay_classifier_dual_gan/dual/train_tfvaegan_dual.py
#       --dataroot /ichec/work/tud01/kaiqiang/action_datasets
#       --resultroot /ichec/home/users/kaiqiang/kay_classifier_dual_gan/dual


class_embedding_text = {'action_class_w2v': 300}
class_embedding_image = {'avg_img_resnet101': 2048}

# but need to consider imbalance issue if doing GZSL:
# training class has around 120 videos, so the number of generated unseen features may not be too large.
# previous exp. used 800
syn_num = [600]  # 200, 400, 600, 800, 1000, 1200, 1400, 1600
# syn_num = [1000, 1200, 1400]
# syn_num = [1600, 1800, 2000]

#fusion_methods = ['max']    # ['avg', 'sum', 'max', 'min']
#classifiers = ['logsoftmax']   # ['svm', 'rf', 'logsoftmax']

for c_t, dim_t in class_embedding_text.items():
    for c_i, dim_i in class_embedding_image.items():
        for syn in syn_num:
            for n in range(17, 31):

                os.system('''CUDA_LAUNCH_BLOCKING=1 python /ichec/work/tucom002c/dual_free/dual/train_tfvaegan_dual.py \
                --dataset hmdb51 --nclass_all 51 --nclass_seen 26 --gzsl_od --manualSeed 806 \
                --dataroot /ichec/work/tud01/kaiqiang/action_datasets \
                --resultroot /ichec/work/tucom002c/dual_free \
                --splits_path hmdb51_semantics --split {split} \
                --action_embedding i3d --resSize 8192 \
                --class_embedding_text {semantics_t} --nz_text {semantics_dimension_t} --attSize_text {semantics_dimension_t} \
                --class_embedding_image {semantics_i} --nz_image {semantics_dimension_i} --attSize_image {semantics_dimension_i} \
                --nepoch 100 --batch_size 64 --syn_num {syn_num} \
                --preprocessing --cuda --gammaD 10 --gammaG 10 \
                --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
                --lr 0.0001 --workers 8 --encoded_noise  \
                --recons_weight 0.1 --feedback_loop 2 --a2 1 --a1 1 \
                --feed_lr 0.00001 --dec_lr 0.0001'''.format(split=n, semantics_t=c_t, semantics_dimension_t=dim_t,
                                                            semantics_i=c_i, semantics_dimension_i=dim_i,
                                                            syn_num=syn))