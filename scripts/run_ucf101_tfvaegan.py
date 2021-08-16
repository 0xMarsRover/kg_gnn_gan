#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
"""
# CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8
os.system('''python /content/gzsl/zero-shot-actions/train_tfvaegan.py \
--encoded_noise --gzsl_od --workers 8 --nclass_all 101 \
--dataset ucf101 --dataroot /content/drive/MyDrive/colab_data/action_datasets \
--syn_num 600 --preprocessing --cuda --gammaD 10 --gammaG 10 \
--action_embedding i3d --class_embedding att \
--nepoch 100 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--batch_size 64 --nz 115 --attSize 115 --resSize 8192 --lr 0.0001 \
--recons_weight 0.1 --feedback_loop 2 --a2 1 --a1 1 --feed_lr 0.00001 --dec_lr 0.0001''')
"""

# --dataset ucf101_i3d/split_{split}
# --image_embedding_path ucf101_i3d

"""
# Tryout: Inital experiment (ZSL setting)
# 3 seen classes + 3 unseen classes
# case 1: i3d(8192d) + w2v(300d)
# case 2: i3d(8192d) + w2v(1200d)
# Need to check: zsl/gzsl, nz/attSize, nclass_all, nepoch
os.system('''python /content/gzsl/zero-shot-actions/train_tfvaegan.py \
--encoded_noise --workers 8 --nclass_all 6 \
--dataset ucf101 --dataroot /content/drive/MyDrive/colab_data/action_datasets_small \
--syn_num 600 --preprocessing --cuda --gammaD 10 --gammaG 10 \
--action_embedding i3d --class_embedding wv \
--nepoch 20 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--batch_size 64 --nz 1200 --attSize 1200 --resSize 8192 --lr 0.0001 \
--recons_weight 0.1 --feedback_loop 2 --a2 1 --a1 1 --feed_lr 0.00001 --dec_lr 0.0001''')

"""

# Tryout: Inital experiment (ZSL setting)
# 10 seen classes + 10 unseen classes
# case 1: action (300d)
# case 2: 1st obj (300d)
# case 3: 2nd obj (300d)
# case 4: 3rd obj (300d)
# case 5: action + 1st obj (600d)
# case 6: action + 2nd obj (600d)
# case 7: action + 3rd obj (600d)

# Need to check: zsl/gzsl, nz/attSize, nclass_all, nepoch
# --object
# --avg_wv
os.system('''python /content/gzsl/zero-shot-actions/train_tfvaegan.py \
--encoded_noise --object --workers 8 --nclass_all 20 \
--dataset ucf101 --dataroot /content/drive/MyDrive/colab_data/action_datasets_small \
--syn_num 600 --preprocessing --cuda --gammaD 10 --gammaG 10 \
--action_embedding i3d --class_embedding wv \
--nepoch 30 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--batch_size 64 --nz 300 --attSize 300 --resSize 8192 --lr 0.0001 \
--recons_weight 0.1 --feedback_loop 2 --a2 1 --a1 1 --feed_lr 0.00001 --dec_lr 0.0001''')
