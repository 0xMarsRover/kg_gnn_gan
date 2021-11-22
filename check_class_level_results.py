import os
import scipy.io as sio
import numpy as np
import csv

# TODO: save unseen labels for all splits
# Mac
#data_root = '/Volumes/GoogleDrive/My Drive/colab_data/action_datasets'
# Windows
data_root = 'G:\\My Drive\\colab_data\\action_datasets'
dataset = 'hmdb51'  # ucf101

# /Volumes/GoogleDrive/My Drive/colab_data/action_datasets/hmdb51'
dataset_path = os.path.join(data_root, dataset)
mat_i3d = sio.loadmat(dataset_path + '/hmdb51_i3d.mat')

# for loop for 30 splits
for i in range(1, 31):
    # /Volumes/GoogleDrive/My Drive/colab_data/action_datasets/hmdb51/hmdb51_semantics/split_1/att_splits.mat'
    split_path = os.path.join(dataset_path, 'hmdb51_semantics/' + 'split_' + str(i))
    mat_split = sio.loadmat(split_path + '/att_splits.mat')

    all_labels = mat_i3d['labels']
    all_labels_names = mat_split['allclasses_names']
    test_unseen_loc = mat_split['test_unseen_loc']

    all_labels = all_labels.squeeze() - 1
    test_unseen_loc = test_unseen_loc.squeeze() - 1

    # print(all_labels)
    # print(len(all_labels))
    # print(test_unseen_loc)
    # print(len(test_unseen_loc))

    all_labels_names_clean = []
    for i in range(len(all_labels_names)):
        all_labels_names_clean.append(all_labels_names[i][0][0])

    # All labels names
    all_labels_names = np.array(all_labels_names_clean)

    unseen_label = all_labels[test_unseen_loc]

    # print(len(np.unique(unseen_label))) # should be 25
    # print(len(all_labels_names[unseen_label])) # split 1: 3616 testing videos

    unique_unseen_label = np.unique(unseen_label)
    unseen_label_name = []
    for i in range(len(unique_unseen_label)):
        unseen_label_name.append(all_labels_names[unique_unseen_label[i]])
    print(unseen_label_name)

    with open('hmdb51_unseen_label_all_splits.csv', 'a+') as f:
        write = csv.writer(f)
        write.writerow(unseen_label_name)


