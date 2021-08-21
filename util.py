import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):

        if opt.dataset == "ucf101":
            # Argparse:
            # --dataroot '/content/drive/MyDrive/colab_data/action_datasets/'
            # --splits_path ucf101_semantics
            # --split = 1 (or 2 ... 30)
            # --dataset ucf101

            # load visual features for ucf101
            print("Action Embedding: ", opt.action_embedding)
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.dataset
                                     + "_" + opt.action_embedding + ".mat")
            feature = matcontent['features'].T
            label = matcontent['labels'].astype(int).squeeze() - 1

            # load action dataset splits and semantics
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" +
                                     opt.splits_path + "/split_" + str(opt.split) +
                                     "/att_splits.mat")
            print("split: ", str(opt.split))
            trainval_loc = matcontent['trainval_loc'].squeeze() - 1
            train_loc = matcontent['train_loc'].squeeze() - 1
            val_unseen_loc = matcontent['val_loc'].squeeze() - 1
            test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
            test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

            # selecting semantics
            print("Semantic Embedding: ", opt.class_embedding)
            self.attribute = torch.from_numpy(matcontent[opt.class_embedding].T).float()
            # L2 Norm.
            self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                      self.attribute.size(1))
            '''
            if opt.class_embedding == "attribute":
                self.attribute = torch.from_numpy(matcontent['attribute'].T).float()
                # L2 Norm.
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "action_class_w2v":
                self.attribute = torch.from_numpy(matcontent['action_class_w2v'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "avg_img_resnet101":
                self.attribute = torch.from_numpy(matcontent['avg_img_resnet101'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "avg_img_googlenet":
                self.attribute = torch.from_numpy(matcontent['avg_img_googlenet'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "avg_img_googlenet_me":
                self.attribute = torch.from_numpy(matcontent['avg_img_googlenet_me'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))

            elif opt.class_embedding == "avg_img_resnet18":
                self.attribute = torch.from_numpy(matcontent['avg_img_resnet18'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "avg_img_resnet50":
                self.attribute = torch.from_numpy(matcontent['avg_img_resnet50'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "avg_img_resnet101":
                self.attribute = torch.from_numpy(matcontent['avg_img_resnet101'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "avg_desc_w2v":
                self.attribute = torch.from_numpy(matcontent['avg_desc_w2v'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "fwv_k1_desc":
                self.attribute = torch.from_numpy(matcontent['fwv_k1_desc'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            # More semantic embedding .....
            else:
                print("Wrong semantics. In UCF101 splits file, att means word2vec and origin_att means attributes.")
            '''

        elif opt.dataset == "hmdb51":
            # Argparse:
            # --splits_path hmdb51_semantics
            # --split 1 (or 2 .... 30)
            # --dataset hmdb51

            # load visual features for hmdb51
            print("Action Embedding: ", opt.action_embedding)
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.dataset
                                     + "_" + opt.action_embedding + ".mat")
            feature = matcontent['features'].T
            label = matcontent['labels'].astype(int).squeeze() - 1

            # load action dataset splits and semantics
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" +
                                     opt.splits_path + "/split_" + str(opt.split) +
                                     "/att_splits.mat")
            print("split: ", str(opt.split))
            trainval_loc = matcontent['trainval_loc'].squeeze() - 1
            train_loc = matcontent['train_loc'].squeeze() - 1
            val_unseen_loc = matcontent['val_loc'].squeeze() - 1
            test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
            test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

            print("Semantic Embedding: ", opt.class_embedding)
            # selecting semantics
            if opt.class_embedding == "action_class_w2v":
                self.attribute = torch.from_numpy(matcontent['action_class_w2v'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "avg_img_resnet101":
                self.attribute = torch.from_numpy(matcontent['avg_img_resnet101'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "avg_img_googlenet":
                self.attribute = torch.from_numpy(matcontent['avg_img_googlenet'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "avg_img_googlenet_me":
                self.attribute = torch.from_numpy(matcontent['avg_img_googlenet_me'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "avg_img_resnet18":
                self.attribute = torch.from_numpy(matcontent['avg_img_resnet18'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "avg_img_resnet50":
                self.attribute = torch.from_numpy(matcontent['avg_img_resnet50'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "avg_img_resnet101":
                self.attribute = torch.from_numpy(matcontent['avg_img_resnet101'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "avg_desc_w2v":
                self.attribute = torch.from_numpy(matcontent['avg_desc_w2v'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "fwv_k1_desc":
                self.attribute = torch.from_numpy(matcontent['fwv_k1_desc'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            else:
                print("Wrong semantics. In HMDB51 splits file, att means word2vec.")

        else:
            print("Wrong dataset!")

        if not opt.validation:
            print("Disable cross validation mode")
            if opt.preprocessing:
                print('Preprocessing (MinMaxScaler)...')
                if opt.standardization:
                    print('Standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            print("Enable cross validation mode")
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))

        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)

    def next_seen_batch(self, seen_batch):
        idx = torch.randperm(self.ntrain)[0:seen_batch]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_att

