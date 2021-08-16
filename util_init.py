# This file is for Init. Experiment （5 seen + 5 unseen）
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
            # load visual features for ucf101
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.dataset
                                     + "_" + opt.action_embedding + ".mat")
            feature = matcontent['features'].T
            label = matcontent['labels'].astype(int).squeeze() - 1

            # load action dataset splits and semantics
            # for inistal exp. (20 classes)
            #matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + "att_split_6classes.mat")
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + "att_split_20classes.mat")

            # trainval_loc = matcontent['trainval_loc'].squeeze() - 1
            train_loc = matcontent['train_loc'].squeeze() - 1
            # val_unseen_loc = matcontent['val_loc'].squeeze() - 1
            #test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
            test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

            if opt.class_embedding == "att":
                self.attribute = torch.from_numpy(matcontent['original_att'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            elif opt.class_embedding == "wv":
                if opt.object:
                    # att_all: 300 + 900d
                    print("with object semantics:")
                    #print("append 3 objects - 1200d")
                    self.attribute = torch.from_numpy(matcontent['att_all'].T).float() # (101, 1200)
                    # action vector
                    #self.action = torch.vstack((self.attribute[:10, :300],
                    #                            self.attribute[10, :300],
                    #                           self.attribute[11:13, :300],
                    #                           self.attribute[13, :300],
                    #                           self.attribute[14:32, :300],
                    #                           self.attribute[32:34, :300],
                    #                           self.attribute[34:89, :300],
                    #                           self.attribute[89, :300],
                    #                           self.attribute[90:, :300]))

                    self.action = self.attribute[:, :300]

                    # best object vector
                    self.best_obj = torch.vstack((self.attribute[:10, 300:600],  # 0-9 (1st)
                                                  self.attribute[10, 600:900],  # 10 (2nd)
                                                  self.attribute[11:13, 300:600],  # 11, 12 (1st)
                                                  self.attribute[13, 600:900],  # 13 (2nd)
                                                  self.attribute[14:32, 300:600],  # 14-31 (1st)
                                                  self.attribute[32:34, 900:],  # 32, 33 (3rd)
                                                  self.attribute[34:89, 300:600],  # 34-88 (1st)
                                                  self.attribute[89, 900:],  # 89 (3rd)
                                                  self.attribute[90:, 300:600]))  # 90-100 (1st)
                    # The first object.
                    self.obj1 = self.attribute[:, :300]
                    # The second object.
                    self.obj2 = self.attribute[:, 600:900]
                    # The third object.
                    self.obj3 = self.attribute[:, 900:]

                    # Different cases:
                    ###############################################################################################
                    # Case: Best object
                    # seen: action + 1st object; Unseen: action + best
                    # seen: 3,8,20,26,55.78.88.95.98.100
                    # unseen (best object): 11 (2), 13(1), 14(2), 33(3), 34(3), 42(1), 73(1), 85(1), 90(3), 92(1)
                    #print("Append best object.")
                    #self.attribute = torch.vstack((self.attribute[:10, :600], # 0-9 (1st)
                    #                               torch.hstack((self.attribute[10, :300], self.attribute[10, 600:900])), # 10 (2nd)
                    #                              self.attribute[11:13, :600],  # 11, 12 (1st)
                    #                              torch.hstack((self.attribute[13, :300], self.attribute[13, 600:900])), # 13 (2nd)
                    #                              self.attribute[14:32, :600], # 14-31 (1st)
                    #                              torch.hstack((self.attribute[32:34, :300], self.attribute[32:34, 900:])), # 32, 33 (3rd)
                    #                              self.attribute[34:89, :600], # 34-88 (1st)
                    #                              torch.hstack((self.attribute[89, :300], self.attribute[89, 900:])), # 89 (3rd)
                    #                              self.attribute[90:, :600])   # 90-100 (1st)
                    #                              )


                    # Case: Repalce with best object
                    # seen: 1st object; Unseen: best object
                    # seen: 3,8,20,26,55.78.88.95.98.100
                    # unseen (best object): 11 (2), 13(1), 14(2), 33(3), 34(3), 42(1), 73(1), 85(1), 90(3), 92(1)
                    #print("Replace action with best object.")
                    #self.attribute = torch.vstack((self.attribute[:10, 300:600], # 0-9 (1st)
                    #                           self.attribute[10, 600:900], # 10 (2nd)
                    #                           self.attribute[11:13, 300:600],  # 11, 12 (1st)
                    #                           self.attribute[13, 600:900], # 13 (2nd)
                    #                           self.attribute[14:32, 300:600], # 14-31 (1st)
                    #                           self.attribute[32:34, 900:], # 32, 33 (3rd)
                    #                           self.attribute[34:89, 300:600], # 34-88 (1st)
                    #                           self.attribute[89, 900:], # 89 (3rd)
                    #                           self.attribute[90:, 300:600]))   # 90-100 (1st)

                    ####################################################################################
                    # Average Experiments
                    # Case: Average action with best object
                    #print("Average action with best object.")
                    #self.attribute = (self.action + self.best_obj) / 2

                    # Case: Average(Class, 1 object, 2 object, 3 object)
                    #print("Average(Class, 1 object, 2 object, 3 object)")
                    #self.attribute = (self.action + self.obj1 + self.obj2 + self.obj3) / 4

                    # Case: Average(1 object, 2 object, 3 object)
                    #print("Average(1 object, 2 object, 3 object)")
                    #self.attribute = (self.obj1 + self.obj2 + self.obj3) / 3

                    # Case: Average(Class, 1 object)
                    #print("Average(Class, 1 object)")
                    #self.attribute = (self.action + self.obj1) / 2

                    # Case: Average(Class, 2 object)
                    #print ("Average(Class, 2 object)")
                    #self.attribute = (self.action + self.obj2) / 2

                    # Case: Average(Class, 3 object)
                    #print ("Average(Class, 3 object)")
                    #self.attribute = (self.action + self.obj3) / 2

                    # Case: Average(Class, 1 object, 2 object)
                    #print ("Average(Class, 1 object, 2 object)")
                    #self.attribute = (self.action + self.obj1 + self.obj2) / 3

                    # Case: Average(Class, 1 object, 3 object)
                    #print ("Average(Class, 1 object, 3 object)")
                    #self.attribute = (self.action + self.obj1 + self.obj3) / 3

                    # Case: Average(Class, 2 object, 3 object)
                    print ("Average(Class, 2 object, 3 object)")
                    self.attribute = (self.action + self.obj2 + self.obj3) / 3


                    ####################################################################################
                    # Case 1: Replace action wv with object wv (300d)
                    #print("replace action wv with 1st object")
                    #self.attribute = self.attribute[:, 300:600]

                    #print("replace action wv with 2nd object")
                    #self.attribute = self.attribute[:, 600:900]

                    #print("replace action wv with 3rd object")
                    #self.attribute = self.attribute[:, 900:]

                    ################################################################################################

                    # Case 2: Append 1 object (600d)
                    #print("append 1st obj.")
                    #self.attribute = self.attribute[:, :600]

                    #print("append 2nd obj.")
                    # a[:,2:6] - including index 2 and excluding index 6
                    #self.attribute = torch.hstack((self.attribute[:, :300], self.attribute[:, 600:900]))

                    #print("append 3rd obj.")
                    #self.attribute = torch.hstack((self.attribute[:, :300], self.attribute[:, 900:]))

                    ################################################################################################

                    # Case 3: Append 2 objects (900d)
                    # 1st + 2nd
                    #print("append 2 objects (1st + 2nd)")
                    #self.attribute = self.attribute[:, :900]
                    # 1st + 3rd
                    #print("append 2 objects (1st + 3rd)")
                    #self.attribute = torch.hstack((self.attribute[:, :600], self.attribute[:, 900:]))
                    # 2nd + 3rd
                    #print("append 2 objects (2nd + 3rd)")
                    #self.attribute = torch.hstack((self.attribute[:, :300], self.attribute[:, 600:]))

                    ################################################################################################

                    print(self.attribute.shape)
                    self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                              self.attribute.size(1))
                elif opt.avg_wv:
                    self.attribute = torch.from_numpy(matcontent['att_all'].T).float()
                    self.action_wv = torch.from_numpy(matcontent['att'].T).float()
                    self.object1_wv = self.attribute[:, 300:600]
                    self.object2_wv = self.attribute[:, 600:900]
                    self.object3_wv = self.attribute[:, 900:]

                    # Case 1: avg action and 1st object
                    print("Averaging action and 1st object semantics")
                    self.attribute = torch.add(self.action_wv, self.object1_wv) / 2

                    print(self.attribute.shape)

                else:
                    print("without object semantics:")
                    self.attribute = torch.from_numpy(matcontent['att'].T).float()
                    self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                              self.attribute.size(1))
                    print(self.attribute.shape)
            else:
                print("Wrong semantics. In UCF101 splits file, att means word2vec and origin_att means attributes.")

        elif opt.dataset == "hmdb51":
            # load visual features for HMDB51
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/"
                                     + opt.dataset + "_" + opt.action_embedding + ".mat")
            # matcontent = sio.loadmat(opt.dataroot + "/" + opt.image_embedding_path + "/" + opt.image_embedding + ".mat")
            feature = matcontent['features'].T
            label = matcontent['labels'].astype(int).squeeze() - 1

            # load action dataset splits and semantics
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.dataset + "_semantics/" +
                                     "split_1/" + "att_splits.mat")
            # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
            trainval_loc = matcontent['trainval_loc'].squeeze() - 1
            train_loc = matcontent['train_loc'].squeeze() - 1
            val_unseen_loc = matcontent['val_loc'].squeeze() - 1
            #test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
            test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

            if opt.class_embedding == "wv":
                self.attribute = torch.from_numpy(matcontent['att'].T).float()
                self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                          self.attribute.size(1))
            else:
                print("Wrong semantics. In HMDB51 splits file, att means word2vec.")

        else:
            print("Wrong dataset!")

        '''
        # use above codes
        if opt.manual_att:
            print("Using manual_att")
            m_att = torch.from_numpy(np.load(opt.dataroot + "/ucf101_i3d/ucf101_manual_att.npy")).float()
            m_att /= m_att.pow(2).sum(1).sqrt().unsqueeze(1).expand(101,m_att.size(1))
            self.attribute = m_att
        '''

        if not opt.validation:
            print("Disable cross validation mode")
            if opt.preprocessing:
                print('Preprocessing (MinMaxScaler)...')
                if opt.standardization:
                    print('Standardization...')
                    scaler = preprocessing.StandardScaler()
                    # scaler_att = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                    # scaler_att = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[train_loc])
                #_test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(label[train_loc]).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                #self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                #self.test_seen_feature.mul_(1 / mx)
                #self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
                # Scaled and transformed (0,1) attributes (bce: binary class embedding)
                # self.bce_att = opt.bce_att
                # select either binary class embedding or norm class embedding for attributes
                # if opt.bce_att:
                #   temp_att = torch.from_numpy(scaler_att.fit_transform(self.original_att)).float()
                # else:
                #   temp_att = torch.from_numpy(scaler_att.fit_transform(self.attribute)).float()
                # mx_att = temp_att.max()
                # temp_att.mul_(1/mx)
                # self.bce_attribute = temp_att
                # self.bce_attribute_norm = self.bce_attribute/self.bce_attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0), self.attribute.size(1))

            else:
                self.train_feature = torch.from_numpy(feature[train_loc]).float()
                self.train_label = torch.from_numpy(label[train_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                #self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
                #self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            print("Enable cross validation mode")
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))

        self.ntrain = self.train_feature.size()[0]
        #self.ntest_seen = self.test_seen_feature.size()[0]
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
        # batch_bce_att = self.bce_attribute[batch_label]
        return batch_feature, batch_att  # batch_bce_att