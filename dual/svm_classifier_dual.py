import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import util_dual
import numpy as np
import copy
from sklearn.metrics import confusion_matrix
from config_dual import opt
from sklearn import svm


class SVM_CLASSIFIER:
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, generalized=False):

        # ZSL:
        #   train_X: syn_unseen_feat;
        #   train_Y: unseen_label
        # GZSL:
        #   train_X: real_seen_feat + syn_unseen_feat;
        #   train_Y: seen_label + unseen_label

        self.train_X = _train_X.clone()
        self.train_Y = _train_Y.clone()

        # real seen feature and seen label for test
        self.test_seen_feature = data_loader.test_seen_feature.clone()
        self.test_seen_label = data_loader.test_seen_label
        # real unseen feature and unseen label for test
        self.test_unseen_feature = data_loader.test_unseen_feature.clone()
        self.test_unseen_label = data_loader.test_unseen_label

        # number of seen classes
        self.seenclasses = data_loader.seenclasses
        # number of unseen classes
        self.unseenclasses = data_loader.unseenclasses
        # number of all classes
        self.nclass = _nclass
        # input dimesion
        self.input_dim = _train_X.size(1)
        # number of classes for training
        self.ntrain = self.train_X.size()[0]
        # Init svm classifier
        #self.clf = svm.SVC(kernel='rbf', decision_function_shape='ovr', gamma='auto')
        self.clf = svm.SVC(kernel='linear', decision_function_shape='ovr')

        if generalized:
            # gzsl
            self.acc_seen, self.acc_per_seen, self.acc_unseen, self.acc_per_unseen, \
                self.H, self.best_model = self.fit_gzsl()
        else:
            # zsl
            self.acc, self.acc_per_class, self.cm = self.fit_zsl()

    # training for zsl
    def fit_zsl(self):
        best_acc = 0
        best_acc_per_class = []
        best_cm = []

        # svm training
        self.clf.fit(self.train_X, self.train_Y)
        print('SVM training Successful.')

        acc, acc_per_class, cm = self.val_zsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
        # print('acc %.4f' % (acc))
        if acc > best_acc:
            best_acc = acc
            best_acc_per_class = acc_per_class
            best_cm = cm
        return best_acc, best_acc_per_class, best_cm

    # Parameter: test_label is integer
    # Validating for zsl
    def val_zsl(self, test_X, test_label, target_classes):
        '''
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        '''
        # prediction stage
        predicted_label = self.clf.predict(test_X)
        print('SVM testing successful')

        acc, acc_per_class = self.compute_per_class_acc(util_dual.map_label(test_label, target_classes),
                                                        predicted_label, target_classes.size(0))

        cm = self.compute_confusion_matrix(util_dual.map_label(test_label, target_classes),
                                           predicted_label, target_classes.size(0))
        return acc, acc_per_class, cm

    # training for gzsl
    def fit_gzsl(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        best_cm = []
        best_model = copy.deepcopy(self.model)
        # early_stopping = EarlyStopping(patience=20, verbose=True)

        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
            # Set evaluation mode
            self.model.eval()
            acc_seen, acc_per_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen, acc_per_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_acc_per_seen = acc_per_seen
                best_acc_per_useen = acc_per_unseen
                best_unseen = acc_unseen
                best_H = H
                best_model = copy.deepcopy(self.model)
        return best_seen, best_acc_per_seen, best_unseen, best_acc_per_useen, best_H, best_model

    # Validating for gzsl
    def val_gzsl(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            if self.cuda:
                with torch.no_grad():
                    inputX = Variable(test_X[start:end].cuda())
            else:
                with torch.no_grad():
                    inputX = Variable(test_X[start:end])
            output = self.model(inputX)
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc, acc_per_class = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc, acc_per_class

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = []
        n = 0
        for i in target_classes:
            idx = (test_label == i)
            acc = torch.sum(test_label[idx] == predicted_label[idx]) / torch.sum(idx)
            acc_per_class = np.append(acc_per_class, acc)
        #acc_per_class /= target_classes.size(0)
            acc_mean = acc_per_class.mean()
            n += 1
        return acc_mean, acc_per_class

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]) / torch.sum(idx)
            acc_mean = acc_per_class.mean()
        return acc_mean, acc_per_class

    # New function: get confusion matrix
    def compute_confusion_matrix(self, test_label, predicted_label, nclass):
        return confusion_matrix(test_label, predicted_label)

    def compute_dec_out(self, test_X, new_size):
        start = 0
        ntest = test_X.size()[0]
        new_test_X = torch.zeros(ntest, new_size)

        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            if self.cuda:
                with torch.no_grad():
                    inputX = Variable(test_X[start:end].cuda())
            else:
                with torch.no_grad():
                    inputX = Variable(test_X[start:end])
            feat1 = self.netDec(inputX)
            feat2 = self.netDec.getLayersOutDet()
            new_test_X[start:end] = torch.cat([inputX, feat1, feat2], dim=1).data.cpu()
            start = end
        return new_test_X

