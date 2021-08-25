# Extracting visual features for images

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import scipy.io as sio
import argparse

# using argparaser for running bash
parser = argparse.ArgumentParser()

# add arg.
parser.add_argument('--dataset', default='ucf101', help='ucf101 or hmdb51')
parser.add_argument('--model', default='googlenet', help='googlenet or resnet18 or resnet50 or resnet101')
opt = parser.parse_args()

# ucf101 or hmdb51
dataset = opt.dataset
print(dataset)
# googlenet resnet18, resnet50, resnet101
MODEL = opt.model
print(MODEL)


# GoogLeNet - 1024D
class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        googlenet = models.googlenet(pretrained=True)
        self.features = nn.Sequential(*list(googlenet.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


# ResNet18 - 512D
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


# ResNet50 - 2048D
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


# ResNet101 - 2048D
class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


if MODEL == 'googlenet':
    SIZE = 1024
    model = GoogleNet()
    model.eval()

elif MODEL == 'resnet18':
    SIZE = 512
    model = ResNet18()
    model.eval()

elif MODEL == 'resnet50':
    SIZE = 2048
    model = ResNet50()
    model.eval()

elif MODEL == 'resnet101':
    SIZE = 2048
    model = ResNet101()
    model.eval()

else:
    print("No pretrained model selected !")

# Image transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


if __name__ == "__main__":

    data_root = '/Volumes/Kellan/datasets/data_KG_GNN_GCN'

    if dataset == 'ucf101':
        all_classes = []
        image_data_root = os.path.join(data_root, 'ucf101_images_400')
        #image_data_root = os.path.join(data_root, 'test_images_400')
        # Get class names from a file
        with open("ucf101_class_index.txt") as fp:
            Lines = fp.readlines()
            for line in Lines:
                all_classes.append(line.strip())

    elif dataset == 'hmdb51':
        all_classes = []
        image_data_root = os.path.join(data_root, 'hmdb51_images_400')
        #image_data_root = os.path.join(data_root, 'test_images_400')

        # Get class names from a file
        with open("hmdb51_class_index.txt") as fp:
            Lines = fp.readlines()
            for line in Lines:
                all_classes.append(line.strip())
    else:
        print("Please select a dataset")

    if os.path.exists(os.path.join(image_data_root, '.DS_Store')):
        os.remove(os.path.join(image_data_root, '.DS_Store'))
    else:
        # get all folder names as action classes
        # TODO: check class order....
        # all_classes = os.listdir(image_data_root)
        avg_action_embedding = np.empty((SIZE, 0))

        for action in all_classes:
            action_path = os.path.join(image_data_root, action)
            print("Current Action Class is: ", action)
            if os.path.exists(os.path.join(action_path, '.DS_Store')):
                # remove non-image file
                os.remove(os.path.join(action_path, '.DS_Store'))
            else:
                all_images_each_class = os.listdir(action_path)
                all_images_embedding = np.empty((SIZE, 0))

            for image in all_images_each_class:
                image_path = os.path.join(action_path, image)
                try:
                    img = Image.open(image_path)
                    img = img.convert('RGB')
                    # preprocess image
                    t_img = preprocess(img).unsqueeze(0)
                    # Feature extraction
                    with torch.no_grad():
                        image_feature = model(t_img).numpy().T
                except Exception:
                    pass

                # deal with invalid images (skip it)
                if np.all((image_feature == 0)):
                    print('Invalid Image !')
                else:
                    # stack all image features into one numpy array
                    all_images_embedding = np.hstack((all_images_embedding, image_feature))
                    # (feature size, number of image)
                    print("all_images_embedding", all_images_embedding.shape)

            # Save each image representation for each action class
            sio.savemat(os.path.join(data_root, dataset + '_img_' + MODEL + '_features',
                                     dataset + '_' + action + '_all_img_' + MODEL + '.mat'),
                        {'all_img_' + MODEL: all_images_embedding})

            # Averaing image features for each class
            avg_image_embedding = np.mean(all_images_embedding, axis=1).reshape(SIZE, 1)
            # put all action embedding together - (Feature SIZE, number of classes)
            avg_action_embedding = np.hstack((avg_action_embedding, avg_image_embedding))
            print("avg_action_embedding", avg_action_embedding.shape)
            # Save averaged img Rep.
            sio.savemat(os.path.join(data_root, dataset + '_avg_img_' + MODEL + '.mat'),
                        {'avg_img_' + MODEL: avg_action_embedding})

