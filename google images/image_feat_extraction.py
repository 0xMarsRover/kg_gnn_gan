# Extracting visual features for images

import torch
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

if MODEL == 'googlenet':
    SIZE = 1024
    model = models.googlenet(pretrained=True)
    # Use the model object to select the desired layer
    layer = model._modules.get('avgpool')
    print(layer)
    # Set model to evaluation mode
    model.eval()

elif MODEL == 'resnet18':
    SIZE = 512
    model = models.resnet18(pretrained=True)
    # Use the model object to select the desired layer
    layer = model._modules.get('avgpool')
    print(layer)
    # Set model to evaluation mode
    model.eval()

elif MODEL == 'resnet50':
    SIZE = 2048
    model = models.resnet50(pretrained=True)
    # Use the model object to select the desired layer
    layer = model._modules.get('avgpool')
    print(layer)
    # Set model to evaluation mode
    model.eval()

elif MODEL == 'resnet101':
    SIZE = 2048
    model = models.resnet101(pretrained=True)
    # Use the model object to select the desired layer
    layer = model._modules.get('avgpool')
    print(layer)
    # Set model to evaluation mode
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


def get_vector(image_name):

    # Create a vector of zeros that will hold our feature vector
    my_embedding = torch.zeros(1, SIZE, 1, 1)

    try:
        # Load the image with Pillow library
        img = Image.open(image_name)
        img = img.convert('RGB')
        # Create a PyTorch Variable with the transformed image
        t_img = preprocess(img).unsqueeze(0)

        # Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        # Attach that function to our selected layer
        h = layer.register_forward_hook(copy_data)
        # Run the model on our transformed image
        model(t_img)
        # Detach our copy function from the layer
        h.remove()

    except Exception:
        pass

    # Return the feature vector
    return my_embedding.numpy()


if __name__ == "__main__":

    data_root = '/Volumes/Kellan/datasets/data_KG_GNN_GCN'

    if dataset == 'ucf101':
        image_data_root = os.path.join(data_root, 'ucf101_images_400')
        #image_data_root = os.path.join(data_root, 'test_images_400')

    elif dataset == 'hmdb51':
        image_data_root = os.path.join(data_root, 'hmdb51_images_400')
        #image_data_root = os.path.join(data_root, 'test_images_400')
    else:
        print("Please select a dataset")

    if os.path.exists(os.path.join(image_data_root, '.DS_Store')):
        os.remove(os.path.join(image_data_root, '.DS_Store'))
    else:
        # get all folder names as action classes
        all_classes = os.listdir(image_data_root)
        avg_action_embedding = np.empty((SIZE, 0))

        for action in all_classes:
            action_path = os.path.join(image_data_root, action)
            if os.path.exists(os.path.join(action_path, '.DS_Store')):
                # remove non-image file
                os.remove(os.path.join(action_path, '.DS_Store'))
            else:
                all_images_each_class = os.listdir(action_path)
                all_images_embedding = np.empty((SIZE, 0))

            for image in all_images_each_class:
                image_path = os.path.join(action_path, image)
                # reshape image features - (size, 1)
                image_feature = get_vector(image_path).reshape(SIZE, 1)
                # TODO: deal with invalid images (skip it)
                if np.all((image_feature == 0)):
                    print('Invalid Image !')
                else:
                    # stack all image features into one numpy array
                    all_images_embedding = np.hstack((all_images_embedding, image_feature))
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

