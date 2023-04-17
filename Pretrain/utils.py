
import os
import numpy as np
from torchvision.datasets import ImageFolder
import torch
from torchvision.datasets import ImageFolder
import torch
from torchvision.datasets import FashionMNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.models import resnet
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset 

import numpy as np

import time

from torch.utils.data import DataLoader
import torchvision
from PIL import Image
import glob
import os
import shutil
from glob import glob
import os
import pandas as pd 
import cv2
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt



import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)



def show_samples(x):
  x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
  img = image_grid(x)
  plt.figure(figsize=(8,8))
  plt.axis('off')
  plt.imshow(img)
  plt.show()

def image_grid(x):
  size =224# config.data.image_size train_model
  channels =3# config.data.num_channels
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  return img


import zipfile
 

 

class MyDataset_Binary(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x, labels,transform):
        'Initialization'
        super(MyDataset_Binary, self).__init__()
        self.labels = labels
        self.x = x
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

  def __getitem__(self, index):
        'Generates one sample of data'
        if self.transform is None:
          x =  self.x[index]
          y = self.labels[index]
        else:
          x = self.transform(self.x[index])
          y = self.labels[index]
       
        return x, y


class Exposure(Dataset):
    def __init__(self, root, transform=None, count=None):
        self.transform = transform
        image_files = glob(os.path.join(root, "*.png"))
        final_length = min(len(image_files), count)
        self.image_files = image_files[:final_length]

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        image =  self.transform(image)

        return image

    def __len__(self):
        return len(self.image_files)



class ImageNetExposure(Dataset):
    def __init__(self, root, count, transform=None):
        self.transform = transform
        image_files = glob(os.path.join(root, "*", "*.JPEG"))
        random.shuffle(image_files)
        final_length =  len(image_files) 
        self.image_files = image_files[:final_length]

        self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image,1

    def __len__(self):
        return len(self.image_files)


import random
orig_transform = transforms.Compose([
            transforms.Resize([32,32]),
            transforms.ToTensor()
        ])
"""
imagenet = ImageNetExposure('/kaggle/working/one_class_train/', transform=orig_transform, count=20)
imagenet_loader = torch.utils.data.DataLoader(imagenet, shuffle=True, batch_size=5000)
imagenet_iterator = iter(imagenet_loader)
imagenet_data = next(imagenet_iterator) 
"""
       


def auc_softmax_adversarial(model, test_loader, test_attack):
  soft = torch.nn.Softmax(dim=1)
  anomaly_scores = []
  test_labels = []
  print('AUC Adversarial Softmax Started ...')
  with tqdm(test_loader, unit="batch") as tepoch:
 
    torch.cuda.empty_cache()
    for i, (data, target) in enumerate(tepoch):
      model.eval()
      data, target = data.to(device), target.to(device)
      labels = target.to(device)
      adv_data = test_attack(data, target)
      output = model(adv_data)
      probs = soft(output).squeeze()
      anomaly_scores += probs[:, 1].detach().cpu().numpy().tolist()
      test_labels += target.detach().cpu().numpy().tolist()

  auc = roc_auc_score(test_labels, anomaly_scores)
  print(f'AUC Adversairal - Softmax - score  is: {auc * 100}')
  return auc


def auc_softmax(model, test_loader   ):
  soft = torch.nn.Softmax(dim=1)
  anomaly_scores = []
  test_labels = []
  print('AUC Softmax Started ...')
  with torch.no_grad():
    with tqdm(test_loader, unit="batch") as tepoch:

      torch.cuda.empty_cache()
      for i, (data, target) in enumerate(tepoch):
        model.eval()
        data, target = data.to(device), target.to(device)
        output = model(data)
        probs = soft(output).squeeze()
        anomaly_scores += probs[:, 1].detach().cpu().numpy().tolist()
        test_labels += target.detach().cpu().numpy().tolist()

  auc = roc_auc_score(test_labels, anomaly_scores)
  print(f'AUC - Softmax - score is: {auc * 100}')
  
  return auc


def write_in_file(title, lst):
    with open(title, 'w') as f:
        for i in range(len(lst)):
            f.write(f"epoch {i}: {lst[i]}\n")
            
def getLoaders(normal_class):
    trans_to_224 = transforms.Compose([  transforms.Resize((224,224))])

    

    img_transform_32 = transforms.Compose([ transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),transforms.Resize((32,32))])

    

    FashionMNIST_train =MNIST('/mnt/new_drive/ExposureExperiment/data/Datasets/MNIST', train=True, download=False, transform=img_transform_32)
    images_train=[]
    
    for x,y in FashionMNIST_train:
        if y==normal_class:
            images_train.append(x)
    FashionMNIST_train_tensor=torch.stack(images_train)



    FashionMNIST_testset =MNIST('/mnt/new_drive/ExposureExperiment/data/Datasets/MNIST', train=False, download=False,transform=img_transform_32)
    
    test_labels=[]
    images_test=[]
    for x,y in FashionMNIST_testset:
        images_test.append(x)
        if y==normal_class:
            test_labels.append(0)
        else:test_labels.append(1)
    images_test_t=torch.stack(images_test)
    #print(images_test_t.shape)

    mir_data_set_test=MyDataset_Binary(images_test_t,test_labels,trans_to_224 )
    Test_loader_mir = DataLoader(mir_data_set_test, batch_size=50, shuffle=False)

    _transform_224 = transforms.Compose([
                transforms.Resize([224,224]) 
            ])

    imagenet_data=torch.from_numpy(np.load("/mnt/new_drive/Masoud_WorkDir/Test_OneClass/OE_imagenet_32x32.npy"))[:FashionMNIST_train_tensor.shape[0]]

    fake_label=[1]*imagenet_data.shape[0]
    fake_dataset2 =MyDataset_Binary(imagenet_data,fake_label,_transform_224 )

    normal_label=[0]*FashionMNIST_train_tensor.shape[0]
    train_dataset=MyDataset_Binary(FashionMNIST_train_tensor ,normal_label,_transform_224 )

    train_dev_sets = torch.utils.data.ConcatDataset([train_dataset ,fake_dataset2])
    bin_train_loader = torch.utils.data.DataLoader(train_dev_sets, shuffle=True, batch_size=40)

    print("Trainset: ", len(train_dev_sets))
    print("Testset: ", len(mir_data_set_test))
    
    return bin_train_loader,Test_loader_mir
