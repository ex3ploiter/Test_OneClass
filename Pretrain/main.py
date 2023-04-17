
import os
import numpy as np
from torchvision.datasets import ImageFolder
import torch
from torchvision.datasets import ImageFolder
import torch
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
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
import numpy as np


from utils import write_in_file,getLoaders,auc_softmax_adversarial,auc_softmax
from model import PretrainedModel
from torchattacks import FGSM, PGD


def performTest(attack_eps,attack_Type=8):
    
    attack_steps = 10
    attack_alpha = 2/255    

    for normal_class in range(10):
        clean_auc = []
        adv_auc = []


        bin_train_loader,Test_loader_mir=getLoaders(normal_class)
        

    
        import torch.optim as optim
        from tqdm import tqdm
        import torch.nn as nn

        
        model = PretrainedModel(18,attack_Type).cuda()

        attack = PGD(model, eps=attack_eps,alpha=attack_alpha, steps=attack_steps)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00005)
        for epoch in range(15):
            total_loss, total_num = 0.0, 0
            loss = nn.CrossEntropyLoss()
            train_bar =  tqdm(bin_train_loader, desc='Train   Binary Classifier ...')
            for (img1, Y ) in train_bar:
                model.train()
                attack = PGD(model, eps=attack_eps,alpha= attack_alpha,steps=attack_steps)
                adv_data = attack(img1, Y)
                optimizer.zero_grad()


                out_1 = model(adv_data.cuda()) 
                loss_ = loss(out_1,Y.cuda())  
                loss_.backward()
                optimizer.step()
                total_num += adv_data.size(0)
                total_loss += loss_.item() * adv_data.size(0)
                total_num += bin_train_loader.batch_size
                total_loss += loss_.item() * bin_train_loader.batch_size
                train_bar.set_description('Train Robust Epoch :  {} , Clf_B Robust Loss: {:.4f}'.format(epoch ,  total_loss / total_num))
            attack = PGD(model, eps=attack_eps,alpha= attack_alpha,steps=attack_steps)
            clean_auc.append(auc_softmax(model, Test_loader_mir))
            adv_auc.append(auc_softmax_adversarial(model, Test_loader_mir, attack))

        write_in_file(f"Fashion_MNIST_{normal_class}_Clean.txt", clean_auc)
        write_in_file(f"Fashion_MNIST_{normal_class}_Adversarial.txt", adv_auc)

        print("----------------------------------------------")



performTest(8/255,attack_Type=8)
performTest(4/255,attack_Type=4)