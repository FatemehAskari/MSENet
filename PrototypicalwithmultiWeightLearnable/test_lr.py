from prototypicalNetwork import PrototypicalNetworks
import argparse
import os.path as osp
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric,evaluate_on_one_task,evaluate
from easyfsl.datasets import MiniImageNet
from torchvision import transforms
from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader
from easyfsl.utils import plot_images, sliding_average
from torchvision.models import resnet18
from torch import nn, optim
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import csv
import os
import numpy as np


def write_csv(dir,arr):
    with open(dir, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow('accuracy_test')
        for j in arr:
            writer.writerow([j])      
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))
    set_gpu(args.gpu)
    ImageSize=84
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
        transforms.Resize([ImageSize, ImageSize])
    ])
        
    convolutional_network=resnet18(pretrained=True)
    convolutional_network.fc=nn.Flatten()
    
    weights=[]
    for i in range(1,20):
        we=[1 + (k)*(0.1*i) for k in range(5)]
        print(we)
        weights.append(we)
    

    filename1 = f'lr_effect/1shotCSVfile_test_change/InfoTest_1shot_we=0_lr=0.001_test=150_mean.csv'    
    dir1=f'lr_effect/save/proto-1_change/newprototypical_true_0.pth'
    
    filename2 = f'lr_effect/1shotCSVfile_test_change/InfoTest_1shot_we=1_lr=0.001_test=150_mean.csv'    
    dir2=f'lr_effect/save/proto-1_change/newprototypical_true_1.pth'
    
    m1=[]
    m2=[]
    root_directory = "../data/mini_imagenet/miniimagenet"
            
    for i in range(50):            
        test_set = MiniImageNet(root=root_directory, split="test", training=False, transform=transform)
        test_set.get_labels = lambda: [test_set[i][1] for i in range(len(test_set))]   
        test_sampler = TaskSampler(
        test_set, n_way=args.test_way, n_shot=args.shot, n_query=args.query, n_tasks=150
        )
        test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,)

     
        model1=PrototypicalNetworks(convolutional_network,weightlearnable=True,weight=weights[18]).cuda()
        model2=PrototypicalNetworks(convolutional_network,weightlearnable=True,weight=weights[18]).cuda()
        
        model1.load_state_dict(torch.load(dir1))
        model1.eval()
        
        model2.load_state_dict(torch.load(dir2))
        model2.eval()
        
        accuracy1=evaluate(test_loader,model=model1)
        accuracy2=evaluate(test_loader,model=model2)
        
        m1.append(accuracy1)
        m2.append(accuracy2)
        
        
        
    print(f'**lr=0.0011mean: {np.mean(m1)} , var: {np.var(m1)}')
    print(f'**lr=0.0011mean: {np.mean(m2)} , var: {np.var(m2)}')
    
                            
    write_csv(filename1,m1)                                
    write_csv(filename2,m2)



 
    