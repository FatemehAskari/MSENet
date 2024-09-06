
from prototypicalNetwork_attention import PrototypicalNetworks1
from prototypicalNetwork_attention import PrototypicalNetworks2
from prototypicalNetwork_attention import PrototypicalNetworks3
from prototypicalNetwork_attention import PrototypicalNetworks4

import argparse
import os.path as osp
import os
from utils_attention import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric,evaluate_on_one_task,evaluate
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
import pandas as pd
import csv
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os
import re
# from easyfsl.datasets import MiniImageNet

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
    parser.add_argument('--shot', type=int, default=5)
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
    

    filename1 = f'MiniImageNet/baseline/InfoTest_5shot_mean.csv'
    filename2 = f'MiniImageNet/baseline+multiscale/InfoTest_5shot_mean.csv'
    filename3 = f'MiniImageNet/baseline+multiscale+weight/InfoTest_5shot_mean.csv'
    
    filename4 = f'MiniImageNet/baseline+multiscale+weight+attention/InfoTest_5shot_mean0.0_0.csv'
    filename5 = f'MiniImageNet/baseline+multiscale+weight+attention/InfoTest_5shot_mean0.0_1.csv'
    
    
    filename6 = f'MiniImageNet/baseline+multiscale+weight+attention/InfoTest_5shot_mean0.1_0.csv'
    filename7 = f'MiniImageNet/baseline+multiscale+weight+attention/InfoTest_5shot_mean0.1_1.csv'
    
    
    filename8 = f'MiniImageNet/baseline+multiscale+weight+attention/InfoTest_5shot_mean0.2_0.csv'
    filename9 = f'MiniImageNet/baseline+multiscale+weight+attention/InfoTest_5shot_mean0.2_1.csv'
    
    filename10 = f'MiniImageNet/baseline+multiscale+weight+attention/InfoTest_5shot_mean0.3_1.csv'
    filename11 = f'MiniImageNet/baseline+multiscale+weight+attention/InfoTest_5shot_mean0.3_0.csv'
    
    
    
    dir1=f'save/baseline/5shot/newprototypical_lr=_0.0001.pth'
    dir2=f'save/baseline+multiscal/5shot/newprototypical_lr=_0.0001.pth'    
    dir3=f'save/baseline+multicale+weight/5shot/newprototypical_true_0.pth'
    
    dir4=f'save/baseline+multiscale+weight+attention/5shot/0.0/model_true_$0.0_$0.pth'
    dir5=f'save/baseline+multiscale+weight+attention/5shot/0.0/model_true_$0.0_$1.pth'
    
    dir6=f'save/baseline+multiscale+weight+attention/5shot/0.1/model_true_$0.1_$0.pth'    
    dir7=f'save/baseline+multiscale+weight+attention/5shot/0.1/model_true_$0.1_$1.pth'
    
    dir8=f'save/baseline+multiscale+weight+attention/5shot/0.2/model_true_$0.2_$0.pth'
    dir9=f'save/baseline+multiscale+weight+attention/5shot/0.2/model_true_$0.2_$1.pth'
    
    dir10=f'save/baseline+multiscale+weight+attention/5shot/0.3/model_true_$0.3_$0.pth'
    dir11=f'save/baseline+multiscale+weight+attention/5shot/0.3/model_true_$0.3_$1.pth'
    # column_names = ['test_accuracy']
    
    m1=[]
    m2=[]
    m3=[]
    
    m4=[]
    m5=[]
    
    m6=[]
    m7=[]
    
    m8=[]
    m9=[]
    
    m10=[]
    m11=[]
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

     
        model1=PrototypicalNetworks1(convolutional_network).cuda()
        model2=PrototypicalNetworks2(convolutional_network).cuda()
        model3=PrototypicalNetworks3(convolutional_network,weightlearnable=True,weight=weights[18]).cuda() 
        
        model4=PrototypicalNetworks4(convolutional_network,weightlearnable=True,weight=weights[18],gamma1=0,gamma2=0).cuda()         
        model5=PrototypicalNetworks4(convolutional_network,weightlearnable=True,weight=weights[18],gamma1=0,gamma2=0).cuda()
        
        model6=PrototypicalNetworks4(convolutional_network,weightlearnable=True,weight=weights[18],gamma1=0,gamma2=0).cuda()                 
        model7=PrototypicalNetworks4(convolutional_network,weightlearnable=True,weight=weights[18],gamma1=0,gamma2=0).cuda()
                            
        model8=PrototypicalNetworks4(convolutional_network,weightlearnable=True,weight=weights[18],gamma1=0,gamma2=0).cuda()         
        model9=PrototypicalNetworks4(convolutional_network,weightlearnable=True,weight=weights[18],gamma1=0,gamma2=0).cuda()
        
        model10=PrototypicalNetworks4(convolutional_network,weightlearnable=True,weight=weights[18],gamma1=0,gamma2=0).cuda()         
        model11=PrototypicalNetworks4(convolutional_network,weightlearnable=True,weight=weights[18],gamma1=0,gamma2=0).cuda()
        
        model1.load_state_dict(torch.load(dir1))
        model1.eval()
        
        model2.load_state_dict(torch.load(dir2))
        model2.eval()
        
        model3.load_state_dict(torch.load(dir3))
        model3.eval()
        
        model4.load_state_dict(torch.load(dir4))
        model4.eval()
        model5.load_state_dict(torch.load(dir5))
        model5.eval()
        
        
        model6.load_state_dict(torch.load(dir6))
        model6.eval()
        model7.load_state_dict(torch.load(dir7))
        model7.eval()
        
        model8.load_state_dict(torch.load(dir8))
        model8.eval()
        model9.load_state_dict(torch.load(dir9))
        model9.eval()
        
        
        model10.load_state_dict(torch.load(dir10))
        model10.eval()
        model11.load_state_dict(torch.load(dir11))
        model11.eval()
        
        accuracy1=evaluate(test_loader,model=model1)
        accuracy2=evaluate(test_loader,model=model2)
        accuracy3=evaluate(test_loader,model=model3)
        
        accuracy4=evaluate(test_loader,model=model4)
        accuracy5=evaluate(test_loader,model=model5)
        
        accuracy6=evaluate(test_loader,model=model6)
        accuracy7=evaluate(test_loader,model=model7)
        
        accuracy8=evaluate(test_loader,model=model8)
        accuracy9=evaluate(test_loader,model=model9)
        
        accuracy10=evaluate(test_loader,model=model10)
        accuracy11=evaluate(test_loader,model=model11)
        
        m1.append(accuracy1)
        m2.append(accuracy2)
        m3.append(accuracy3)
        
        m4.append(accuracy4)
        m5.append(accuracy5)
        
        m6.append(accuracy6)
        m7.append(accuracy7)
        
        m8.append(accuracy6)
        m9.append(accuracy9)
        
        m10.append(accuracy10)
        m11.append(accuracy11)
        
        
        
    print(f'**1mean: {np.mean(m1)} , var: {np.var(m1)}')
    print(f'*2mean: {np.mean(m2)} , var: {np.var(m2)}') 
    print(f'**3mean: {np.mean(m3)} , var: {np.var(m3)}')
     
    print(f'**4mean: {np.mean(m4)} , var: {np.var(m4)}')
    print(f'**5mean: {np.mean(m5)} , var: {np.var(m5)}')
    
    print(f'**6mean: {np.mean(m6)} , var: {np.var(m6)}')
    print(f'**7mean: {np.mean(m7)} , var: {np.var(m7)}')
    
    print(f'**8mean: {np.mean(m8)} , var: {np.var(m8)}')
    print(f'**9mean: {np.mean(m9)} , var: {np.var(m9)}')
    
    print(f'**10mean: {np.mean(m10)} , var: {np.var(m10)}')
    print(f'**11mean: {np.mean(m11)} , var: {np.var(m11)}')
    
                            
    write_csv(filename1,m1)
    write_csv(filename2,m2)
    write_csv(filename3,m3)
    
    write_csv(filename4,m4)
    write_csv(filename5,m5)
    
    write_csv(filename6,m6)
    write_csv(filename7,m7)

    write_csv(filename8,m8)
    write_csv(filename9,m9)        

    write_csv(filename10,m10)
    write_csv(filename11,m11)



 
    