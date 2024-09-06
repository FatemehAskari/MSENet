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
import os
import re

import csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--load', default='MiniImageNet/save/proto-5')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    
    # some hyparameters
    ImageSize=84
    root_directory = "../data/mini_imagenet/miniimagenet"
    # root_directory ="../data/FC100_dataset/fc100"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
        transforms.Resize([ImageSize, ImageSize])
    ])
    
    
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
    
    convolutional_network=resnet18(pretrained=True)
    convolutional_network.fc=nn.Flatten()
    
    allmodel=os.listdir(args.load)
    filename = f'MiniImageNet/5shotCSVfile_test/result/InfoTest_test=150.csv'
    column_names = ['lr','test_accuracy']
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        
        for name in allmodel:
            combined_directory = os.path.join(args.load, name)
            print(name)
            basename=os.path.basename(name)
            split_parts = basename.split('_lr=_')
            part2= split_parts[1].split('.pth')    
            print(float(part2[0]))
            
            model=PrototypicalNetworks(convolutional_network).cuda()
            model.load_state_dict(torch.load(combined_directory))
            model.eval()
            
            accuracy=evaluate(test_loader,model=model)
            value=[float(part2[0]),accuracy]
            writer.writerow(value)

    
    
    