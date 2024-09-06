from prototypicalNetwork import PrototypicalNetworks
import argparse
import os.path as osp
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric,evaluate_on_one_task,evaluate,evaluate_test
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
    parser.add_argument('--query', type=int, default=10)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--load', default='save/proto-5')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    
    # some hyparameters
    ImageSize=84
    root_directory = "../data/mini_imagenet/miniimagenet"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
        transforms.Resize([ImageSize, ImageSize])
    ])
    
    
    test_set = MiniImageNet(root=root_directory, split="test", training=False, transform=transform)
    test_set.get_labels = lambda: [test_set[i][1] for i in range(len(test_set))]   
    test_sampler = TaskSampler(
    test_set, n_way=args.test_way, n_shot=args.shot, n_query=args.query, n_tasks=2000
    )
    test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,)
    
    convolutional_network=resnet18(pretrained=True)
    convolutional_network.fc=nn.Flatten()
    
    model=PrototypicalNetworks(convolutional_network).cuda()
    model.load_state_dict(torch.load('save/proto-5/newprototypical_lr=_2e-05.pth'))
    model.eval()
    
    accuracy=evaluate_test(test_loader,model=model)
    print(accuracy)

        
    # model=PrototypicalNetworks(convolutional_network).cuda()
    # model.load_state_dict(torch.load(args.load))
    # model.eval()
    
    # evaluate(test_loader,model=model)
    
    
    