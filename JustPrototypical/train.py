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
import pandas as pd
import csv
from easyfsl.datasets import CUB


# hierarchical clustering
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='CUB/save/proto-5')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(args.save_path)
    
    # some hyparameters
    ImageSize=112
    # root_directory = "../data/mini_imagenet/miniimagenet"
    # root_directory = "../data/FC100_dataset/fc100"
    root_directory = "../data/CUB"
    
    print(root_directory)
    N_TRAINING_EPISODES=100
    N_VALIDATION_TASKS=50

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
        transforms.Resize([ImageSize, ImageSize])
    ])
    
    batch_size = 128
    
    
    # Instantiate the MiniImageNet dataset for training    
    # train_set = MiniImageNet(root=root_directory, split="train", training=True, transform=transform)
    train_set = CUB(split="train", training=True)
    
    train_set.get_labels = lambda: [train_set[i][1] for i in range(len(train_set))]   
    train_sampler = TaskSampler(
    train_set, n_way=args.train_way, n_shot=args.shot, n_query=args.query, n_tasks=N_TRAINING_EPISODES
    )
    train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,)
    
    

    
    # # Instantiate the MiniImageNet dataset for validation
    # val_set = MiniImageNet(root=root_directory, split="val", training=False, transform=transform)
    val_set = CUB(split="val", training=False)  
      
    val_set.get_labels=lambda: [val_set[i][1] for i in range(len(val_set))]
    val_sampler=TaskSampler(
        val_set,n_way=args.test_way ,n_shot=args.shot,n_query=args.query,n_tasks=N_VALIDATION_TASKS
    ) 
    val_loader=DataLoader(
        val_set,
        batch_sampler=val_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=val_sampler.episodic_collate_fn,
    )
    
    # show support and query images in validation
    
    (
    example_support_images_val,
    example_support_labels_val,
    example_query_images_val,
    example_query_labels_val,
    example_class_ids_val,
    ) = next(iter(val_loader))
    
    # save_directory = "JustPrototypical/images"
    # plot_images(example_support_images_val, "support images", images_per_row=args.shot,name="support_image.png",save_directory=save_directory)
    # plot_images(example_query_images_val, "query images", images_per_row=args.query,name="query_image.png",save_directory=save_directory) 

    # define model
    convolutional_network=resnet18(pretrained=False)
    convolutional_network.fc=nn.Flatten()
    
    lr=[0.001,0.0002,0.0001,0.00002,0.00001]
    for i in lr:
        filename = f'CUB/5shotCSVfile_train/Allinfo_5Shot_lr={i}.csv'
        column_names = ['lr','loss_train','accuracy_validation']
        
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)
            
            convolutional_network=resnet18(pretrained=False)
            convolutional_network.fc=nn.Flatten()
            model_true=PrototypicalNetworks(convolutional_network).cuda()
            
            model_true.eval()
            example_scores = model_true(
                example_support_images_val.cuda(),
                example_support_labels_val.cuda(),
                example_query_images_val.cuda(),
            ).detach()
            
            _, example_predicted_labels = torch.max(example_scores.data, 1)
            
            result_before =evaluate(val_loader,model=model_true)
            
            
            # add optimizer and loss
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model_true.parameters(), lr=i)
            
            
            def fit(
                support_images: torch.Tensor,
                support_labels: torch.Tensor,
                query_images: torch.Tensor,
                query_labels: torch.Tensor,
                model
            ) -> float:
                optimizer.zero_grad()
                classification_scores = model(
                    support_images.cuda(), support_labels.cuda(), query_images.cuda()
                )
            
                loss = criterion(classification_scores, query_labels.cuda())
                loss.backward()
                optimizer.step()
            
                return loss.item()
        
        
        
            def calculate_accuracy(model, images, labels):
                model.eval()
                with torch.no_grad():
                    logits = model(images)
                    predicted_labels = torch.argmax(logits, dim=1).cuda().numpy()
                model.train()
                accuracy = accuracy_score(labels.cuda().numpy(), predicted_labels)
                return accuracy
            
            # Train the model yourself with this cell
            log_update_frequency = 10
            max1=0
            for k in range(5):
                correct_loss=[]                     
                all_loss = []
                model_high=model_true
                model_true.train()
                with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
                    for episode_index, (
                        support_images,
                        support_labels,
                        query_images,
                        query_labels,
                        _,
                    ) in tqdm_train:
                        loss_value = fit(support_images, support_labels, query_images, query_labels,model_true)
                        all_loss.append(loss_value)
                
                        if episode_index % log_update_frequency == 0:
                            tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))
                            
                result_after_true=evaluate(val_loader,model=model_high)
                
                if max1>result_after_true:
                    max1=result_after_true
                    correct_loss=all_loss
                    model_high=model_true              
            torch.save(model_high.state_dict(), osp.join(args.save_path, f'newprototypical_lr=_{i}' + '.pth'))
            for j in correct_loss:
                values = [i,j,result_after_true]
                writer.writerow(values)              
    
    