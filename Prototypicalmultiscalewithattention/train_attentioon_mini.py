from prototypicalNetwork_attention import PrototypicalNetworks
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

# hierarchical clustering
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='learngamma_MiniImageNet/save/1testshot')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(args.save_path)
    
    # some hyparameters
    ImageSize=84
    root_directory = "../data/mini_imagenet/miniimagenet"
    # root_directory = "../data/FC100_dataset/fc100"
    
    N_TRAINING_EPISODES=360
    N_VALIDATION_TASKS=100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
        transforms.Resize([ImageSize, ImageSize])
    ])
    
    
    # Instantiate the MiniImageNet dataset for training
    train_set = MiniImageNet(root=root_directory, split="train", training=True, transform=transform)
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
    
    

    
    # Instantiate the MiniImageNet dataset for validation
    val_set = MiniImageNet(root=root_directory, split="val", training=False, transform=transform)  
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

    plot_images(example_support_images_val, "support images", images_per_row=args.shot,  name="support_val")
    plot_images(example_query_images_val, "query images", images_per_row=args.query, name= "query_val") 

    # define model
    convolutional_network=resnet18(pretrained=True)
    convolutional_network.fc=nn.Flatten()
    df = pd.read_excel('InitializeWeight.xlsx')
    # weights = df.values[:,1:6].tolist()
    # weights=[[1,1 ,1 ,1 ,1],
    #      [1 ,1.1 ,1.2 ,1.3 ,1.4],
    #     ]
    weights=[
         [1 ,1 ,1 ,1 ,1],
        ]
    
    gamma_list = [i/10 for i in range(6)]
    for j in range(len(gamma_list)):
        save_path = os.path.join(args.save_path, str(gamma_list[j]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        filename = f'learngamma_MiniImageNet/test/1shot${gamma_list[j]}.csv'
        column_names = ['weight1', 'weight2', 'weight3', 'weight4', 'weight5','weight_change1', 
                        'weight_change2', 'weight_change3', 'weight_change4', 'weight_change5','gamma1','gamma2',
                        'changegamma1','changegamma2','lr1','accuracy_true' , 'gammawithoutlearn','accuracywithoutlearn']
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)
            convolutional_network=resnet18(pretrained=True)
            convolutional_network.fc=nn.Flatten()
            for i in range (len(weights)):
                convolutional_network=resnet18(pretrained=True)
                convolutional_network.fc=nn.Flatten()
                model_true=PrototypicalNetworks(convolutional_network,weightlearnable=True,gammalearnable=False,weight=weights[i],gamma1=gamma_list[j],
                                                gamma2=gamma_list[j]).cuda()
                

                
                
                # add optimizer and loss
                criterion = nn.CrossEntropyLoss()
                regular_params = [param for param in model_true.parameters() 
                                  if not any(param is 
                                             weight_param for weight_param 
                                             in [model_true.weight_1, model_true.weight_2
                                                 , model_true.weight_3, model_true.weight_4
                                                 , model_true.weight_5,
                                                 model_true.gamma1_l,model_true.gamma2_l])]

                optimizer = optim.Adam([
                    {'params': regular_params},  
                    {'params': [model_true.weight_1, model_true.weight_2, 
                                model_true.weight_3, model_true.weight_4, 
                                model_true.weight_5], 'lr': 0.005},
                    {'params':[model_true.gamma1_l,
                                model_true.gamma2_l],'lr': 0.005}
                ], lr=0.0001)
                
                # optimizer = optim.Adam(model_true.parameters(), lr=0.0001)                
                
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
                
                all_loss = []
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
                            
                result_after_true=evaluate(val_loader,model=model_true)
                
                weight_after1_true=model_true.weight_1.item()
                weight_after2_true=model_true.weight_2.item()
                weight_after3_true=model_true.weight_3.item()
                weight_after4_true=model_true.weight_4.item()
                weight_after5_true=model_true.weight_5.item()
                
                torch.save(model_true.state_dict(), osp.join(save_path, f'model_true_${gamma_list[j]}_${i}' + '.pth'))
                print(model_true.gamma1_l.item(),model_true.gamma2_l.item())
                
                
                # without learn

                convolutional_network=resnet18(pretrained=True)
                convolutional_network.fc=nn.Flatten()
                model_false=PrototypicalNetworks(convolutional_network,weightlearnable=True,gammalearnable=False,weight=weights[i],gamma1=gamma_list[j],
                                                gamma2=gamma_list[j]).cuda()
                

                
                
                # add optimizer and loss
                criterion = nn.CrossEntropyLoss()
                regular_params = [param for param in model_false.parameters() 
                                  if not any(param is 
                                             weight_param for weight_param 
                                             in [model_false.weight_1, model_false.weight_2
                                                 , model_false.weight_3, model_false.weight_4
                                                 , model_false.weight_5,
                                                 model_false.gamma1_l,model_false.gamma2_l])]

                optimizer = optim.Adam([
                    {'params': regular_params},  
                    {'params': [model_false.weight_1, model_false.weight_2, 
                                model_false.weight_3, model_false.weight_4, 
                                model_false.weight_5], 'lr': 0.005},
                    {'params':[model_false.gamma1_l,
                                model_false.gamma2_l],'lr': 0.005}
                ], lr=0.0001)
                
                
                model_false.train()                
                                                
                result_after_false=evaluate(val_loader,model=model_false)
                torch.save(model_false.state_dict(), osp.join(save_path, f'model_false_${gamma_list[j]}_${i}' + '.pth'))                                                                                

                values = [weights[i][0],weights[i][1],weights[i][2],
                        weights[i][3],weights[i][4], weight_after1_true,
                        weight_after2_true,weight_after3_true,weight_after4_true,
                        weight_after5_true,gamma_list[j],gamma_list[j],model_true.gamma1_l.item(),model_true.gamma2_l.item(),0.0001,
                        result_after_true,gamma_list[j],result_after_false]
                writer.writerow(values)
                
            
            
            
            
            
        
    
    
    
    
    
    
    
    