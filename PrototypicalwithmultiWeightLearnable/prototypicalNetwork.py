from torchvision import transforms
from torch import nn, optim
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
import torch.nn.functional as F

class PrototypicalNetworks(nn.Module):
    def __init__(self,backbone:nn.Module,weightlearnable,weight):
        super(PrototypicalNetworks, self).__init__()
        self.backbone=backbone
        self.weightlearnable=weightlearnable
        self.weight=weight
        
        self.weight_1 = nn.Parameter(torch.Tensor([weight[0]]))
        self.weight_2 = nn.Parameter(torch.Tensor([weight[1]]))
        self.weight_3 = nn.Parameter(torch.Tensor([weight[2]]))
        self.weight_4 = nn.Parameter(torch.Tensor([weight[3]]))
        self.weight_5 = nn.Parameter(torch.Tensor([weight[4]]))

    def calculate(self,support_images:torch.Tensor,support_labels:torch.Tensor,query_images: torch.Tensor,num,numker):
        global_avg_pool = nn.AdaptiveAvgPool2d((1))
        train_nodes, eval_nodes = get_graph_node_names(self.backbone)
        return_nodes2 = {
            str(train_nodes[num]): "maxpool1"
        }
        f2 = create_feature_extractor(self.backbone, return_nodes=return_nodes2)
        out2 = f2(support_images)
        out2_pooled = global_avg_pool(out2['maxpool1'])
        z_support  = out2_pooled.view(out2_pooled.size(0), -1)
        
        # Calculate mean and standard deviation
        # mean = z_support.mean()
        # std = z_support.std()
        
        # Apply z-score normalization
        # normalized_z_support = (z_support - mean) / std

        out3 = f2(query_images)
        out3_pooled = global_avg_pool(out3['maxpool1'])
        z_query  = out3_pooled.view(out3_pooled.size(0), -1)
        
        # Calculate mean and standard deviation
        mean = z_query.mean()
        std = z_query.std()
        
        # Apply z-score normalization
        # normalized_z_query = (z_query - mean) / std
        
        n_way=len(torch.unique(support_labels))

        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )
        
        dists = torch.cdist(z_query, z_proto)
        # print(dists,"****")
        return dists
    def forward(self, support_images: torch.Tensor, support_labels: torch.Tensor, query_images: torch.Tensor):
        d1 = self.calculate(support_images, support_labels, query_images, 4, 64)
        d2 = self.calculate(support_images, support_labels, query_images, 19, 128)
        d3 = self.calculate(support_images, support_labels, query_images, 35, 256)
        d4 = self.calculate(support_images, support_labels, query_images, 51, 512)
        d5 = self.calculate(support_images, support_labels, query_images, 67, 512)

        if self.weightlearnable:
            scores4 = -(F.relu(self.weight_1) * d1 + F.relu(self.weight_2) * d2 + F.relu(self.weight_3) * d3 + F.relu(self.weight_4) * d4 + F.relu(self.weight_5) * d5)
        else:
            scores4 = -(self.weight[0] * d1 + self.weight[1] * d2 + self.weight[2] * d3 + self.weight[3] * d4 + self.weight[4] * d5)

        return scores4