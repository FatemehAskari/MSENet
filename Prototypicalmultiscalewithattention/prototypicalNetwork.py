from torchvision import transforms
from torch import nn, optim
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
import torch.nn.functional as F
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,gamma):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = gamma

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class ConvLayer(nn.Module):
    def __init__(self, n_in, n_out, ks, ndim, norm_type, act_cls, bias):
        super(ConvLayer, self).__init__()
        # Your initialization code here

    def forward(self, x):
        # Your forward pass implementation here
        return x  # Placeholder implementation

class SelfAttention(nn.Module):
    def __init__(self, n_channels,gamma):
        super(SelfAttention, self).__init__()
        self.query, self.key, self.value = [self._conv(n_channels, c) for c in (n_channels // 8, n_channels // 8, n_channels)]
        self.gamma = gamma
        
    def _conv(self, n_in, n_out):
        return ConvLayer(n_in, n_out, ks=1, ndim=1, norm_type=None, act_cls=None, bias=False)
        # return nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, bias=False)
    def forward(self, x):
        # Notation from the paper.
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = torch.softmax(torch.bmm(f.transpose(1, 2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        # print(self.gamma.item())
        return o.view(*size).contiguous()

    
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
        self.gamma1  =   nn.Parameter(torch.Tensor([0.0]))
        self.gamma2  =   nn.Parameter(torch.Tensor([0.0]))

    def calculate(self,support_images:torch.Tensor,support_labels:torch.Tensor,query_images: torch.Tensor,num,numker):

                
        global_avg_pool = nn.AdaptiveAvgPool2d((1))
        train_nodes, eval_nodes = get_graph_node_names(self.backbone)
        return_nodes2 = {
            str(train_nodes[num]): "maxpool1"
        }
        f2 = create_feature_extractor(self.backbone, return_nodes=return_nodes2)
        out2 = f2(support_images)
        
        input_size = out2['maxpool1']  # Batch size, number of channels, sequence length
        n_channels = input_size.size()[1]
        self_attention = Self_Attn(in_dim=n_channels,activation='relu',gamma=self.gamma1).to('cuda')
        output,_ = self_attention(input_size)
        # print(output[0][0])
        
        out2_pooled = global_avg_pool(output)              
        z_support  = out2_pooled.view(out2_pooled.size(0), -1)

        out3 = f2(query_images)
        input_size = out3['maxpool1']  # Batch size, number of channels, sequence length
        n_channels = input_size.size()[1]
        self_attention = Self_Attn(in_dim=n_channels,activation='relu',gamma=self.gamma2).to('cuda')
        output,_ = self_attention(input_size)
        # print(output[0][0])
        
        out3_pooled = global_avg_pool(output)
        z_query  = out3_pooled.view(out3_pooled.size(0), -1)
        
        n_way=len(torch.unique(support_labels))

        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )
        dists = torch.cdist(z_query, z_proto)
        return dists
    def forward(self,support_images:torch.Tensor,support_labels:torch.Tensor,query_images: torch.Tensor):
        d1=self.calculate(support_images,support_labels,query_images,4,64)
        d2=self.calculate(support_images,support_labels,query_images,19,128)
        d3=self.calculate(support_images,support_labels,query_images,35,256)
        d4=self.calculate(support_images,support_labels,query_images,51,512)
        d5=self.calculate(support_images,support_labels,query_images,67,512)
        if self.weightlearnable:
            scores4 = -(self.weight_1 * d1 + self.weight_2 * d2 + self.weight_3 * d3 + self.weight_4 * d4+self.weight_5 * d5)
        else:
           scores4 = -(self.weight[0] * d1 + self.weight[1] * d2 + self.weight[2] * d3 + self.weight[3] * d4+self.weight[4] * d5)
        return scores4