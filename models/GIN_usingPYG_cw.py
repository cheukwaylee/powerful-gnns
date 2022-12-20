# https://codeantenna.com/a/z1hConS0ug

"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
# from dgl.nn.pytorch.conv import GINConv
# from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

# 自定义节点更新特征的方式，这里是mlp+bn+relu，实际是对应原文公式4.1第一项
class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):  # mlp:(FC+BN+relu)xN + FC + BN + relu
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


# MLP with lienar output: (FC+BN+relu) x N + FC, N = 0, 1, 2...
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
        '''
    
        super(MLP, self).__init__()
        self.linear_or_not = True  #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))  # first layer
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))  # intermediate layers
            self.linears.append(nn.Linear(hidden_dim, output_dim))  # last layer

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))  # the last layer do not need BN

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)  # only one layer: only FC
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))  # FC+BN+relu
            return self.linears[self.num_layers - 1](h)  # last layer: only FC


class GIN(nn.Module):
    """GIN model初始化"""
    def __init__(self, num_layers, num_mlp_layers, 
                 input_dim, hidden_dim, output_dim, 
                 final_dropout, learn_eps, 
                 graph_pooling_type, neighbor_pooling_type):
        """model parameters setting
        Paramters
        ---------
        num_layers: 
            The number of linear layers in the neural network
        num_mlp_layers: intMLP的层数
            The number of linear layers in mlps
        
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        
        final_dropout: float最后一层的抓爆率
            dropout ratio on the final linear layer
        learn_eps: boolean在学习epsilon参数时是否区分节点本身和邻居节点
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        
        neighbor_pooling_type: str邻居汇聚方式，原文公式4.1的后半部分
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str全图汇聚方式，和上面的邻居汇聚方式可以不一样
            how to aggregate entire nodes in a graph (sum, mean or max)
        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):#GIN有几层，除了最后一层每层都定义一个MLP（num_mlp_layers层）来进行COMBINE
            if layer == 0:#第一层GIN，注意输入维度，
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            #更新特征的方式是ApplyNodeFunc，邻居汇聚方式为neighbor_pooling_type
            #具体参考：https://docs.dgl.ai/api/python/nn.pytorch.html#ginconv
            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        
        #以下代码是将每一层点的表征保存下来，然后作为最后的图的表征计算
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        #图表征消息汇聚的方式
        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):#前向传播
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers - 1):#根据GIN层数做循环
            h = self.ginlayers[i](g, h)#做原文公式4.1的操作            
            h = self.batch_norms[i](h)#接BN
            h = F.relu(h)#接RELU
            hidden_rep.append(h)#保存每一层的输出，作为最后图表征的计算

        score_over_layer = 0

        #根据hidden_rep计算图表征
        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)  # h1, h2, h3 ...
            
            # h_i -> linear -> R1 (out_dim) -> drop???
            # score_over_layer = sum(...)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return score_over_layer
