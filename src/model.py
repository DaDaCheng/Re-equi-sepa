import torch
from torch import nn
class MLP(nn.Module):
    def __init__(self,width_list,p=0):
        super(MLP, self).__init__()
        self.depth=len(width_list)-1
        self.fc_list=torch.nn.ModuleList([])
        self.bn_list=torch.nn.ModuleList([])
        self.p=p
        for i in range(self.depth):
            self.fc_list.append(nn.Linear(width_list[i], width_list[i+1]))
            self.bn_list.append(nn.BatchNorm1d(width_list[i]))
        if self.p>0:
            self.do_list=torch.nn.ModuleList([])
            for i in range(self.depth-1):
                self.do_list.append(nn.Dropout(p=p))
    def forward(self, x):
        out_list=[]
        out_list.append(x)
        for i in range(self.depth-1):
            x=self.fc_list[i](self.bn_list[i](x))
            x=x.relu()
            out_list.append(x)
            if self.p>0:
                x=self.do_list[i](x)
        x=self.fc_list[-1](self.bn_list[-1](x))
        out_list.append(x)
        return out_list