from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from module import BFR_S, BFR2, BFR3,BFR


class MLP_BFR_S(nn.Module):
    def __init__(self,in_dim,n_hid,out_dim,graphs,device,drop_rate=0.):
        super().__init__()
        self.graph=BFR_S(1,4,4,in_dim,graphs,device)
        self.hiddens=nn.Linear(4*in_dim,n_hid)
        self.outs=nn.Linear(n_hid,out_dim)
    def forward(self,x,device,alpha=0.0001,beta=0.0001,mode='cat'):
        x=x.unsqueeze(-1)
        x=self.graph(x,alpha=alpha,beta=beta,mode=mode)
        x=x.reshape(x.shape[0],-1)
        x=self.hiddens(x)
        x=F.elu(x)
        x=self.outs(x)
        return x

class MLP_BFR(nn.Module):
    def __init__(self,in_dim,n_hid,out_dim,graphs1,graphs2,device,drop_rate=0.):
        super().__init__()
        self.graph=BFR(1,4,4,in_dim,graphs1,graphs2,device)
        self.hiddens=nn.Linear(4*in_dim,n_hid)
        self.outs=nn.Linear(n_hid,out_dim)
    def forward(self,x,device,alpha=0.0001,beta=0.0001,mode='cat'):
        x=x.unsqueeze(-1)
        x=self.graph(x,device,alpha=alpha,beta=beta,mode=mode)
        x=x.reshape(x.shape[0],-1)
        x=F.elu(x)
        x=self.hiddens(x)
        x=F.elu(x)
        x=self.outs(x)
        return x

class MLP_BFR2(nn.Module):
    def __init__(self,in_dim,n_hid,out_dim,graphs1,graphs2,device,drop_rate=0.):
        super().__init__()
        self.graph=BFR2(1,4,4,in_dim,graphs1,graphs2,device)
        self.hiddens=nn.Linear(4*in_dim,n_hid)
        self.outs=nn.Linear(n_hid,out_dim)
    def forward(self,x,device,alpha=0.0001,beta=0.0001,mode='cat'):
        x=x.unsqueeze(-1)
        x=self.graph(x,device,alpha=alpha,beta=beta,mode=mode)
        x=x.reshape(x.shape[0],-1)
        x=F.elu(x)
        x=self.hiddens(x)
        x=F.elu(x)
        x=self.outs(x)
        return x

class MLP_BFR3(nn.Module):
    def __init__(self,in_dim,n_hid,out_dim,graphs1,graphs2,graphs3,device,drop_rate=0.):
        super().__init__()
        self.graph=BFR3(1,4,4,in_dim,graphs1,graphs2,graphs3,device)
        self.graph_out=graphs3.shape[0]
        self.hiddens=nn.Linear(4*self.graph_out,n_hid)

        self.outs=nn.Linear(n_hid,out_dim)
    def forward(self,x,device,alpha=0.0001,beta=0.0001,mode='cat'):
        x=x.unsqueeze(-1)
        x=self.graph(x,device,alpha=alpha,beta=beta,mode=mode)
        x=x.reshape(x.shape[0],-1)
        x=F.elu(x)
        x=self.hiddens(x)
        x=F.elu(x)
        x=self.outs(x)
        return x