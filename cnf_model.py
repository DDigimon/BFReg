import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperNetwork2(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize


    def forward(self, t):
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)
        
        W2 = params[self.blocksize:self.blocksize*2].reshape(self.width, self.in_out_dim, 1)

        U = params[2*self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

        B = params[4 * self.blocksize:].reshape(self.width, 1, 1)
        return [W,W2, B, U]

def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()

class CNF_G(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, width, edges):
        super().__init__()
        in_out_dim=1
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.tmp1=nn.Linear(1,4)
        self.tmp2=nn.Linear(4,1)
        self.hyper_net = HyperNetwork2(in_out_dim, hidden_dim, width)      
        self.ori_edge=edges
        
    def forward(self, t, states):
        z = states[0]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            W,W2, B, U = self.hyper_net(t)
            W=W.unsqueeze(-1)
            B=B.unsqueeze(-1)
            U=U.unsqueeze(-1)
            W2=W2.unsqueeze(-1)
            
            Z = torch.unsqueeze(z, 0)
            Z=Z.repeat(self.width, 1, 1)
            ori_z=Z
            
            
            my_edge=self.ori_edge
            ori_edge=my_edge.repeat(self.width,1,1)
            Z=Z.transpose(-1,-2)
            Z= torch.matmul(ori_edge,Z)
            Z=Z.transpose(-1,-2)
            Z=Z.unsqueeze(-1)
            ori_z=ori_z.unsqueeze(-1)
            
            h = torch.tanh(torch.matmul(Z, W) + B)
            
            dz_dt = torch.matmul(h, U)
            dz_dt = dz_dt.mean(0)
            dz_dt=dz_dt.squeeze(-1)
            
            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)

