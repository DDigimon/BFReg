import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch_geometric.nn import HypergraphConv

class BFR_S(nn.Module):
    '''
    single layer
    '''
    def __init__(self,n_in,n_hid,n_out,gene_num,edges,device,drop_rate=0.):
        super().__init__()
        self.drop_rate = drop_rate
        self.ori_edge=edges.to_dense().float()
        inv=self.ori_edge.bool()
        self.thres_edge=~inv
        self.thres_edge=self.thres_edge.float()

        self.infer=nn.Linear(n_in,n_hid)
        self.infer2=nn.Linear(n_in,n_hid)
        
        self.mlp_edge=nn.Linear(n_hid*2,1)
        self.nodes_mlp=nn.Linear(n_hid*2,n_out)

        self.n_hid=n_hid

        self.mesg_merge=nn.Linear(n_hid*2,n_out)

        diag = np.ones([gene_num, gene_num]) 
        node_shape1 = np.array(self.encode_onehot(np.where(diag)[0]), dtype=np.float32)
        node_shape2 = np.array(self.encode_onehot(np.where(diag)[1]), dtype=np.float32)
        node_shape1 = torch.FloatTensor(node_shape1)
        node_shape2 = torch.FloatTensor(node_shape2)
        self.node_shape1 = Variable(node_shape1).to(device)
        self.node_shape2 = Variable(node_shape2).to(device)

    def encode_onehot(self,labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = list(map(classes_dict.get, labels))
        labels_onehot=np.array(labels_onehot,dtype=np.int32)
        
        return labels_onehot

    def forward(self,x,alpha=0.005,beta=0.00005,use_edge=True,mode='cat'):
        ## encode the edge
        batch,gene_num,_=x.shape

        node_shape1=self.node_shape1
        node_shape2=self.node_shape2

        ori_x=x

        x=F.elu(self.infer(x))
        x = F.dropout(x, self.drop_rate)

        x2=F.elu(self.infer2(ori_x))
        x2 = F.dropout(x2, self.drop_rate)

        shape1 = torch.matmul(node_shape1, x)
        shape1 = shape1.view(x.size(0) * shape1.size(1),x.size(2))

        
        shape2 = torch.matmul(node_shape2, x)
        shape2 = shape2.view(x.size(0) * shape2.size(1),x.size(2))

        edges = torch.cat([shape2, shape1], dim=1)

        edges = edges.view(batch, gene_num*gene_num, -1)

        edges=F.sigmoid(self.mlp_edge(edges))
        

        my_edge=self.ori_edge
        
        ori_edge=my_edge.repeat(batch,1,1)
        ori_edge=ori_edge+alpha*self.thres_edge
        ori_edge=ori_edge.view(batch,gene_num*gene_num,-1)
        edges=edges*ori_edge

        
        edges=edges.view(batch,gene_num*gene_num,-1)

        edges = F.dropout(edges, self.drop_rate)
     

        ### decode
        shape1 = torch.matmul(node_shape1, x)
        shape2 = torch.matmul(node_shape2, x)
        node_infor = torch.cat([shape2, shape1], dim=-1)
        edges=edges.view(batch,gene_num*gene_num,-1)

        if use_edge:
            node_infor=node_infor*edges

        recive_infor = node_infor.transpose(-2, -1).matmul(node_shape1).transpose(-2, -1)
        recive_infor = recive_infor.contiguous()
        recive_infor=self.nodes_mlp(recive_infor)
        recive_infor=F.elu(recive_infor)
        recive_infor = F.dropout(recive_infor, self.drop_rate)

        if mode=='cat':
            update_infor=torch.cat([recive_infor,x],dim=-1)
            update_infor=self.mesg_merge(update_infor)
        elif mode=='sum':
            update_infor=recive_infor+x

        update_infor=F.dropout(update_infor,self.drop_rate)

        return update_infor
        
    def get_edges(self,x):
        
        batch,gene_num,_=x.shape
        node_shape1=self.node_shape1
        node_shape2=self.node_shape2

        x=F.elu(self.infer(x))
        x = F.dropout(x, self.drop_rate)

        shape1 = torch.matmul(node_shape1, x)
        shape1 = shape1.view(x.size(0) * shape1.size(1), x.size(2))

        
        shape2 = torch.matmul(node_shape2, x)
        shape2 = shape2.view(x.size(0) * shape2.size(1),x.size(2))

        edges = torch.cat([shape2, shape1], dim=1)

        edges = edges.view(batch, gene_num*gene_num, -1)

        edges=F.relu(self.mlp_edge(edges))
        return edges


class BFR(nn.Module):
    '''
    directly map the genes to proteins
    '''
    def __init__(self,n_in,n_hid,n_out,gene_num,edges1,edges2,device,drop_rate=0.):
        super().__init__()
        self.drop_rate = drop_rate

        self.ori_edge1=edges1.to_dense().float()
        inv=self.ori_edge1.bool()
        self.thres_edge1=~inv
        self.thres_edge1=self.thres_edge1.float()

        self.ori_edge2=edges2.to_dense().float()
        inv=self.ori_edge2.bool()
        self.thres_edge2=~inv
        self.thres_edge2=self.thres_edge2.float()


        self.infer=nn.Linear(n_in,n_hid)
        self.infer2=nn.Linear(n_in,n_hid)
        
        self.mlp_edge1=nn.Linear(n_hid*2,1)
        self.mlp_edge2=nn.Linear(n_hid*2,1)

        self.mesg_merge1=nn.Linear(n_hid*2,n_out)
        self.mesg_merge2=nn.Linear(n_hid*2,n_out)

        self.nodes_mlp1=nn.Linear(n_hid*2,n_out)
        self.nodes_mlp2=nn.Linear(n_hid*2,n_out)

        self.n_hid=n_hid

        diag = np.ones([gene_num, gene_num])

        node_shape1 = np.array(self.encode_onehot(np.where(diag)[0]), dtype=np.float32)
        node_shape2 = np.array(self.encode_onehot(np.where(diag)[1]), dtype=np.float32)
        self.node_shape1 = torch.FloatTensor(node_shape1)
        self.node_shape2 = torch.FloatTensor(node_shape2)

        self.batch_norm=nn.BatchNorm1d(gene_num)
    def encode_onehot(self,labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = list(map(classes_dict.get, labels))
        labels_onehot=np.array(labels_onehot,dtype=np.int32)
        return labels_onehot


    def forward(self,x,device,alpha=0.005,beta=0.00005,use_edge=True,mode='cat'):
        batch,gene_num,_=x.shape

        node_shape1 = Variable(self.node_shape1).to(device)
        node_shape2 = Variable(self.node_shape2).to(device)
        self.ori_edge1=self.ori_edge1.to(device)
        self.ori_edge2=self.ori_edge2.to(device)
        self.thres_edge1=self.thres_edge1.to(device)
        self.thres_edge2=self.thres_edge2.to(device)
        x=F.elu(self.infer(x))
        x = F.dropout(x, self.drop_rate)

        shape1 = torch.matmul(node_shape1, x)
        shape1 = shape1.view(x.size(0) * shape1.size(1), x.size(2))

        
        shape2 = torch.matmul(node_shape2, x)
        shape2 = shape2.view(x.size(0) * shape2.size(1), x.size(2))

        edges = torch.cat([shape2, shape1], dim=1)

        edges = edges.view(batch, gene_num*gene_num, -1)

        edges1=F.sigmoid(self.mlp_edge1(edges))

        my_edge1=self.ori_edge1
        
        ori_edge1=my_edge1.repeat(batch,1,1)
        ori_edge1=ori_edge1+alpha*self.thres_edge1
        ori_edge1=ori_edge1.view(batch,gene_num*gene_num,-1)
        edges1=edges1*ori_edge1        
        edges1=edges1.view(batch,gene_num*gene_num,-1)

        ### decode
        shape1 = torch.matmul(node_shape1, x)
        shape2 = torch.matmul(node_shape2, x)
        node_infor = torch.cat([shape2, shape1], dim=-1)
        edges1=edges1.view(batch,gene_num*gene_num,-1)

        if use_edge:
            node_infor1=node_infor*edges1

        recive_infor1 = node_infor1.transpose(-2, -1).matmul(node_shape1).transpose(-2, -1)
        recive_infor1 = recive_infor1.contiguous()
        recive_infor1 = F.dropout(recive_infor1, self.drop_rate)
        recive_infor1=self.nodes_mlp1(recive_infor1)
        recive_infor1=F.elu(recive_infor1)
        recive_infor1 = F.dropout(recive_infor1, self.drop_rate)

        if mode=='cat':
            update_infor=torch.cat([recive_infor1,x],dim=-1)
            update_infor=F.elu(self.mesg_merge1(update_infor))
        elif mode=='sum':
            update_infor=recive_infor1+x
        x=update_infor
        x=self.batch_norm(x)

        shape1 = torch.matmul(node_shape1, x)
        shape1 = shape1.view(x.size(0) * shape1.size(1), x.size(2))
        
        shape2 = torch.matmul(node_shape2, x)
        shape2 = shape2.view(x.size(0) * shape2.size(1), x.size(2))

        edges = torch.cat([shape2, shape1], dim=1)

        edges = edges.view(batch, gene_num*gene_num, -1)

        edges2=F.sigmoid(self.mlp_edge2(edges))

        my_edge2=self.ori_edge2
        ori_edge2=my_edge2.repeat(batch,1,1)
        ori_edge2=ori_edge2+beta*self.thres_edge2
        ori_edge2=ori_edge2.view(batch,gene_num*gene_num,-1)
        edges2=edges2*ori_edge2
        
        edges2=edges2.view(batch,gene_num*gene_num,-1)

        ### decode
        shape1 = torch.matmul(node_shape1, x)
        shape2 = torch.matmul(node_shape2, x)
        node_infor = torch.cat([shape2, shape1], dim=-1)
        edges1=edges1.view(batch,gene_num*gene_num,-1)
        edges2=edges2.view(batch,gene_num*gene_num,-1)

        if use_edge:
            node_infor2=node_infor*edges2

        recive_infor2 = node_infor2.transpose(-2, -1).matmul(node_shape1).transpose(-2, -1)
        recive_infor2 = recive_infor2.contiguous()
        recive_infor2 = F.dropout(recive_infor2, self.drop_rate)
        recive_infor2=self.nodes_mlp2(recive_infor2)
        recive_infor2=F.elu(recive_infor2)
        recive_infor2 = F.dropout(recive_infor2, self.drop_rate)

        if mode=='cat':
            update_infor=torch.cat([recive_infor2,x],dim=-1)
            update_infor=F.elu(self.mesg_merge2(update_infor))
        elif mode=='sum':
            update_infor=recive_infor2+x
        update_infor=F.dropout(update_infor,self.drop_rate)

        return update_infor
        
    def get_edges(self,x):
        
        batch,gene_num,_=x.shape
        node_shape1=self.node_shape1
        node_shape2=self.node_shape2

        x=F.elu(self.infer(x))
        x = F.dropout(x, self.drop_rate)

        node1 = torch.matmul(node_shape1, x)
        node1 = node1.view(x.size(0) * node1.size(1),x.size(2))

        
        node2 = torch.matmul(node_shape2, x)
        node2 = node2.view(x.size(0) * node2.size(1), x.size(2))

        edges = torch.cat([node2, node1], dim=1)

        edges = edges.view(batch, gene_num*gene_num, -1)
        # print(edges.shape)

        edges=F.relu(self.mlp_edge(edges))
        return edges


class BFR2(nn.Module):
    def __init__(self,n_in,n_hid,n_out,gene_num,edges1,edges2,device,drop_rate=0.):
        super().__init__()

        self.W1=nn.Parameter(torch.randn(gene_num,gene_num,requires_grad=True)).to(device)
        nn.init.kaiming_uniform_(self.W1,a=math.sqrt(5))
        self.W1.requires_grad_()
        self.b1=nn.Parameter(torch.randn(gene_num,requires_grad=True)).to(device)
        nn.init.uniform_(self.b1)
        self.b1.requires_grad_()
        self.adj=torch.eye(gene_num).to(device)

        self.drop_rate = drop_rate
        self.ori_edge1=edges1.to_dense().float()
        inv=self.ori_edge1.bool()
        self.thres_edge1=~inv
        self.thres_edge1=self.thres_edge1.float()

        self.ori_edge2=edges2.to_dense().float()
        inv=self.ori_edge2.bool()
        self.thres_edge2=~inv
        self.thres_edge2=self.thres_edge2.float()


        self.infer=nn.Linear(n_in,n_hid)
        self.infer2=nn.Linear(n_in,n_hid)
        
        self.mlp_edge1=nn.Linear(n_hid*2,1)
        self.mlp_edge2=nn.Linear(n_hid*2,1)

        self.mesg_merge1=nn.Linear(n_hid*2,n_out)
        self.mesg_merge2=nn.Linear(n_hid*2,n_out)

        self.nodes_mlp1=nn.Linear(n_hid*2,n_out)
        self.nodes_mlp2=nn.Linear(n_hid*2,n_out)

        self.n_hid=n_hid


        diag = np.ones([gene_num, gene_num])

        node_shape1 = np.array(self.encode_onehot(np.where(diag)[0]), dtype=np.float32)
        node_shape2 = np.array(self.encode_onehot(np.where(diag)[1]), dtype=np.float32)
        self.node_shape1 = torch.FloatTensor(node_shape1)
        self.node_shape2 = torch.FloatTensor(node_shape2)
        

        self.batch_norm=nn.BatchNorm1d(gene_num)
    def encode_onehot(self,labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = list(map(classes_dict.get, labels))
        labels_onehot=np.array(labels_onehot,dtype=np.int32)
        return labels_onehot


    def forward(self,x,device,alpha=0.005,beta=0.00005,use_edge=True,mode='cat'):
        batch,gene_num,_=x.shape

        node_shape1 = Variable(self.node_shape1).to(device)
        node_shape2 = Variable(self.node_shape2).to(device)
        self.ori_edge1=self.ori_edge1.to(device)
        self.ori_edge2=self.ori_edge2.to(device)
        self.thres_edge1=self.thres_edge1.to(device)
        self.thres_edge2=self.thres_edge2.to(device)

        x=F.elu(self.infer(x))
        x = F.dropout(x, self.drop_rate)


        shape1 = torch.matmul(node_shape1, x)
        shape1 = shape1.view(x.size(0) * shape1.size(1), x.size(2))

        shape2 = torch.matmul(node_shape2, x)
        shape2 = shape2.view(x.size(0) * shape2.size(1), x.size(2))

        edges = torch.cat([shape2, shape1], dim=1)

        edges = edges.view(batch, gene_num*gene_num, -1)

        edges1=F.sigmoid(self.mlp_edge1(edges))

        my_edge1=self.ori_edge1

        ori_edge1=my_edge1.repeat(batch,1,1)
        ori_edge1=ori_edge1+alpha*self.thres_edge1
        ori_edge1=ori_edge1.view(batch,gene_num*gene_num,-1)
        
        edges1=edges1*ori_edge1
        
        edges1=edges1.view(batch,gene_num*gene_num,-1)

        ### decode
        shape1 = torch.matmul(node_shape1, x)
        shape2 = torch.matmul(node_shape2, x)
        node_infor = torch.cat([shape2, shape1], dim=-1)
        edges1=edges1.view(batch,gene_num*gene_num,-1)

        if use_edge:
            node_infor1=node_infor*edges1

        recive_infor1 = node_infor1.transpose(-2, -1).matmul(node_shape1).transpose(-2, -1)
        recive_infor1 = recive_infor1.contiguous()
        recive_infor1 = F.dropout(recive_infor1, self.drop_rate)
        recive_infor1=self.nodes_mlp1(recive_infor1)
        recive_infor1=F.elu(recive_infor1)
        recive_infor1 = F.dropout(recive_infor1, self.drop_rate)

        if mode=='cat':
            update_infor=torch.cat([recive_infor1,x],dim=-1)
            update_infor=F.elu(self.mesg_merge1(update_infor))
        elif mode=='sum':
            update_infor=recive_infor1+x
        w1=self.adj*self.W1
        update_infor=update_infor.transpose(1,2)
        x=F.elu(F.linear(update_infor,w1,self.b1))
        x=x.transpose(1,2)
        x=self.batch_norm(x)

        shape1 = torch.matmul(node_shape1, x)
        shape1 = shape1.view(x.size(0) * shape1.size(1), x.size(2))
        
        shape2 = torch.matmul(node_shape2, x)
        shape2 = shape2.view(x.size(0) * shape2.size(1), x.size(2))

        edges = torch.cat([shape2, shape1], dim=1)

        edges = edges.view(batch, gene_num*gene_num, -1)

        edges2=F.sigmoid(self.mlp_edge2(edges))

        my_edge2=self.ori_edge2
        ori_edge2=my_edge2.repeat(batch,1,1)
        ori_edge2=ori_edge2+beta*self.thres_edge2
        ori_edge2=ori_edge2.view(batch,gene_num*gene_num,-1)
        edges2=edges2*ori_edge2
        
        edges2=edges2.view(batch,gene_num*gene_num,-1)

        ### decode
        shape1 = torch.matmul(node_shape1, x)
        shape2 = torch.matmul(node_shape2, x)
        node_infor = torch.cat([shape2, shape1], dim=-1)
        edges1=edges1.view(batch,gene_num*gene_num,-1)
        edges2=edges2.view(batch,gene_num*gene_num,-1)

        if use_edge:
            node_infor2=node_infor*edges2

        recive_infor2 = node_infor2.transpose(-2, -1).matmul(node_shape1).transpose(-2, -1)
        recive_infor2 = recive_infor2.contiguous()
        recive_infor2 = F.dropout(recive_infor2, self.drop_rate)
        recive_infor2=self.nodes_mlp2(recive_infor2)
        recive_infor2=F.elu(recive_infor2)
        recive_infor2 = F.dropout(recive_infor2, self.drop_rate)

        if mode=='cat':
            update_infor=torch.cat([recive_infor2,x],dim=-1)
            update_infor=F.elu(self.mesg_merge2(update_infor))
        elif mode=='sum':
            update_infor=recive_infor2+x
        update_infor=F.dropout(update_infor,self.drop_rate)

        return update_infor
        
    def get_edges(self,x):
        
        batch,gene_num,_=x.shape
        node_shape1=self.node_shape1
        node_shape2=self.node_shape2

        x=F.elu(self.infer(x))
        x = F.dropout(x, self.drop_rate)

        node1 = torch.matmul(node_shape1, x)
        node1 = node1.view(x.size(0) * node1.size(1),x.size(2))

        
        node2 = torch.matmul(node_shape2, x)
        node2 = node2.view(x.size(0) * node2.size(1), x.size(2))

        edges = torch.cat([node2, node1], dim=1)

        edges = edges.view(batch, gene_num*gene_num, -1)

        edges=F.relu(self.mlp_edge(edges))
        return edges

class BFR3(nn.Module):
    def __init__(self,n_in,n_hid,n_out,gene_num,edges1,edges2,edges3,device,drop_rate=0.):
        super().__init__()
        '''
        gene + protein + pathway
        '''
        self.W1=nn.Parameter(torch.randn(gene_num,gene_num,requires_grad=True)).to(device)
        nn.init.kaiming_uniform_(self.W1,a=math.sqrt(5))
        self.W1.requires_grad_()
        self.b1=nn.Parameter(torch.randn(gene_num,requires_grad=True)).to(device)
        nn.init.uniform_(self.b1)
        self.b1.requires_grad_()

        self.W2=nn.Parameter(torch.randn(edges3.shape[0],gene_num,requires_grad=True)).to(device)
        nn.init.kaiming_uniform_(self.W2,a=math.sqrt(5))
        self.W2.requires_grad_()
        self.b2=nn.Parameter(torch.randn(edges3.shape[0],requires_grad=True)).to(device)
        nn.init.uniform_(self.b2)
        self.b2.requires_grad_()

        self.W3=nn.Parameter(torch.randn(edges3.shape[0],gene_num,requires_grad=True)).to(device)
        nn.init.kaiming_uniform_(self.W3,a=math.sqrt(5))
        self.W3.requires_grad_()
        self.b3=nn.Parameter(torch.randn(gene_num,requires_grad=True)).to(device)
        nn.init.uniform_(self.b3)
        self.b3.requires_grad_()


        self.inner_w=nn.Parameter(torch.randn(edges3.shape[0],edges3.shape[0],requires_grad=True)).to(device)
        nn.init.kaiming_uniform_(self.inner_w,a=math.sqrt(5))
        self.inner_w.requires_grad_()

        self.adj=torch.eye(gene_num).to(device)
        self.gene_num=gene_num

        self.drop_rate = drop_rate
        self.ori_edge1=edges1.to_dense().float()
        inv=self.ori_edge1.bool()
        self.thres_edge1=~inv
        self.thres_edge1=self.thres_edge1.float()

        self.ori_edge2=edges2.to_dense().float()
        inv=self.ori_edge2.bool()
        self.thres_edge2=~inv
        self.thres_edge2=self.thres_edge2.float()


        self.ori_edge3=edges3
        inv=edges3.bool()
        self.thres_edge3=~inv
        self.thres_edge3=self.thres_edge3.float()

        self.infer=nn.Linear(n_in,n_hid)
        self.infer2=nn.Linear(n_in,n_hid)
        
        self.mlp_edge1=nn.Linear(n_hid*2,1)
        self.mlp_edge2=nn.Linear(n_hid*2,1)

        self.mesg_merge1=nn.Linear(n_hid*2,n_out)
        self.mesg_merge2=nn.Linear(n_hid*2,n_out)
        self.mesg_merge3=nn.Linear(n_hid*2,n_hid)

        self.nodes_mlp1=nn.Linear(n_hid*2,n_out)
        self.nodes_mlp2=nn.Linear(n_hid*2,n_out)

        self.hyper_graph1=HypergraphConv(4,4)
        self.hyper_graph2=HypergraphConv(4,4)

        self.n_hid=n_hid


        diag = np.ones([gene_num, gene_num])

        node_shape1 = np.array(self.encode_onehot(np.where(diag)[0]), dtype=np.float32)
        node_shape2 = np.array(self.encode_onehot(np.where(diag)[1]), dtype=np.float32)
        self.node_shape1 = torch.FloatTensor(node_shape1)
        self.node_shape2 = torch.FloatTensor(node_shape2)
        

        self.batch_norm=nn.BatchNorm1d(gene_num)

        self.batch_norm2=nn.BatchNorm1d(gene_num)

        self.batch_norm3=nn.BatchNorm1d(gene_num)

    def encode_onehot(self,labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = list(map(classes_dict.get, labels))
        labels_onehot=np.array(labels_onehot,dtype=np.int32)
        return labels_onehot

    def rewrite_hyper_adj(self,batchs,adj):
        edge_set=[]
        for i in range(batchs):
            new_adj=torch.clone(adj)
            new_adj[0,:]=new_adj[0,:]+i*self.gene_num
            edge_set.append(new_adj)
        edge_set=torch.cat([e for e in edge_set],dim=-1)
        return edge_set

    def forward(self,x,device,alpha=0.005,beta=0.00005,use_edge=True,mode='cat'):
        batch,gene_num,_=x.shape

        node_shape1 = Variable(self.node_shape1).to(device)
        node_shape2 = Variable(self.node_shape2).to(device)
        self.ori_edge1=self.ori_edge1.to(device)
        self.ori_edge2=self.ori_edge2.to(device)
        self.thres_edge1=self.thres_edge1.to(device)
        self.thres_edge2=self.thres_edge2.to(device)


        x=F.elu(self.infer(x))
        x = F.dropout(x, self.drop_rate)

        shape1 = torch.matmul(node_shape1, x)
        shape1 = shape1.view(x.size(0) * shape1.size(1), x.size(2))

        shape2 = torch.matmul(node_shape2, x)
        shape2 = shape2.view(x.size(0) * shape2.size(1), x.size(2))

        edges = torch.cat([shape2, shape1], dim=1)

        edges = edges.view(batch, gene_num*gene_num, -1)

        edges1=F.sigmoid(self.mlp_edge1(edges))

        my_edge1=self.ori_edge1

        ori_edge1=my_edge1.repeat(batch,1,1)
        ori_edge1=ori_edge1+alpha*self.thres_edge1
        ori_edge1=ori_edge1.view(batch,gene_num*gene_num,-1)
        
        edges1=edges1*ori_edge1
        
        edges1=edges1.view(batch,gene_num*gene_num,-1)


        ### decode
        shape1 = torch.matmul(node_shape1, x)
        shape2 = torch.matmul(node_shape2, x)
        node_infor = torch.cat([shape2, shape1], dim=-1)
        edges1=edges1.view(batch,gene_num*gene_num,-1)

        if use_edge:
            node_infor1=node_infor*edges1

        recive_infor1 = node_infor1.transpose(-2, -1).matmul(node_shape1).transpose(-2, -1)
        recive_infor1 = recive_infor1.contiguous()
        recive_infor1 = F.dropout(recive_infor1, self.drop_rate)
        recive_infor1=self.nodes_mlp1(recive_infor1)
        recive_infor1=F.elu(recive_infor1)
        recive_infor1 = F.dropout(recive_infor1, self.drop_rate)

        if mode=='cat':
            update_infor=torch.cat([recive_infor1,x],dim=-1)
            update_infor=F.elu(self.mesg_merge1(update_infor))
        elif mode=='sum':
            update_infor=recive_infor1+x
        w1=self.adj*self.W1
        
        update_infor=update_infor.transpose(1,2)
        x=F.elu(F.linear(update_infor,w1,self.b1))
        x=x.transpose(1,2)
        x=self.batch_norm(x)
        

        shape1 = torch.matmul(node_shape1, x)
        shape1 = shape1.view(x.size(0) * shape1.size(1), x.size(2))
        
        shape2 = torch.matmul(node_shape2, x)
        shape2 = shape2.view(x.size(0) * shape2.size(1), x.size(2))

        edges = torch.cat([shape2, shape1], dim=1)

        edges = edges.view(batch, gene_num*gene_num, -1)

        edges2=F.sigmoid(self.mlp_edge2(edges))

        my_edge2=self.ori_edge2
        ori_edge2=my_edge2.repeat(batch,1,1)
        ori_edge2=ori_edge2+beta*self.thres_edge2
        ori_edge2=ori_edge2.view(batch,gene_num*gene_num,-1)
        edges2=edges2*ori_edge2
        
        edges2=edges2.view(batch,gene_num*gene_num,-1)

        ### decode
        shape1 = torch.matmul(node_shape1, x)
        shape2 = torch.matmul(node_shape2, x)
        node_infor = torch.cat([shape2, shape1], dim=-1)
        edges1=edges1.view(batch,gene_num*gene_num,-1)
        edges2=edges2.view(batch,gene_num*gene_num,-1)

        if use_edge:
            node_infor2=node_infor*edges2

        recive_infor2 = node_infor2.transpose(-2, -1).matmul(node_shape1).transpose(-2, -1)
        recive_infor2 = recive_infor2.contiguous()
        recive_infor2 = F.dropout(recive_infor2, self.drop_rate)
        recive_infor2=self.nodes_mlp2(recive_infor2)
        recive_infor2=F.elu(recive_infor2)
        recive_infor2 = F.dropout(recive_infor2, self.drop_rate)

        if mode=='cat':
            update_infor2=torch.cat([recive_infor2,x],dim=-1)
            update_infor2=F.elu(self.mesg_merge2(update_infor2))
        elif mode=='sum':
            update_infor2=recive_infor2+x
        update_infor2=F.dropout(update_infor2,self.drop_rate)

        update_infor2=self.batch_norm2(update_infor2)

        my_edge3=self.ori_edge3.float() # +self.thres_edge3
        hyper_adj=self.rewrite_hyper_adj(batch,my_edge3)
        
        x=update_infor2.reshape((batch*self.gene_num),-1)
        hyper_adj=hyper_adj.long()
        hyper_x=self.hyper_graph1(x,hyper_adj)
        hyper_x=F.elu(hyper_x)
        hyper_x=self.hyper_graph2(x,hyper_adj)
        hyper_x=F.elu(hyper_x)
        hyper_x=hyper_x.reshape((batch,self.gene_num,-1))
        all_infor=torch.cat([update_infor2,hyper_x],dim=-1)
        all_infor=F.elu(self.mesg_merge3(all_infor))
        all_infor=F.dropout(all_infor,self.drop_rate)

        return all_infor
        
    def get_edges(self,x):
        
        batch,gene_num,_=x.shape
        node_shape1=self.node_shape1
        node_shape2=self.node_shape2

        x=F.elu(self.infer(x))
        x = F.dropout(x, self.drop_rate)

        node1 = torch.matmul(node_shape1, x)
        node1 = node1.view(x.size(0) * node1.size(1),x.size(2))

        
        node2 = torch.matmul(node_shape2, x)
        node2 = node2.view(x.size(0) * node2.size(1), x.size(2))

        edges = torch.cat([node2, node1], dim=1)

        edges = edges.view(batch, gene_num*gene_num, -1)

        edges=F.relu(self.mlp_edge(edges))
        return edges

class GraphBlockD(nn.Module):
    def __init__(self,n_in,n_hid,n_out,gene_num,edges,device,drop_rate=0.):
        super().__init__()

        self.dropout_prob = drop_rate
        self.ori_edge=edges.to_dense().float()
        inv=self.ori_edge.bool()
        self.thres_edge=~inv
        self.thres_edge=self.thres_edge.float()

        self.infer=nn.Linear(n_in,n_hid)
        
        self.mlp_edge=nn.Linear(n_hid*2,1)
        self.nodes_mlp=nn.Linear(n_hid*2,n_out)

        self.n_hid=n_hid
        off_diag = np.ones([gene_num, gene_num])
        rel_rec = np.array(self.encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(self.encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        rel_rec = torch.FloatTensor(rel_rec)
        rel_send = torch.FloatTensor(rel_send)
        self.rel_rec = Variable(rel_rec).to(device)
        self.rel_send = Variable(rel_send).to(device)

    def encode_onehot(self,labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = list(map(classes_dict.get, labels))
        labels_onehot=np.array(labels_onehot,dtype=np.int32)
        
        return labels_onehot



    def forward(self,x,alpha=0.005,use_edge=True):
        batch,gene_num,_=x.shape

        rel_rec=self.rel_rec
        rel_send=self.rel_send

        x=F.elu(self.infer(x))
        x = F.dropout(x, self.dropout_prob)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(x.size(0) * receivers.size(1),
                                   x.size(2))

        
        senders = torch.matmul(rel_send, x)
        senders = senders.view(x.size(0) * senders.size(1),
                               x.size(2))

        edges = torch.cat([senders, receivers], dim=1)

        edges = edges.view(batch, gene_num*gene_num, -1)

        edges=F.relu(self.mlp_edge(edges))
        

        my_edge=self.ori_edge
        
        ori_edge=my_edge.repeat(batch,1,1)
        ori_edge=ori_edge+alpha*self.thres_edge
        ori_edge=ori_edge.view(batch,gene_num*gene_num,-1)
        edges=edges*ori_edge

        
        edges=edges.view(batch,gene_num*gene_num,-1)

        edges = F.dropout(edges, self.dropout_prob)
     
        ### decode
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        pre_msg = torch.cat([senders, receivers], dim=-1)
        edges=edges.view(batch,gene_num*gene_num,-1)

        if use_edge:
            pre_msg=pre_msg*edges

        agg_msgs = pre_msg.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()
        
        agg_msgs=self.nodes_mlp(agg_msgs)
        agg_msgs = F.dropout(agg_msgs, self.dropout_prob)
        agg_msgs=F.elu(agg_msgs)

        agg_msgs=agg_msgs+x

        return agg_msgs
        