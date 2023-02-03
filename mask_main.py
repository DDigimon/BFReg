
import pickle
import random
import numpy as np
import os
from copy import deepcopy
from torch.cuda.amp import autocast as autocast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from module import BFR_S,BFR, BFR2
import argparse
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str, default='A549', help='gene dataset')
    parser.add_argument('--data_path',type=str, default='./data/processed/', help='prior knowledge dataset')
    parser.add_argument('--model_path',type=str, default='./model/', help='model save path')
    
    parser.add_argument('--dropout',type=float, default=0, help='dropout')
    parser.add_argument('--model',type=str, default='BFR', help='model')
    
    parser.add_argument('--rate',type=str,default='1')
    parser.add_argument('--adj',type=str, default='gp', help='adj type')

    parser.add_argument('--alpha',type=str, default=0.0001, help='alpha1')
    parser.add_argument('--beta',type=str, default=0.0001, help='alpha2')

    parser.add_argument('--concat_mode',type=str, default=0.0001, help='concat mode')
    

    args = parser.parse_known_args()[0]
    return args
    

args=parse()
print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")

gene_name=args.dataset
batch_size=32
model_name=args.model
rates=args.rate
adj_type=args.adj
dropout=args.dropout
concat_mode=args.concat_mode
alpha=float(args.alpha)
beta=float(args.beta)

# prior_knowledge_set=[gene2gene_graphs,gene2protein_graphs,protein2protein_graph,pathway_dicts,my_gene_value]
data_path=args.data_path
prior_knowledge_path=data_path+gene_name+'_prior.pkl'

model_path=args.model_path+'/'+model_name+'/'

if os.path.exists(model_path)==False:
    os.makedirs(model_path)

save_path=model_path+gene_name+'.pkl'

with open(prior_knowledge_path,'rb') as f:
    prior_knowledge=pickle.load(f)

gene_data_path=data_path+gene_name+'_mask0.6.pkl'
with open(gene_data_path,'rb') as f:
    gene_data=pickle.load(f)

gene2gene_graphs=prior_knowledge[0]
gene2protein_graphs=prior_knowledge[1]
protein2protein_graphs=prior_knowledge[2]
pathway_dicts=prior_knowledge[3]
my_gene_value=prior_knowledge[4]

gene_num=int(gene_data['train'].shape[1]/2)
protein_num=len(protein2protein_graphs['nodes'])
print('gene number',gene_num)
print('protein num',protein_num)
print('label num',len(my_gene_value))

def generate_adj(graph,num,dense=True):
    g2g_edges_index=torch.from_numpy(np.array(graph['edges']).T).long()
    g2g_edge_value=torch.ones(g2g_edges_index.shape[1])
    g2g_shape=torch.Size([num,num])
    g2g_edge=torch.sparse.FloatTensor(g2g_edges_index,g2g_edge_value,g2g_shape)
    if dense:
        g2g_edge=g2g_edge.to_dense()
        return g2g_edge
    else:
        return g2g_edges_index

gene_all=np.arange(0,gene_num).tolist()
x_gene=[val for val in gene_all if val not in my_gene_value]
my_gene_value=list(set(my_gene_value))

print('train_size',np.array(gene_data['train']).shape)
print('valid_size',np.array(gene_data['valid']).shape)
print('test_size',np.array(gene_data['test']).shape)
test_size=gene_data['test'].shape[0]
train_size=gene_data['train'].shape[0]
valid_size=gene_data['valid'].shape[0]
train_data=DataLoader(np.array(gene_data['train']),batch_size=batch_size,shuffle=True)
valid_data=DataLoader(np.array(gene_data['valid']),batch_size=batch_size,shuffle=True)
test_data=DataLoader(np.array(gene_data['test']),batch_size=batch_size,shuffle=False)

    
gene_adj=generate_adj(gene2gene_graphs,gene_num,dense=False).to(device)
protein_adj=generate_adj(protein2protein_graphs,gene_num,dense=False).to(device)
print('gene',gene_adj.shape)
print('protein',protein_adj.shape)

class MLPBFR_S(nn.Module):
    def __init__(self, gene_num,protein_num,gene2gene_graphs,protein2protein_graphs,device):
        super().__init__()
        # print(gene2gene_graphs)
        self.mlp=nn.Linear(gene_num,gene_num)
        self.mlp1=nn.Linear(gene_num*4, 1024)
        self.mlp2=nn.Linear(1024, gene_num)

        self.dropout=nn.Dropout(dropout)
        self.graph=BFR_S(1,4,4,gene_num,gene2gene_graphs,device,dropout)
        
        self.emb=nn.Embedding(1,1)

    def forward(self,x,masks,device,alpha=alpha,beta=beta,use_edge=True):
        batchs=x.shape[0]
        values=x
        masks_inv=masks.bool()
        masks_inv=~masks_inv
        masks_inv=masks_inv.float()

        tmp_value_emb=torch.zeros_like(values).long().to(device)
        embed_x=self.emb(tmp_value_emb).squeeze()
        # print(masks.shape)
        values=torch.mul(values,masks)
        embed_x=torch.mul(embed_x,masks_inv)
        x=values+embed_x
        x=x.float()

        x=x.unsqueeze(-1)
        x=self.graph(x,alpha=alpha,beta=beta,use_edge=use_edge,mode=concat_mode)

        x=x.reshape(batchs,-1)
        x=self.mlp1(x)
        x=self.dropout(x)
        x=self.mlp2(F.elu(x))

        avaliable_genes=x.unsqueeze(-1)
        return avaliable_genes


class MLPBFR(nn.Module):
    def __init__(self, gene_num,protein_num,gene2gene_graphs,protein2protein_graphs,device):
        super().__init__()
        self.mlp=nn.Linear(gene_num,gene_num)
        self.mlp1=nn.Linear(4*gene_num, 1024)
        self.mlp2=nn.Linear(1024, gene_num)
        self.graph=BFR(1,4,4,gene_num,gene2gene_graphs,protein2protein_graphs,device,drop_rate=dropout)

        self.dropout=nn.Dropout(dropout)


        self.emb=nn.Embedding(1,1)

    def forward(self,x,masks,device):
        batchs=x.shape[0]
        values=x
        masks_inv=masks.bool()
        masks_inv=~masks_inv
        masks_inv=masks_inv.float()

        tmp_value_emb=torch.zeros_like(values).long().to(device)
        embed_x=self.emb(tmp_value_emb).squeeze()

        values=torch.mul(values,masks)
        embed_x=torch.mul(embed_x,masks_inv)
        x=values+embed_x
        x=x.float()
        

        x=x.unsqueeze(-1)
        x=self.graph(x,device)
        x=x.reshape(batchs,-1)
        x=self.mlp1(x)
        x=self.dropout(x)
        x=F.elu(x)
        x=self.mlp2(F.elu(x))

        avaliable_genes=x.unsqueeze(-1)
        return avaliable_genes

class MLPBFR2(nn.Module):
    def __init__(self, gene_num,protein_num,gene2gene_graphs,protein2protein_graphs,device):
        super().__init__()
        self.mlp=nn.Linear(gene_num,gene_num)
        self.mlp1=nn.Linear(4*gene_num, 1024)
        self.mlp2=nn.Linear(1024, gene_num)
        self.graph=BFR2(1,4,4,gene_num,gene2gene_graphs,protein2protein_graphs,device,drop_rate=dropout)

        self.dropout=nn.Dropout(dropout)
        self.emb=nn.Embedding(1,1)

    def forward(self,x,masks,device):
        batchs=x.shape[0]
        values=x
        masks_inv=masks.bool()
        masks_inv=~masks_inv
        masks_inv=masks_inv.float()

        tmp_value_emb=torch.zeros_like(values).long().to(device)
        embed_x=self.emb(tmp_value_emb).squeeze()

        values=torch.mul(values,masks)
        embed_x=torch.mul(embed_x,masks_inv)
        x=values+embed_x
        x=x.float()
        

        x=x.unsqueeze(-1)
        x=self.graph(x,device)
        x=x.reshape(batchs,-1)
        x=self.mlp1(x)
        x=self.dropout(x)
        x=F.elu(x)
        x=self.mlp2(F.elu(x))

        avaliable_genes=x.unsqueeze(-1)
        return avaliable_genes


v=[1 for _ in range(gene_adj.shape[1])]
v=torch.from_numpy(np.array(v)).to(device)
ori_gene=torch.sparse_coo_tensor(gene_adj,v,size=(gene_num,gene_num))

v=[1 for _ in range(protein_adj.shape[1])]
v=torch.from_numpy(np.array(v)).to(device)
ori_protein=torch.sparse_coo_tensor(protein_adj,v,size=(gene_num,gene_num))


if model_name=='BFR':
    model=MLPBFR(gene_num,protein_num,ori_gene,ori_protein,device).to(device)
elif 'BFR_S' in model_name:
    model=MLPBFR_S(gene_num,protein_num,ori_gene,ori_protein,device).to(device)

elif model_name=='BFR2':
    model=MLPBFR2(gene_num,protein_num,ori_gene,ori_protein,device).to(device)


loss_func=torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

global_valid=100

all_adj=torch.ones((gene_num,gene_num)).to_sparse().indices()
index = torch.LongTensor(random.sample(range(gene_num*gene_num), 10000))
sampling_adj=torch.index_select(all_adj, 1, index)
diag_edges=torch.ones(gene_num)
diag_edges=torch.diag_embed(diag_edges)
diag_edges=diag_edges.to_sparse().indices()
protein_adj=diag_edges.to(device)
global_test=0
counts=0

max_patient=30
for e in range(200):
    loss=0
    loss_v_all=0
    loss_size_all=0
    if 'BFR' in model_name:model.to(device)
    model.train()
    for idx,data in enumerate(train_data):
        data=data.to(device)
        loss_t=0
        optimizer.zero_grad() 
        marker=deepcopy(data)

        data_x=data[:,:gene_num]
        mask=data[:,gene_num:]
        masks_inv=mask.bool()
        masks_inv=~masks_inv
        avaliable_gene=model(data_x,mask,device)

        avaliable_gene_t=avaliable_gene.squeeze(-1)
        avaliable_gene_t=torch.mul(avaliable_gene_t,masks_inv).float()
        marker_t=torch.mul(marker[:,:gene_num],masks_inv).float()
        loss_v=loss_func(avaliable_gene_t,marker_t)
        loss_t+=loss_v

        loss_v_all+=loss_v

        tmp_loss=loss_t
        tmp_loss.backward()

        loss+=tmp_loss
        optimizer.step()

    loss/=len(train_data)
    loss_v_all/=len(train_data)
    loss_size_all/=len(train_data)
    
    if 'BFR' in model_name:model.to(cpu_device)

    model.eval()
    valid_loss=0
    for idx,data in enumerate(valid_data):
        loss_t=0
        if 'BFR' in model_name:data=data.to(cpu_device)
        else:data=data.to(device)
        marker=deepcopy(data)

        data_x=data[:,:gene_num]
        mask=data[:,gene_num:]
        masks_inv=mask.bool()
        masks_inv=~masks_inv

        avaliable_gene=model(data_x,mask,cpu_device)
        avaliable_gene_t=avaliable_gene.squeeze(-1)
        avaliable_gene_t=torch.mul(avaliable_gene_t,masks_inv).float()
        marker_t=torch.mul(marker[:,:gene_num],masks_inv).float()
        loss_t+=loss_func(avaliable_gene_t,marker_t)
        
        tmp_loss=loss_t 
        valid_loss+=tmp_loss
    valid_loss/=len(valid_data)
    
    
    if valid_loss<global_valid:
        global_valid=valid_loss
        test_loss=0
        model.eval()
        for idx,data in enumerate(test_data):
            loss_t=0
            if 'BFR' in model_name:data=data.to(cpu_device)
            else:data=data.to(device)
            marker=deepcopy(data)
            
            data_x=data[:,:gene_num]
            mask=data[:,gene_num:]
            masks_inv=mask.bool()
            masks_inv=~masks_inv
            avaliable_gene=model(data_x,mask,cpu_device)
            avaliable_gene_t=avaliable_gene.squeeze(-1)
            avaliable_gene_t=torch.mul(avaliable_gene_t,masks_inv).float()
            marker_t=torch.mul(marker[:,:gene_num],masks_inv).float()
            loss_t+=loss_func(avaliable_gene_t,marker_t)
            
            tmp_loss=loss_t 
            test_loss+=tmp_loss
        test_loss/=len(test_data)

        global_test=test_loss
        print('save!')
    else:
        counts+=1
    print(e,loss.item(),valid_loss.item(),test_loss.item())
    if counts==max_patient:break
with open(model_name+'_records.txt','a') as f:
    f.write(gene_name+'\t'+model_name+'\t'+adj_type+'\t'+rates+'\t'+str(dropout)+'\t'+str(alpha)+'\t'+str(beta)+'\t'+concat_mode+'\t'+str(global_test.item())+'\n')
