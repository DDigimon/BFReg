    

import pickle
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from c_model import MLP_BFR_S, MLP_BFR2,MLP_BFR,MLP_BFR3
from sklearn.metrics import roc_auc_score
import argparse
'''

adj choose
N
gene
protein

'''
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str, default='muscle', help='gene dataset')
    parser.add_argument('--data_path',type=str, default='./data/processed/c/', help='prior knowledge dataset')
    parser.add_argument('--model_path',type=str, default='./model/', help='model save path')
    
    parser.add_argument('--dropout',type=float, default=0, help='dropout')
    parser.add_argument('--model',type=str, default='BFR', help='model name')
    
    parser.add_argument('--rate',type=str,default='1')
    parser.add_argument('--adj',type=str, default='gene', help='adj type')

    parser.add_argument('--alpha',type=str, default=0.0001, help='alpha1')
    parser.add_argument('--beta',type=str, default=0.0001, help='alpha2')

    parser.add_argument('--concat_mode',type=str, default='cat', help='concat mode')
    

    args = parser.parse_known_args()[0]
    return args
    

args=parse()
print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_name=args.dataset
batch_size=32
model_name=args.model
rates=args.rate
adj_type=args.adj
dropout=args.dropout
concat_mode=args.concat_mode
alpha=float(args.alpha)
beta=float(args.beta)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_func=nn.CrossEntropyLoss()


data_path=args.data_path
prior_knowledge_path=data_path+dataset_name+'/'+dataset_name+'.pkl'
selected_subgraph_path=data_path+dataset_name+'_sub.pkl'
# gene_data_path=data_path+gene_name+'.pkl'

model_path=args.model_path+'/'+model_name+'/'
if os.path.exists(model_path)==False:
    os.makedirs(model_path)

save_path=model_path+dataset_name+'.pkl'

with open(prior_knowledge_path,'rb') as f:
    prior_knowledge=pickle.load(f)


class my_dataset(Dataset):
    def __init__(self,train_data,train_label):
        super(my_dataset).__init__()
        self.x=train_data
        self.y=train_label
    def __getitem__(self, index):
        test_x=self.x[index,:]
        test_label=self.y[index,:]
        return [test_x,test_label]
    def __len__(self):
        return self.x.shape[0]

gene2gene_graphs=prior_knowledge[0]
gene2protein_graphs=prior_knowledge[1]
protein2protein_graphs=prior_knowledge[2]

g2g_graph=prior_knowledge[0]
g2pgraph=prior_knowledge[1]
p2p_graph=prior_knowledge[2]

train_data_path=data_path+dataset_name+'/ctrain.npy'
valid_data_path=data_path+dataset_name+'/cvalid.npy'
test_data_path=data_path+dataset_name+'/ctest.npy'
train_label_path=data_path+dataset_name+'/ctrainlabel.npy'
valid_label_path=data_path+dataset_name+'/cvalidlabel.npy'
test_label_path=data_path+dataset_name+'/ctestlabel.npy'

def to_onehot(data,class_num):
    new_data=np.zeros((data.shape[0],class_num))
    for idx,d in enumerate(data):
        new_data[idx,d]=1
    return new_data

train_data=np.load(train_data_path)
valid_data=np.load(valid_data_path)
test_data=np.load(test_data_path)
train_label=np.load(train_label_path)
valid_label=np.load(valid_label_path)
test_label=np.load(test_label_path)
class_num=np.unique(train_label).shape[0]
gene_num=train_data.shape[1]

train_label=to_onehot(train_label,class_num)
valid_label=to_onehot(valid_label,class_num)
test_label=to_onehot(test_label,class_num)

print(train_data.shape,train_label.shape)
train_data=my_dataset(train_data,train_label)
valid_data=my_dataset(valid_data,valid_label)
test_data=my_dataset(test_data,test_label)

train_data=DataLoader(train_data,batch_size=batch_size,shuffle=True)
valid_data=DataLoader(valid_data,batch_size=batch_size,shuffle=False)
test_data=DataLoader(test_data,batch_size=batch_size,shuffle=False)

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

gene_adj=generate_adj(gene2gene_graphs,gene_num,dense=False).to(device)
protein_adj=generate_adj(protein2protein_graphs,gene_num,dense=False).to(device)

if model_name=='BFR_S':
    v=[1 for _ in range(gene_adj.shape[1])]
    v=torch.from_numpy(np.array(v)).to(device)
    ori_gene=torch.sparse_coo_tensor(gene_adj,v,size=(gene_num,gene_num))

    v=[1 for _ in range(protein_adj.shape[1])]
    v=torch.from_numpy(np.array(v)).to(device)
    ori_protein=torch.sparse_coo_tensor(protein_adj,v,size=(gene_num,gene_num))
    if adj_type=='gene':
        model=MLP_BFR_S(gene_num,256,class_num,ori_gene,device,dropout).to(device)
    elif adj_type=='protein':
        model=MLP_BFR_S(gene_num,256,class_num,ori_protein,device,dropout).to(device)

elif model_name=='BFR':
    v=[1 for _ in range(gene_adj.shape[1])]
    v=torch.from_numpy(np.array(v)).to(device)
    ori_gene=torch.sparse_coo_tensor(gene_adj,v,size=(gene_num,gene_num))

    v=[1 for _ in range(protein_adj.shape[1])]
    v=torch.from_numpy(np.array(v)).to(device)
    ori_protein=torch.sparse_coo_tensor(protein_adj,v,size=(gene_num,gene_num))

    model=MLP_BFR(gene_num,256,class_num,ori_gene,ori_protein,device,dropout).to(device)

elif model_name=='BFR2':
    v=[1 for _ in range(gene_adj.shape[1])]
    v=torch.from_numpy(np.array(v)).to(device)
    ori_gene=torch.sparse_coo_tensor(gene_adj,v,size=(gene_num,gene_num))

    v=[1 for _ in range(protein_adj.shape[1])]
    v=torch.from_numpy(np.array(v)).to(device)
    ori_protein=torch.sparse_coo_tensor(protein_adj,v,size=(gene_num,gene_num))

    model=MLP_BFR2(gene_num,256,class_num,ori_gene,ori_protein,device,dropout).to(device)

elif model_name=='BFR3':
    v=[1 for _ in range(gene_adj.shape[1])]
    v=torch.from_numpy(np.array(v)).to(device)
    ori_gene=torch.sparse_coo_tensor(gene_adj,v,size=(gene_num,gene_num))

    v=[1 for _ in range(protein_adj.shape[1])]
    v=torch.from_numpy(np.array(v)).to(device)
    ori_protein=torch.sparse_coo_tensor(protein_adj,v,size=(gene_num,gene_num))

    pathways=np.load(data_path+dataset_name+"/pathway.npy")
    def trans(a):
        new=[]
        for i in range(a.shape[0]):
            for j in range(a[i,:].shape[0]):
                if a[i,j]==1:
                    new.append([i,j])
        return np.array(new)
    pathways=trans(pathways)
    pathways=torch.from_numpy(pathways).to(device)

    model=MLP_BFR3(gene_num,256,class_num,ori_gene,ori_protein,pathways,device,dropout).to(device)

optimizer = optim.Adam(model.parameters(), lr = 5e-4)

epoch=200
global_valid=10
patient=5
counts=0
for e in range(epoch):
    train_loss=0
    for data in train_data:
        optimizer.zero_grad()
        label=data[1].float().to(device)
        data=data[0].float().to(device)
        out=model(data,device,alpha=alpha,beta=beta)
        loss=loss_func(out,label)
        loss.backward()
        optimizer.step()
        train_loss+=loss
    train_loss/=len(train_data)


    valid_loss=0
    for data in valid_data:
        label=data[1].float().to(device)
        data=data[0].float().to(device)
        out=model(data,device,alpha=alpha,beta=beta)
        loss=loss_func(out,label)
        valid_loss+=loss
    valid_loss/=len(valid_data)

    print(e,train_loss.item(),valid_loss.item())
    if valid_loss<global_valid:
        global_valid=valid_loss
        test_loss=0
        counts=0
        standrad=[]
        out_list=[]
        for data in test_data:
            label=data[1].float().to(device)
            data=data[0].float().to(device)
            out=model(data,alpha=alpha,beta=beta)
            loss=loss_func(out,label)
            standrad.append(label)
            out_list.append(out)
            test_loss+=loss
        test_loss/=len(test_data)
        standrad=torch.cat([s for s in standrad],dim=0)
        out_list=torch.cat([s for s in out_list],dim=0)
        standrad=standrad.detach().cpu()
        out_list=out_list.detach().cpu()
        auc=roc_auc_score(standrad,out_list,average='macro')
        print('Save!')
        print(auc)
    else:
        counts+=1
    if counts==patient:
        break

with open(dataset_name+'cresults.txt','a') as f:
    f.write(model_name+'\t'+adj_type+'\t'+str(alpha)+'\t'+str(beta)+'\t'+concat_mode+'\t'+str(0.)+'\t'+str(auc)+'\n')