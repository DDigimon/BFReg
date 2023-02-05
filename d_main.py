import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
import torch.optim as optim
from torch.autograd import Variable
import argparse
from sklearn.metrics import mean_squared_error
import scipy
import torch.nn.functional as F
import math
from module import GraphBlockD

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set',type=str, default='MX1,HCC1395,CAL148,HCC1569,HCC1599,BT20,HCC1187,HCC38,MPE600,ZR7530,CAL851,MDAMB415,MDAMB134VI,MFM223,MCF10F,CAL51,HCC1419,UACC893,BT474,HCC3153,HCC2157,HDQP1,MDAMB453,HCC70,HCC1937,T47D,EVSAT,MCF7,HBL100,MDAMB175VII,JIMT1,EFM192A', help='gene dataset')
    parser.add_argument('--test_set',type=str, default='HCC2185,HCC1954,184A1,MDAMB157,HCC1500,MCF10A,UACC812,MDAkb2,DU4475,MDAMB361,BT549,OCUBM', help='gene dataset')
    parser.add_argument('--model',type=str, default='MLP_G_NEK', help='prior knowledge dataset')
    parser.add_argument('--set_id',type=str,default='-1')
    parser.add_argument('--cell_line',type=str,default='other')

    args = parser.parse_known_args()[0]
    return args



args=parse()
print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cell_line_set=args.cell_line
gene_list=  [ 'b.CATENIN','cleavedCas', 'CyclinB', 'GAPDH', 'IdU', 'Ki.67', 'p.4EBP1',
       'p.Akt.Ser473.', 'p.AKT.Thr308.', 'p.AMPK', 'p.BTK', 'p.CREB', 'p.ERK',
       'p.FAK', 'p.GSK3b', 'p.H3', 'p.HER2', 'p.JNK', 'p.MAP2K3', 'p.MAPKAPK2',
       'p.MEK', 'p.MKK3.MKK6', 'p.MKK4', 'p.NFkB', 'p.p38', 'p.p53',
       'p.p90RSK', 'p.PDPK1', 'p.PLCg2', 'p.RB', 'p.S6', 'p.S6K', 'p.SMAD23',
       'p.SRC', 'p.STAT1', 'p.STAT3', 'p.STAT5']
gene_lens=len(gene_list)
graph_list=[]
with open('./data/processed/bc/knowledge.txt') as f:
    for line in f.readlines():
        line=line.split('\n')[0].split('\t')
        graph_list.append([gene_list.index(line[0]),gene_list.index(line[2])])

model_name=args.model



def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = list(map(classes_dict.get, labels))
    labels_onehot=np.array(labels_onehot,dtype=np.int32)
    
    return labels_onehot

off_diag = np.ones([gene_lens, gene_lens]) # - np.eye(gene_lens)

rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)
# shape (n*n,n)
rel_rec = Variable(rel_rec).to(device)
rel_send = Variable(rel_send).to(device)



treatment_list=['EGF', 'full', 'iEGFR', 'iMEK', 'iPI3K', 'iPKC']

train_list=args.train_set.split(',')
test_list=args.test_set.split(',')
if cell_line_set=='self':
    train_list=train_list+test_list
set_id=args.set_id

data=pd.read_csv('./data/cells/median_phospho_data.csv')
train_data_list=[]
test_data_list=[]
for train_data in train_list:
    tmp_data=deepcopy(data)
    train_data_list.append(tmp_data[tmp_data.cell_line==train_data])
for test_data in test_list:
    tmp_data=deepcopy(data)
    test_data_list.append(tmp_data[tmp_data.cell_line==test_data])

ori_train_list=[]
ori_test_list=[]
for treat in treatment_list:
    tmp_list=[]
    for train_data in train_data_list:
        tmp=train_data[train_data.treatment==treat]
        tmp_list.append(tmp)
    ori_train_list.append(tmp_list)
    tmp_list=[]
    for test_data in test_data_list:
        tmp=test_data[test_data.treatment==treat]
        tmp_list.append(tmp)
    ori_test_list.append(tmp_list)

# print(ori_train_list)

train_set_X=[]
train_set_Y=[]
for train_data in ori_train_list:
    
    for data in train_data:
        data=data.values
        X_set=[]
        Y_set=[]
        for treat_id in range(1,data.shape[0]):
            if data[treat_id,[1]]=='EGF':continue
            if treat_id==1:
                x_data=data[treat_id-1,3:].tolist()
                X_set.append(np.array(x_data))
            Y_set.append(data[treat_id,3:])
        if len(X_set)==0:continue
        X_set=np.array(X_set)
        Y_set=np.array(Y_set)
        train_set_X.append(X_set)
        train_set_Y.append(Y_set)

train_set_X=np.array(train_set_X,dtype=np.float16)
train_set_Y=np.array(train_set_Y,dtype=np.float16)
all_size=train_set_X.shape[0]
valid_set_X=train_set_X[int(0.7*all_size):,:,:]
valid_set_Y=train_set_Y[int(0.7*all_size):,:,:]

train_set_X=train_set_X[:int(0.7*all_size),:,:]
train_set_Y=train_set_Y[:int(0.7*all_size),:,:]

train_set_X=np.reshape(train_set_X,(train_set_X.shape[0],-1))
train_set_Y=np.reshape(train_set_Y,(train_set_Y.shape[0],-1))
train_set_X[np.isnan(train_set_X)]=0
train_set_Y[np.isnan(train_set_Y)]=0

valid_set_X=np.reshape(valid_set_X,(valid_set_X.shape[0],-1))
valid_set_Y=np.reshape(valid_set_Y,(valid_set_Y.shape[0],-1))
valid_set_X[np.isnan(valid_set_X)]=0
valid_set_Y[np.isnan(valid_set_Y)]=0


train_set=np.concatenate((train_set_X,train_set_Y),axis=-1)
valid_set=np.concatenate((valid_set_X,valid_set_Y),axis=-1)

print(train_set_X.shape,train_set_Y.shape)
if cell_line_set=="self":
    my_list=ori_train_list
elif cell_line_set=="other":
    my_list=ori_test_list

test_set_X=[]
test_set_Y=[]

for train_data in my_list:
    
    for data in train_data:
        data=data.values
        X_set=[]
        Y_set=[]
        for treat_id in range(1,data.shape[0]):
            if data[treat_id,[1]]!='EGF':continue
            x_data=data[treat_id-1,2:].tolist()
            if x_data[0]==5.5 or x_data[0]==23 or x_data[0]==30:continue
            if treat_id==1:
                X_set.append(np.array(x_data[1:]))
            Y_set.append(data[treat_id,3:])
        if len(X_set)==0:continue
        X_set=np.array(X_set)
        Y_set=np.array(Y_set)
        test_set_X.append(X_set)
        test_set_Y.append(Y_set)

test_set_X=np.array(test_set_X,dtype=np.float16)
test_set_Y=np.array(test_set_Y,dtype=np.float16)

test_set_X[np.isnan(test_set_X)]=0
test_set_Y[np.isnan(test_set_Y)]=0

test_set_X=np.reshape(test_set_X,(test_set_X.shape[0],-1))
test_set_Y=np.reshape(test_set_Y,(test_set_Y.shape[0],-1))
test_set=np.concatenate((test_set_X,test_set_Y),axis=-1)


train_loader=DataLoader(train_set,batch_size=4,shuffle=True)
valid_loader=DataLoader(valid_set,batch_size=4,shuffle=True)
test_loader=DataLoader(test_set,batch_size=test_set.shape[0],shuffle=False)




def training(graph_list,use_edge=False):
    # graph_set=deepcopy(graph_list)
    
    graph_list=np.array(graph_list).T
    print(graph_list.shape)
    graph_list=torch.from_numpy(graph_list).long().to(device)

    v=[1 for _ in range(graph_list.shape[1])]
    v=torch.from_numpy(np.array(v)).to(device)
    ori_edges=torch.sparse_coo_tensor(graph_list,v,size=(gene_lens,gene_lens))

    class MLP_G(nn.Module):
        def __init__(self):
            super().__init__()
            self.infer=nn.Linear(1,16)
            self.graph=GraphBlockD(1,16,16,37,ori_edges,device)
            self.mlp1=nn.Linear(16*37,512)
            self.mlp2=nn.Linear(512,222)
        def forward(self,x,alpha=0.005,use_edge=True):
            x=x.unsqueeze(-1)
            x=self.graph(x,alpha,use_edge)
            x=x.reshape(x.shape[0],-1)
            x=F.elu(x)
            x=self.mlp1(x)
            x=F.elu(x)
            x=self.mlp2(x)
            return x
        
    model=MLP_G().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    loss_func=nn.MSELoss()

    epoch=2000 # default 2000
    global_loss=10
    max_patient=30 # default 30
    counts=0
    for e in range(epoch):
        model.train()
        train_loss=0
        for data in train_loader:
            optimizer.zero_grad()
            data=data.to(device)
            x=data[:,:37].float()
            y=data[:,37:].float()
            out=model(x)
            score=loss_func(out,y)
            score.backward()
            optimizer.step()
            train_loss+=score
        train_loss/=len(train_loader)
        valid_loss=0
        model.eval()
        for data in valid_loader:
            data=data.to(device)
            x=data[:,:37].float()
            y=data[:,37:].float()
            out=model(x)
            score=loss_func(out,y)
            valid_loss+=score
        valid_loss/=len(valid_loader)
        print(e,train_loss.item(),valid_loss.item())
        if valid_loss<global_loss:
            global_loss=valid_loss
            for data in test_loader:
                data=data.to(device)
                x=data[:,:37].float()
                y=data[:,37:].float()
                out=model(x)
            out=out.detach().cpu().numpy()
            y=deepcopy(test_set_Y)
            t_mse_value=mean_squared_error(y,out)
            global_cor=0
            y=y.reshape(y.shape[0],gene_lens,-1)
            out=out.reshape(y.shape[0],gene_lens,-1)
            for i in range(gene_lens):
                global_cor+=scipy.stats.pearsonr(y[:,i,:].reshape(-1),out[:,i,:].reshape(-1))[0]
            global_cor/=gene_lens

            print(str(e),score.item(),global_cor)

            counts=0
        else:
            counts+=1
        if counts==max_patient:break
    
    return model,t_mse_value,global_cor

model,t_mse_value,global_cor=training(graph_list)

with open('records.txt','a') as f:
    f.write(model_name+'\t'+cell_line_set+'\t0\t'+str(set_id)+'\t'+str(t_mse_value)+'\t0\t'+str(global_cor)+'\t0\n')