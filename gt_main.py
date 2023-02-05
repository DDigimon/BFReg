import argparse
import torch
from cnf_model import CNF_G
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
from copy import deepcopy
from geomloss import SamplesLoss

import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error

from dataset import Dream_BC

from sklearn import preprocessing
import scipy.sparse as sp


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_line',type=str, default='DU4475', help='gene dataset')
    parser.add_argument('--treatment',type=str, default='EGF', help='gene dataset')

    args = parser.parse_known_args()[0]
    return args

args=parse()

path='./data/bc/'


data_name=args.cell_line
treatment=args.treatment
model_name='CNF_G'
data_mode='future_pred'

print(args)

start_time=0
add_num=10
time_list=[0,7,9,13,17,40,60]

time_list_rev=deepcopy(time_list)
time_list_rev.reverse()
train_num=2
test_num=len(time_list)-train_num
train_time=time_list[:train_num]
train_time_rev=deepcopy(train_time)
train_time_rev.reverse()
test_time=time_list[train_num:]
test_time_rev=deepcopy(test_time)
test_time_rev.reverse()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data=Dream_BC(data_name,path,treatment,time_list)
graph_data=data.graph
graph_data=torch.from_numpy(np.array(graph_data).T).long().to(device)

batch_size=512
test_sample=batch_size
gene_num=37

v=[1 for _ in range(graph_data.shape[1])]
v=torch.from_numpy(np.array(v)).to(device)
ori_edges=torch.sparse_coo_tensor(graph_data,v,size=(gene_num,gene_num))
ori_edges=ori_edges.to_dense().detach().cpu().numpy()
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
inv=ori_edges
thres_edge=~inv
ori_edges=ori_edges+0.0005*thres_edge
ori_edges=normalize(ori_edges)
ori_edges=torch.from_numpy(ori_edges).float().to(device)


model=CNF_G(gene_num,32,64,ori_edges).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

global_wse=10
save_mse=0
save_wse=0
save_test_kl=10
patient=5
counts=0
validation_samples=10
aveg_wd=0 
loss_func = SamplesLoss(loss="sinkhorn")
global_test_time=[]
global_mse=0
for e in range(2000):
    model.train()
    optimizer.zero_grad()
    x=data.load_data(batch_size)
    x_ori=deepcopy(x)
    
    x=torch.from_numpy(x).float().to(device)
    
    logp_diff_t1 = torch.zeros(batch_size, 1).type(torch.float32).to(device)
    loss=0

    for j,idx in enumerate(train_time):
        i=time_list.index(idx)
        
        if i==0 :continue
        prev_i=time_list.index(train_time[j-1])
        if i==1:
            x_in=x[0,:,:]
        else:
            x_in=x[prev_i,:,:]
        x_out=x[i,:,:].squeeze()
        z_t, logp_diff_t = odeint(
            model,
            (x_out, logp_diff_t1),
            torch.tensor([time_list[i],time_list[prev_i]]).type(torch.float32).to(device),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5',
        )
        z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
        z_t0=torch.sigmoid(z_t0)
        x_in=torch.sigmoid(x_in)
        logp_x = loss_func(z_t0,x_in)
        loss+=logp_x
        
    loss/=len(train_time)
    loss.backward()
    optimizer.step()

    # valid
    model.eval()
    aveg_valid=0
    aveg_test=0
    aveg_train_wd=0
    aveg_mse=0
    for _ in range(validation_samples):
        x=data.load_data(test_sample)
        x=torch.from_numpy(x).float().to(device)
        x=x[0,:,:].squeeze()
        logp_diff_t0 = torch.zeros(test_sample, 1).type(torch.float32).to(device)
        logp_diff_t1 = torch.zeros(x.shape[0], 1).type(torch.float32).to(device)

        z_t_samples, _ = odeint(
            model,
            (x, logp_diff_t1),
            torch.tensor(time_list_rev).float().to(device),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5',
        )
        predicts_values=z_t_samples.detach().cpu().numpy()
        x_ori_values=x_ori
        
        z_t_samples=z_t_samples
        x_ori=torch.from_numpy(x_ori)
        
        x_ori=x_ori.detach().cpu().numpy()
        predicts=z_t_samples.detach().cpu().numpy()

        train_values=0
        train_gene_values=0
        train_time_values=0
        train_wd=0
        for idx in train_time:
            i=time_list.index(idx)
            alls=np.concatenate((x_ori[i,:,:],predicts[i,:,:]))
        
            scaler = preprocessing.MinMaxScaler().fit(alls)
            x_orii = scaler.transform(x_ori[i,:,:])        
            predictsi = scaler.transform(predicts[i,:,:])
            
            train_value=loss_func(torch.from_numpy(predictsi).double(),torch.from_numpy(x_orii).double())
            
            train_values+=train_value
            for g in range(gene_num):
                train_wd+=wasserstein_distance(predicts[i,:,g],x_ori[i,:,g])
        train_values/=train_num
        train_wd/=train_num*gene_num
        aveg_valid+=train_values.item()
        aveg_train_wd+=train_wd

        test_values=0
        test_mse_value=0
        wd=0
        for idx in test_time:
            i=time_list.index(idx)
            alls=np.concatenate((x_ori[i,:,:],predicts[i,:,:]))
        
            scaler = preprocessing.MinMaxScaler().fit(alls)
            x_orii = scaler.transform(x_ori[i,:,:])        
            predictsi = scaler.transform(predicts[i,:,:])
            
            test_value=loss_func(torch.from_numpy(predictsi).double(),torch.from_numpy(x_orii).double())
            test_values+=test_value 
            
            for g in range(gene_num):    
                wd+=wasserstein_distance(predicts[i,:,g],x_ori[i,:,g])
            
            predicts_means=predicts_values[i,:,:].mean(-1)
            ori_means=x_ori_values[i,:,:].mean(-1)
            test_mse=mean_squared_error(predicts_means,ori_means)
            test_mse_value+=test_mse
        
        aveg_mse+=test_mse_value/test_num        
        test_values/=test_num
        wd/=test_num*gene_num
        aveg_test+=test_values.item()
        aveg_wd+=wd
    aveg_mse/=validation_samples
    aveg_valid/=validation_samples
    aveg_train_wd/=validation_samples
    aveg_test/=validation_samples
    aveg_wd/=validation_samples

    print(e,'loss',loss.item(),'trains',aveg_valid,aveg_train_wd,'tests',aveg_test,aveg_wd,aveg_mse)
    if aveg_valid<global_wse:
        global_wse=aveg_valid
        global_wd=aveg_wd
        save_test_kl=aveg_test
        global_mse=aveg_mse
        counts=0
    else:
        counts+=1
        if counts==patient:break

with open('result.txt','a') as f:
    f.write(data_name+'\t'+model_name+'\t'+treatment+'\t'+data_mode+'\t'+str(save_test_kl)+'\t'+str(global_wd)+'\t'+str(global_mse)+'\n')
    
