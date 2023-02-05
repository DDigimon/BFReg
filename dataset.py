import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler


class Dream_BC():
    def __init__(self,data_name,path,treatment,time_list) -> None:
        self.gene_list=  [ 'b.CATENIN','cleavedCas', 'CyclinB', 'GAPDH', 'IdU', 'Ki.67', 'p.4EBP1',
       'p.Akt.Ser473.', 'p.AKT.Thr308.', 'p.AMPK', 'p.BTK', 'p.CREB', 'p.ERK',
       'p.FAK', 'p.GSK3b', 'p.H3', 'p.HER2', 'p.JNK', 'p.MAP2K3', 'p.MAPKAPK2',
       'p.MEK', 'p.MKK3.MKK6', 'p.MKK4', 'p.NFkB', 'p.p38', 'p.p53',
       'p.p90RSK', 'p.PDPK1', 'p.PLCg2', 'p.RB', 'p.S6', 'p.S6K', 'p.SMAD23',
       'p.SRC', 'p.STAT1', 'p.STAT3', 'p.STAT5']
        self.treatment_list=['EGF', 'full', 'iEGFR', 'iMEK', 'iPI3K', 'iPKC']
        self.data_name=data_name
        self.treatment=treatment
        self.time_list=time_list
        self.data_dicts={'EGF':[],'full':[],'iEGFR':[],'iMEK':[],'iPI3K':[],'iPKC':[]}
        self.m_data_dicts={'EGF':[],'full':[],'iEGFR':[],'iMEK':[],'iPI3K':[],'iPKC':[]}

        self.scaler=MinMaxScaler()

        self.load_file(data_name,path)
        self.graph=self.load_graph()

    def load_graph(self):
        graph_list=[]
        with open('./data/processed/bc/knowledge.txt') as f:
            for line in f.readlines():
                line=line.split('\n')[0].split('\t')
                graph_list.append([self.gene_list.index(line[0]),self.gene_list.index(line[2])])
        return graph_list

    def load_file(self,data_name,path):
        
        data=pd.read_csv(path+data_name+'.csv')

        for treat in self.treatment_list:
            new_data=data[data.treatment==treat]
            time_list=pd.unique(new_data['time'])
            data_list=[]
            for time in time_list:
                time_data=new_data[new_data.time==time]
                datas=time_data[self.gene_list]
                np_data=np.array(datas)
                data_list.append(np.nan_to_num(np_data))
               
            self.data_dicts[treat]=data_list
    
    def load_data(self,batch_size):
        data=[]
        select_data=self.data_dicts[self.treatment]
        for time in self.time_list:
            my_data=select_data[self.time_list.index(time)]
            choice_list=np.array([i for i in range(my_data.shape[0])])
            choices=np.random.choice(choice_list,batch_size)
            new_data=my_data[choices,:]
            data.append(new_data)
        return np.array(data)


        