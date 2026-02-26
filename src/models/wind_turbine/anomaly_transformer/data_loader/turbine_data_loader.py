import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os

class TurbineDataLoader(Dataset):
    def __init__(self, data_path, win_size, step, model, mode="train", scaler=None):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler() if scaler is None else scaler
        
        # Load data
        data_csv = os.path.join(data_path,str(model)+'.csv')
        data = pd.read_csv(f"{data_csv}", sep=",")
        
        # Store timestamp
        self.date_time = data[['time_stamp']].values
        
        data.drop(columns=['time_stamp','asset_id','id','train_test'], inplace=True)
        self.features = list(data.columns)
        
        data = data.values
        
        data = np.nan_to_num(data)
        
        train_size, val_size = 0,0
        self.anomaly_intervals = []
        self.background_data = None
        
        if mode == 'test':
        
            # generator bearing
            if model[-1] == '0':
                self.anomaly_intervals = [
                    (288,2299),
                    (2982,7489),
                ]
                
            # gearbox
            if model[-1] == '1':
                self.anomaly_intervals = [
                    (0,1268),
                    (1557,2853),
                ]
                
            # transformer
            if model[-1] == '2':
                self.anomaly_intervals = [
                    (0, 2013)
                ]
                
            # hydraulic
            if model[-1] == '3':
                self.anomaly_intervals = [
                    (575, 1439),
                    (1728, 2732),
                ]
            
        # Split train data
        else:
            
            # generator bearing    
            if model[-1] == '0':
                train_size = 46800
                val_size = 5328
            
            # gearbox
            if model[-1] == '1':
                train_size = 88723
                val_size = int(len(data)-train_size)
                
            # transformer
            if model[-1] == '2':
                train_size = 48960
                val_size = 3024
                
            # hydraulic
            if model[-1] == '3':
                 train_size = 45648
                 val_size = 6624
        
        train_data = data[:train_size]
        
        vali_data = data[train_size:train_size + val_size]
        
        test_data = data[train_size + val_size:]
        
        if scaler is None:
           self.scaler.fit(train_data)
        
        if len(train_data)>0:
            train_data = self.scaler.transform(train_data)  
        if len(vali_data)>0:
            vali_data = self.scaler.transform(vali_data)
        if len(test_data)>0:
            test_data = self.scaler.transform(test_data)
        
        test_labels = []
        
        self.train = torch.tensor(train_data,dtype=torch.float32)
        self.val = torch.tensor(vali_data,dtype=torch.float32)
        self.test = torch.tensor(test_data,dtype=torch.float32)
        
        # Also split date_time
        self.train_date_time = self.date_time[:train_size]
        self.val_date_time = self.date_time[train_size:train_size + val_size]
        self.test_date_time = self.date_time[train_size + val_size:]
        
        if mode == "train":
            self.data = self.train
            self.date_time = self.train_date_time
            
            background_data = []
            for i in range(7):
                start = i * win_size
                end = start + win_size
                if end <= self.train.shape[0]:
                    background_data.append(self.train[start:end])
                else:
                    break
             
            self.background_data = torch.stack(background_data,dim=0)
        elif mode == "val":
            self.data = self.val
            self.date_time = self.val_date_time
        elif mode == "test":
            self.data = self.test
            self.date_time = self.test_date_time
            
            for start in range(0, len(test_data) - self.win_size + 1, self.step):
                end = start + self.win_size - 1
                label = 0
                # label = 1 if any overlap with an anomaly interval
                for a0, a1 in self.anomaly_intervals:
                   if not (end < a0 or start > a1):
                       label = 1
                       break
                
                test_labels.append(label)
        
            self.test_labels = torch.tensor(test_labels,dtype=torch.int64) 
            
        if mode == "train" or mode == "test" or mode == "val" :
            print(f"{mode}: {self.data.shape}")
        else:
            raise ValueError("Invalid mode")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
            #print(f"Test data length: {self.test.shape[0]}, Window size: {self.win_size}, Step: {self.step}, Length: {length}")
            #return length
        else:
            raise ValueError("Invalid mode")

    def __getitem__(self, index):
        i = index
        index = index * self.step
        if self.mode == "train":
            return (self.train[index:index + self.win_size], self.train[index:index + self.win_size], self.train_date_time[index:index + self.win_size])
        elif self.mode == 'val':
            return (self.val[index:index + self.win_size], self.val[index:index + self.win_size], self.val_date_time[index:index + self.win_size])
        elif self.mode == 'test':
            return (self.test[index:index + self.win_size], self.test_labels[i], self.test_date_time[index:index + self.win_size])
        else:
            raise ValueError("Invalid mode")
        

def get_turbine_data_loader(data_path, model, batch_size=64, win_size=100, step=100, mode='train', scaler=None, seed=42):
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)
    dataset = TurbineDataLoader(data_path, win_size, step, model, mode, scaler)
    shuffle = True if mode == 'train' else False
    
    def custom_collate_fn(batch):
        inputs = torch.stack([item[0] for item in batch])  # Stack input tensors
        labels = torch.stack([item[1] for item in batch])  # Stack labels
        date_time = [item[2] for item in batch]  # List of date_time arrays
        return inputs, labels, date_time
        
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0,
                             collate_fn=custom_collate_fn)
                                 
    return data_loader,dataset.features,dataset.scaler,dataset.background_data

