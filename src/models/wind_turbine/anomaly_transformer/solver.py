import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_loader.turbine_data_loader import get_turbine_data_loader
#import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import re
import pandas as pd
import joblib
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from plotter import *
from scipy.signal import find_peaks
from torch.autograd import Function

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
	
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0.0):
        self.patience    = patience
        self.verbose     = verbose
        self.counter     = 0
        self.best_score2 = None
        self.early_stop  = False
        self.val_loss2_min = np.Inf
        self.delta       = delta
        self.dataset     = dataset_name

        self.call_count = 0
    def __call__(self, val_loss2, model, path):
        self.call_count += 1
        
        if self.call_count <= 0:
            if self.verbose:
                print("EarlyStopping: first validation—no check performed.")
            return
        
 
        score2 = val_loss2

        if self.best_score2 is None:
            self.best_score2 = score2
            self.save_checkpoint(val_loss2, model, path)

        elif score2 >= self.best_score2 + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            # Improvement on the PRIOR objective → reset
            self.best_score2 = score2
            self.save_checkpoint(val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss2_min:.6f} --> {val_loss2:.6f}).  Saving model …')
        torch.save(model.state_dict(),path)
        self.val_loss2_min = val_loss2

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x
    @staticmethod
    def backward(ctx, grad_output):
        # flip sign and scale by lambd
        return grad_output.neg() * ctx.lambd, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradReverse.apply(x, self.alpha)
 
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.device = torch.device("cpu")
        self.scaler=None
        scaler_pth = os.path.join(self.scaler_path, 'scaler.joblib')
        data_path = os.path.join(str(self.data_path), str(self.mode))
        model = self.dataset+self.model_
        if self.mode=='test':
            self.scaler = joblib.load(scaler_pth)
            print(f"scaler loaded from {scaler_pth}")
            self.test_loader,self.features, _, _  = get_turbine_data_loader(data_path, model, batch_size=self.batch_size, win_size=self.win_size, step=self.win_size, mode='test', scaler=self.scaler)
        
        
        if self.mode=='train':            
            self.train_loader,self.features,self.scaler,self.background_data = get_turbine_data_loader(data_path, model, batch_size=self.batch_size, win_size=self.win_size, step=self.win_size, mode='train', scaler=self.scaler)
            
            if self.scaler is not None:
                joblib.dump(self.scaler, scaler_pth)
                print(f"scaler saved to {scaler_pth}.")
            
            background_data = self.background_data.float().to(self.device)
            print(f"Background data shape: {background_data.shape}")
            bg_file = os.path.join(self.background_data_path, "background_data.pt")
            torch.save(background_data, bg_file) # Save CPU tensor
            print(f"Background data saved to {bg_file}")  
            
        
            self.vali_loader,self.features, _, _  = get_turbine_data_loader(data_path, model, batch_size=self.batch_size, win_size=self.win_size, step=self.win_size, mode='val', scaler=self.scaler)
        
        self.build_model()
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.grl = GradientReversalLayer(alpha=1.0)
        
        fusion_params = [self.model.logits_e, self.model.logits_h]
        fusion_param_ids = list(map(id, fusion_params))

        core_params = [p for p in self.model.parameters() if id(p) not in fusion_param_ids]

        self.optimizer = torch.optim.Adam(core_params, lr=self.lr)
        self.fusion_optimizer = torch.optim.Adam([
            {'params': fusion_params, 'lr': self.lr}
        ])
        
    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        loss = []
        for i, (input_data, _, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            
            series_loss = 0.0
            prior_loss = 0.0
            
            for u in range(len(prior)):
                series_loss += torch.mean(my_kl_loss(self.grl(series[u]), (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach()))
                prior_loss += torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), series[u].detach()))
            
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            rec_loss = self.criterion(output, input)
            
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())
            
        return np.average(loss_2), np.average(loss_1)

    
    def train(self):
        print("======================TRAIN MODE======================")
        time_now = time.time()
        path = self.model_save_path
        checkpoint_file = os.path.join(path, str(self.dataset)+str(self.model_)+'_checkpoint1_.pth')
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        early_stopping = EarlyStopping(patience=7, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        
        if os.path.exists(checkpoint_file):
            print(f"Loading checkpoint from {checkpoint_file}...")
            self.model.load_state_dict(torch.load(checkpoint_file))
            

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            epoch_time = time.time()
            self.model.train()
            for i, (input_data,labels,date_time) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)
                
                series_loss = 0.0
                prior_loss = 0.0
                
                for u in range(len(prior)):
                    series_loss += torch.mean(my_kl_loss(self.grl(series[u]), (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach()))
                    prior_loss += torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), series[u].detach()))
                
                series_loss = series_loss / len(series)
                prior_loss = prior_loss / len(prior)
                
                rec_loss = self.criterion(output, input)
                
                loss1_list.append((rec_loss + self.k * prior_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss
                #loss = rec_loss + series_loss + prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()
                
                self.fusion_optimizer.zero_grad()

                output = self.model(input,use_fused_series=True)
    
                fused_loss = self.criterion(output, input)

                fused_loss.backward()
                self.fusion_optimizer.step()
                

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list) #loss1_list[0].detach().numpy() #np.average(loss1_list.detach().numpy())
            vali_loss1,vali_loss2 = self.vali(self.vali_loader)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, self.model, checkpoint_file)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            #adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
        
        print(f"Model saved to {path}")
        self.thre()
            
    def thre(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + str(self.model_) + '_checkpoint1_.pth'))
        )
        self.model.eval()
        temperature = 50

        print("======================THRE MODE======================")

        criterion = nn.MSELoss(reduction='none')

        # (1) statistic on the train set
        attens_energy = []
        
        original_data = []
        reconstructed_data = []
        
        for i, (input_data, _,_) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            
            output, series, prior, sigmas = self.model(input)
            
            features = self.features
            
            print(input.cpu().numpy().shape)
            
            original_data.append(input.cpu().numpy())
            reconstructed_data.append(output.detach().cpu().numpy())
            
            print(len(original_data),len(reconstructed_data))

            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        
        th = self.anormly_ratio
        thresh = np.percentile(train_energy, 100 - th)
        print(f"Threshold ({th}) : {thresh}")
    
    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + str(self.model_) + '_checkpoint1_.pth')), strict=False
        )
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')

        threshmap={}
        
        threshmap[1.5]={ '0' : 0.00013377491304709106, '1' : 0.00044517662157886675, '2' : 0.00014006024604896057, '3': 7.12080005177995e-05 } 
        thresh = threshmap.get(self.anormly_ratio,1.5)[self.model_] if self.thresh is None else self.thresh
        print(f"Threshold: {thresh}")
        
        def log_percentile_threshold(row: torch.Tensor, q: float = 0.5, eps=1e-44):
            """
            row: 1-D tensor of attention/prior weights (sums to 1).
            q: quantile in [0,1] on the log10 scale (0.5=median, 0.75=75th perc, etc).
            eps: floor so log10(row+eps) is well-defined.
            Returns:
            idxs: indices in the row whose weight >= 10**(quantile of log10-values)
            """
            # take only the positive entries (others might be exact zero)
            positive = row[row > 0]
            logs = torch.log10(positive + eps)  # their exponents, e.g. -3, -10, -40
    
            # find the q-quantile exponent
            thr_exp = torch.quantile(logs, q)   # e.g. -5.2 if the median log is -5.2
            thr     = 10 ** thr_exp             # back to linear scale
    
            # pick everything above that
            return torch.where(row >= thr)[0]
    
        attens_energy = []
        anomaly_timestamps = []
        reconstructed_outputs = []
        
        
        sensor_names = self.features[1:]
        window_records_global = []
        window_records_local = []
        
        anomalous_inputs_to_explain = []
        
        input_file = os.path.join(self.anomalous_inputs_path, "anomalous_inputs.pt")

        true_labels = []
        pred_labels = []
        
        for i, (input_data, labels, date_time) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            output, series, prior, sigmas = self.model(input)
            
            full_series = torch.stack(list(series),dim=0)
            #full_series_sum = full_series[0][0].sum(dim=-1)
            #print(full_series.shape,full_series_sum.shape,full_series_sum.min().item(),full_series_sum.max().item())
            
            #print(torch.stack(list(series),dim=-1).shape) # torch.tensor(series).shape)
            
            e, b, h, L, _ = full_series.shape

            # 1) parametrized weights for layers and heads
            logits_e = self.model.logits_e             # one per layer
            logits_h = self.model.logits_h             # one per head

            # 2) softmax to get non-negative weights that sum to 1
            w_e = torch.softmax(logits_e, dim=0)                # [e]
            w_h = torch.softmax(logits_h, dim=0)                # [h]

            # 3) fuse with broadcasting, then sum
            #    full_series: [e, b, h, L, L]
            #    w_e.view(e,1,1,1,1) --> [e,1,1,1,1]
            #    w_h.view(1,1,h,1,1) --> [1,1,h,1,1]
            series_fused = (w_e.view(e,1,1,1,1) * w_h.view(1,1,h,1,1) * full_series).sum(dim=(0,2))

            
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

                    
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            
            stored_windows_in_batch = [False] * cri.shape[0]
            
            attens_energy.append(cri)
            cri = attens_energy[i]
            
            #print(cri.shape,input_data.shape)
            
            for j in range(len(cri)):
               #count = 0
               for k in range(len(cri[j])):
                   if cri[j][k] > thresh:
                       #print(date_time[j][k])
                       # Anomaly detected, add corresponding date and timestamp
                       anomaly_timestamps.append(date_time[j][k])
                       #count+=1
                       
            input_np_ = input_data.cpu().numpy()       # shape: (B, win_size, num_features)
            output_np_ = output.detach().cpu().numpy() # shape: (B, win_size, num_features)
            date_np = np.array(date_time)             # shape: (B, win_size)
            cri_np = cri                              # shape: (B, win_size)
            loss_np = loss.detach().cpu().numpy()
            
            true_batch = labels.view(-1).cpu().numpy().astype(int).tolist()
            pred_batch = []
            
            def robust_peak_idx(cri, smooth_k, min_prom, min_width, thresh):
                """
                Return the index of the most 'structural' peak in cri.
    
                - cri:      1D anomaly‐score sequence, length L
                - smooth_k: smoothing kernel size (odd int)
                - min_prom: minimum prominence unit
                - min_width: minimum width in samples
    
                """
                # 1) smooth
                kernel = np.ones(smooth_k) / smooth_k
                smooth = np.convolve(cri, kernel, mode="same")

                # 2) find peaks
                peaks, props = find_peaks(
                    smooth,
                    prominence=min_prom * thresh,
                    width=min_width
                )
                valid = [i for i,p in enumerate(peaks) if smooth[p] > thresh]
                #print(len(valid))

                if len(peaks):
                    # rank by prom*width to get the most structural
                    scores = props["prominences"] * props["widths"]
                    return int(peaks[np.argmax(scores)])
                else:
                    # fallback to the highest smoothed point
                    return int(np.argmax(smooth))
        
            sub_w    = 12
            n_sub    = input_np_.shape[1] // sub_w  # 12

            for b in range(input_np_.shape[0]):
                
                # Compute the average sensor values over the window.
                window_mean = input_np_[b][:,1:].mean(axis=0)
                # Compute discrepancy score for the window
                discrepancy_score = np.mean(cri_np[b])
                # Get the last timestamp of the window.
                timestamp = date_np[b][-1][0]
                
                fault_label = int(np.sum(cri_np[b]>thresh) >= np.round(self.anormly_ratio * cri_np[b].shape[0]/100)) 
                
                pred_batch.append(fault_label)

                if fault_label == 0:
                    # Create a record with timestamp, sensor averages,
                    # discrepancy score, and fault label.
                    record = {"timestamp": timestamp}
                    record["window"] = b+1
                else:
                    max_idx = robust_peak_idx(cri_np[b], smooth_k=3, min_prom=0.5, min_width=2, thresh=thresh)
                    #print(max_idx)
                    timestamp = date_np[b][max_idx][0]
                    
                    #if max_idx==53:
                    #    #print(b,max_idx)
                    #    plot_window_scores(cri_np[b], thresh, max_idx)
                    #    plot_attention_with_hotspots(series_fused[b], idx=max_idx,timestamp=timestamp)
                    
                    record = {"timestamp": timestamp}
                    record["window"] = b+1
                    attn_idcs = log_percentile_threshold(series_fused[b,max_idx], q=0.9).cpu().numpy()
                    window_mean = input_np_[b][attn_idcs,1:].mean(axis=0)
                    discrepancy_score = cri_np[b][max_idx] #np.mean(cri_np[b][attn_idcs])
                    #print(max_idx,attn_idcs.shape,window_mean.shape,discrepancy_score)
                    anomalous_inputs_to_explain.append(input_data[b].cpu())
                
                for j, val in enumerate(window_mean):
                        record[sensor_names[j]] = val
                record["discrepancy_score"] = discrepancy_score
                record["fault"] = fault_label

                window_records_global.append(record)
                
                #fault_label = int(np.sum(cri_np[b]>thresh) >= np.round(self.anormly_ratio * cri_np[b].shape[0]/100))
                
                for s in range(n_sub):
                    start = s * sub_w
                    end   = start + sub_w

                    # slice sub-window
                    Xsub = input_np_[b, start:end]   # (sub_w, num_features)
                    Csub = cri_np[b,  start:end]     # (sub_w,)
                    Dsub = date_np[b, start:end]     # (sub_w,)

                    max_idx = robust_peak_idx(Csub, smooth_k=3, min_prom=0.5, min_width=2, thresh=thresh)
                    
                    # 1) avg sensor values (skip column 0 if that’s a timestamp)
                    window_mean = Xsub[:max_idx+1, 1:].mean(axis=0)

                    # 2) peak discrepancy
                    discrepancy_score = Csub[max_idx]   #max(Csub) #mean()

                    # 3) last timestamp in sub-window
                    timestamp = Dsub[max_idx][0]

                    # 6) build record
                    record = {
                    "timestamp": timestamp,
                    "window":    b + 1,
                    "hour": (sub_w//6)*(s + 1)
                    }

                    # fill sensor means
                    for j, val in enumerate(window_mean):
                        record[sensor_names[j]] = val
                    record["discrepancy_score"] = discrepancy_score
                    record["fault"]             = fault_label

                    window_records_local.append(record)
                   
            true_labels.extend(true_batch)
            pred_labels.extend(pred_batch)
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        
        pred = (test_energy > thresh).astype(int)

        print("Anomaly detection complete. Number of fault windows detected:", sum(pred_labels))
        
        if self.model_=='0':  #slicing out a window for test time
            anomalous_inputs_to_explain.pop(40)
            window_records_global.pop(40)
            del window_records_local[40*n_sub:41*n_sub]
            
        anomalous_inputs_tensor = torch.stack(anomalous_inputs_to_explain)
        
        anomalous_inputs_dict = { 'anomalous_inputs' : anomalous_inputs_tensor, 'feature_names' : self.features }
        torch.save(anomalous_inputs_dict, input_file)
        print(f"Anomalous inputs saved to {input_file}")
        
        #window_records = window_records[40*n_sub:41*n_sub]
        
        fd_global = os.path.join(self.discrepancy_table_path, "window_discp_table_with_fault_global.csv")
        fd_local = os.path.join(self.discrepancy_table_path, "window_discp_table_with_fault_local.csv")
        
        # Create a DataFrame from the window records and save it to CSV.
        df_out = pd.DataFrame(window_records_global)
        df_out.to_csv(f"{fd_global}", index=False)
        print(f"Global fault window data saved in {fd_global}")
        
        df_out = pd.DataFrame(window_records_local)
        df_out.to_csv(f"{fd_local}", index=False)
        print(f"Local fault window data saved in {fd_local}")
        
        #pred_labels = [0,1,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,1,1]
        
        acc  = accuracy_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels, zero_division=0)
        rec  = recall_score(true_labels, pred_labels, zero_division=0)
        f1   = f1_score(true_labels, pred_labels, zero_division=0)

        print(f"Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
        
        """
        #fig = plot_and_save_confusion(true_labels, pred_labels, "Gearbox Fault", save_path=".")
        fig = plot_confusion_with_metrics(
            true_labels=true_labels,
            pred_labels=pred_labels,
            fault_event_name="Transformer Fault",
            save_path=".",
            cmap='YlGnBu'
        )
        plt.show()
        """

        return pred
        
