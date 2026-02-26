import torch
import torch.nn as nn
import numpy as np
from solver import my_kl_loss

class ModelWrapperForSHAP(nn.Module):
    def __init__(self, model, win_size, temperature=50):
        super().__init__()
        self.model = model
        self.win_size = win_size
        self.temperature = temperature
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, input_data):
        # Ensure model is in eval mode
        self.model.eval()
        input_data = input_data.to(next(self.model.parameters()).device) # Ensure input is on model's device

        output, series, prior, sigmas = self.model(input_data)

        loss = torch.mean(self.criterion(input_data, output), dim=-1) # Shape: [batch_size, window_size]
        
        #rec_error = loss.mean(dim=1, keepdim=True)

        # --- Start KL loss calculation ---
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * self.temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                    series[u].detach()) * self.temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * self.temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                    series[u].detach()) * self.temperature

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        #cri = cri.detach().cpu().numpy()        
        # --- RETURN VALUE FOR SHAP ---
        # Explain the average score at the window
        discp_score = torch.mean(cri, dim=1, keepdim=True) 
        
        return  discp_score #torch.cat([rec_error, discp_score],dim=1)  #cri[:, -1] # Shape: [batch_size]
