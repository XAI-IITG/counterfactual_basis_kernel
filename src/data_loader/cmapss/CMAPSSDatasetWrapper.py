import torch
from torch.utils.data import Dataset

# class CMAPSSDatasetWrapper(Dataset):
#     def __init__(self, sequences, targets):
#         self.sequences = sequences
#         self.targets = targets
    
#     def __len__(self):
#         return len(self.sequences)
    
#     def __getitem__(self, idx):
#         return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])
    

class CMAPSSDatasetWrapper(Dataset):
    def __init__(self, sequences, targets_cycles, label_mode="scaled", max_rul=125):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        y = torch.tensor(targets_cycles, dtype=torch.float32)
        self.y = (y / float(max_rul)) if label_mode == "scaled" else y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]