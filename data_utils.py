import json

from torch.utils.data import Dataset, DataLoader
import pickle
import os


class FB17K_237(Dataset):
    def __init__(self, args, type):
        self.args = args
        self.type = type
        if (self.type == 'train') or (self.type == 'valid'):
            with open(os.path.join(args.save_path, type + '_all_triples.pkl'), 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            with open(os.path.join(args.save_path, type + '_head_triples.pkl'), 'rb') as f:
                self.dataset = pickle.load(f)


    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample

    def __len__(self):
        return len(self.dataset)