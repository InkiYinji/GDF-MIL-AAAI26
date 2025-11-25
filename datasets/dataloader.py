import pandas as pd
import numpy as np
import math
import torch
import h5py
from torch.utils.data import Dataset


class DatasetWSI(Dataset):
    def __init__(self, dataset_info_csv, dataset_info_feat_path, data_type="h5"):

        if type(dataset_info_csv) == str:
            self.dataset_df = pd.read_csv(dataset_info_csv)
            self.slide_path_list = self.dataset_df['slide_id'].values
            self.labels_list = self.dataset_df['label'].values
        else:
            self.slide_path_list = dataset_info_csv['slide_id'].values
            self.labels_list = dataset_info_csv['label'].values
        self.dataset_info_feat_path = dataset_info_feat_path
        self.data_type = data_type

        self.slide_path_list = [x for x in self.slide_path_list if isinstance(x,str)]
        self.labels_list = [x for x in self.labels_list if not math.isnan(x)]

    def __len__(self):
        return len(self.slide_path_list)

    def __getitem__(self, idx):
        slide_path = self.dataset_info_feat_path + self.slide_path_list[idx].split('.')[0] + '.' + self.data_type
        # print(self.slide_path_list[idx].split('.')[0])
        label = int(self.labels_list[idx])
        label = torch.tensor(label)

        # 打开文件
        if self.data_type == 'h5':
            with h5py.File(slide_path, 'r') as file:
                feat = file['features'][:]
                coords = file['coords'][:]
        elif self.data_type == "csv":
            feat = pd.read_csv(slide_path).values
            coords = torch.rand(1)
        else:
            raise Exception("No such data type!")

        feat = torch.tensor(feat)

        # feat = torch.load(slide_path)
        return feat, label, coords


class DatasetBag(Dataset):

    def __init__(self, bags, bags_label, idx=None):
        """"""
        self.bags = bags
        self.idx = idx
        if self.idx is None:
            self.idx = list(range(len(self.bags)))
        self.num_idx = len(self.idx)
        self.bags_label = bags_label[self.idx]

    # def shuffle(self):
    #     np.random.shuffle(self.idx)

    def __getitem__(self, idx):
        bag = [self.bags[self.idx[idx], 0][:, :-1].tolist()]
        bag = torch.from_numpy(np.array(bag))

        return bag.float(), torch.tensor([self.bags_label[idx].tolist()]).float()

    def __len__(self):
        """"""
        return self.num_idx

