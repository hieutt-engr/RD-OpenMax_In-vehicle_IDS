import random
import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import Dataset
    
class CANDatasetEnet(Dataset):
    def __init__(self, root_dir, window_size, is_train=True, include_data=False, transform=None, target_size=64):
        self.target_size = target_size
        if is_train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'val')
            
        self.include_data = include_data
        self.is_train = is_train
        self.transform = transform
        self.window_size = window_size
        self.total_size = len(os.listdir(self.root_dir))
    
    def __getitem__(self, idx):
        filenames = '{}/{}.tfrec'.format(self.root_dir, idx)
        index_path = None
        description = {'id_seq': 'int', 'data_seq': 'int', 'timestamp': 'float', 'label': 'int'}
        dataset = TFRecordDataset(filenames, index_path, description)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        data = next(iter(dataloader))
        
        id_seq, data_seq, timestamp, label = data['id_seq'], data['data_seq'], data['timestamp'], data['label']

        id_seq = id_seq.to(torch.float).view(self.target_size, self.target_size)
        data_seq = data_seq.to(torch.float).view(self.target_size, self.target_size)
        timestamp = timestamp.to(torch.float).view(self.target_size, self.target_size)

        combined_tensor = torch.stack([id_seq, data_seq, timestamp], dim=0)

        if self.transform:
            combined_tensor = self.transform(combined_tensor)
        return combined_tensor, label[0][0]
        
    def __len__(self):
        return self.total_size

class CANDataset5(Dataset):
    def __init__(self, root_dir, window_size, is_train=True, include_data=False, transform=None, target_size=64):
        self.target_size = target_size
        self.window_size = window_size
        self.include_data = include_data
        self.transform = transform

        if is_train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'val')

        self.total_size = len(os.listdir(self.root_dir))

        # TFRecord schema with all 5 channels
        self.description = {
            'id_seq': 'int',
            'data_seq': 'int',
            'timestamp': 'float',
            'delta_ts': 'float',
            'label': 'int'
        }

    def __getitem__(self, idx):
        filenames = '{}/{}.tfrec'.format(self.root_dir, idx)
        dataset = TFRecordDataset(filenames, index_path=None, description=self.description)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        data = next(iter(dataloader))

        # Extract and reshape
        id_seq = data['id_seq'].to(torch.float32).view(self.target_size, self.target_size)
        data_seq = data['data_seq'].to(torch.float32).view(self.target_size, self.target_size)
        timestamp = data['timestamp'].to(torch.float32).view(self.target_size, self.target_size)
        delta_ts = data['delta_ts'].to(torch.float32).view(self.target_size, self.target_size)

        # Stack into 4-channel tensor
        combined_tensor = torch.stack([id_seq, data_seq, timestamp, delta_ts], dim=0)

        if self.transform:
            combined_tensor = self.transform(combined_tensor)

        return combined_tensor, data['label'][0][0]

    def __len__(self):
        return self.total_size

class OE_CANDataset(Dataset):
    def __init__(self, root_dir, window_size, sample_ratio=0.3, is_train=True, transform=None, target_size=64):
        self.target_size = target_size
        self.window_size = window_size
        self.transform = transform

        if is_train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'val')

        self.filenames = sorted(os.listdir(self.root_dir))
        self.total_size = len(self.filenames)

        # Randomly sample subset for OE
        sample_size = int(self.total_size * sample_ratio)
        self.filenames = random.sample(self.filenames, sample_size)

        self.description = {
            'id_seq': 'int',
            'data_seq': 'int',
            'timestamp': 'float',
            'delta_ts': 'float',
            'label': 'int'
        }

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = os.path.join(self.root_dir, self.filenames[idx])

        # Load the TFRecord file
        dataset = TFRecordDataset(filename, index_path=None, description=self.description)

        # Scan for a valid attack sample (label ≠ 0)
        for data in dataset:
            label_val = int(data['label'][0])
            if label_val != 0:
                # Valid OE sample
                id_seq = torch.tensor(data['id_seq'], dtype=torch.float32).view(self.target_size, self.target_size)
                data_seq = torch.tensor(data['data_seq'], dtype=torch.float32).view(self.target_size, self.target_size)
                timestamp = torch.tensor(data['timestamp'], dtype=torch.float32).view(self.target_size, self.target_size)
                delta_ts = torch.tensor(data['delta_ts'], dtype=torch.float32).view(self.target_size, self.target_size)

                x = torch.stack([id_seq, data_seq, timestamp, delta_ts], dim=0)

                if self.transform:
                    x = self.transform(x)

                y = -1  # OE label
                return x, y

        # Fallback: no attack sample found → skip by random resample
        new_idx = random.randint(0, len(self.filenames) - 1)
        return self.__getitem__(new_idx)


from tfrecord.torch.dataset import TFRecordDataset

class OEOfflineDataset(torch.utils.data.Dataset):
    def __init__(self, tfrecord_path, target_size=64, transform=None):
        self.target_size = target_size
        self.transform = transform
        self.samples = []  # ✅ preload toàn bộ sample có label ≠ 0

        dataset = TFRecordDataset(
            tfrecord_path,
            index_path=tfrecord_path + ".index",
            description={
                'id_seq': 'float',
                'data_seq': 'float',
                'timestamp': 'float',
                # 'delta_ts': 'float',
                'label': 'int',
            }
        )

        for data in dataset:
            if int(data['label'][0]) != 0:
                self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        id_seq = torch.tensor(data['id_seq'], dtype=torch.float32).view(self.target_size, self.target_size)
        data_seq = torch.tensor(data['data_seq'], dtype=torch.float32).view(self.target_size, self.target_size)
        timestamp = torch.tensor(data['timestamp'], dtype=torch.float32).view(self.target_size, self.target_size)
        # delta_ts = torch.tensor(data['delta_ts'], dtype=torch.float32).view(self.target_size, self.target_size)
        x = torch.stack([id_seq, data_seq, timestamp], dim=0)

        if self.transform:
            x = self.transform(x)

        return x, -1




