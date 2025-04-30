import numpy as np
import os
from skimage import transform
from torch.utils.data import Dataset
import torch

def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

class CustomImageDataset(Dataset):
    def __init__(self, base, data_name, datasize, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.image_loc = os.path.join(base, data_name)
        self.datasize = datasize

        # Memory-map the file: does NOT load into RAM!
        self.cube = np.memmap(self.image_loc, dtype=np.float32, mode='r')
        self.cube = self.cube.reshape((600, 600, 600))  # Adjust axis order if needed

        # Precompute slice indices
        self.slice_list = []
        for axis in range(3):  # 0=x, 1=y, 2=z
            for idx in range(600):
                self.slice_list.append((axis, idx))

    def __len__(self):
        return len(self.slice_list)  # 1800 slices

    def __getitem__(self, idx):
        axis, slice_idx = self.slice_list[idx]
        #print('axis = {}, idx = {}'.format(axis,idx))
        if axis == 0:  # x-slice
            image = self.cube[slice_idx, :, :]
        elif axis == 1:  # y-slice
            image = self.cube[:, slice_idx, :]
        elif axis == 2:  # z-slice
            image = self.cube[:, :, slice_idx]
        else:
            raise ValueError('Invalid axis')
        label = f'z=12.0'
        image = transform.resize(image, (self.datasize, self.datasize))
        image = normalize_2d(image)

        # Convert to tensor
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
