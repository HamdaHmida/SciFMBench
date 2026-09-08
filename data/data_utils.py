import numpy as np
import h5py
import torch

class Parametric_2D_Dataset():
    def __init__(
        self,
        list_files,
        list_para,
        T_in=1,
        T_out=10,
        start_sample = 0,
        finish_sample = 1000,
        reduced_resolution=1,
        reduced_resolution_t=1,
        reduced_batch=1,
        normalize=False,
        epsilon=1e-8,
    ):
        self.list_para = list_para
        for i in range(len(list_files)):
            print(f"Loading file {list_files[i]} : {i+1} / {len(list_files)}")
            with h5py.File(list_files[i], "r") as f:
                _data = np.array(
                    f["tensor"], dtype=np.float32
                )
                _data = _data[
                    ::reduced_batch,
                    ::reduced_resolution_t,
                    ::reduced_resolution,
                    ::reduced_resolution,
                ]
                
                if not hasattr(self, "data"):
                    self.data = _data[start_sample:finish_sample, :T_in+T_out, :, :, None]
                else:
                    self.data = np.concatenate([self.data, _data[start_sample:finish_sample, :T_in+T_out, :, :, None]], axis=0)
                
                nb_sam= finish_sample - start_sample

                if not hasattr(self, "para"):
                    self.para = np.array([list_para[i]]*nb_sam, dtype=np.float32)
                else:
                    self.para = np.concatenate([self.para, np.array([list_para[i]]*nb_sam, dtype=np.float32)], axis=0)

        if len(self.data.shape) == 6:
            self.data = self.data.squeeze(-2)
        self.data = np.transpose(self.data[:, :, :, :], (0, 2, 3, 1, 4))

        self.normalize = normalize
        self.epsilon = epsilon
        if self.normalize:
            # Calculate mean and std per channel across all samples, spatial points, and time
            # Shape of self.data is [B, Nx, Ny, T, C]
            # We compute over axes (0, 1, 2, 3) to get stats for each channel in C
            self.mean = np.mean(self.data, axis=(0, 1, 2, 3), keepdims=True)
            self.std = np.std(self.data, axis=(0, 1, 2, 3), keepdims=True)
            
            # Per-variable normalization: (x - mean) / std
            self.data = (self.data - self.mean) / (self.std + self.epsilon)
            print(f"Dataset normalized using Mean/Std. Means: {self.mean.flatten()}, Stds: {self.std.flatten()}")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.para[idx]
    
    def denormalize(self, x):
        """Reverts the Mean-Std normalization for visualization or metrics"""
        if self.normalize:
            # Handles both numpy and torch tensors
            return x * (self.std + self.epsilon) + self.mean
        else:
            print("Dataset was not normalized")
            return x
