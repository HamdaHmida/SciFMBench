from data import Parametric_2D_Dataset
import torch


list_of_files = [
    "/home/hamda/PhD_Thesis/dataset/Burgers_nu_0.1.hdf5",
    "/home/hamda/PhD_Thesis/dataset/Burgers_nu_0.01.hdf5",
]

list_of_PDE_parameters = [
    [0.1],  # nu
    [0.01],
]

train_dataset = Parametric_2D_Dataset(
    list_files = list_of_files,
    list_para = list_of_PDE_parameters,
    T_out = 30,
    finish_sample=50,
    )

_batch_size=2
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=_batch_size, shuffle=True, num_workers=2
)

if __name__ == "__main__":
    # Test the data loader
    for i, (data, para) in enumerate(train_loader):
        print(f"Batch {i}: data shape = {data.shape}, para shape = {para.shape}")
        if i >= 2:
            break
