import torch

from nnsimple.utils.data.custom_dataset import CustomImageDataset


def get_mean_std(dataset):
    if not isinstance(dataset, CustomImageDataset):
        raise TypeError("Object passed is not an instance of CustomImageDataset class")

    data = []
    target = []

    for _, (x, y) in enumerate(dataset):
        data.append(x)
        target.append(y)

    data = torch.stack(data, 0)
    data = torch.permute(data, (0, 2, 3, 1))

    mean = data.mean(axis=(0, 1, 2))
    std = data.std(axis=(0, 1, 2))
    return mean, std


