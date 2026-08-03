import torch

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader


def create_data_loaders(
    dataset,
    batch_size=16,
    train_ratio=0.8,
    val_ratio=0.1,
    random_seed=42,
    num_workers=0
):
    """把完整晶体图数据集划分为训练集、验证集和测试集。"""

    num_total = len(dataset)

    num_train = int(train_ratio * num_total)
    num_val = int(val_ratio * num_total)
    num_test = num_total - num_train - num_val

    generator = torch.Generator()
    generator.manual_seed(random_seed)

    train_set, val_set, test_set = random_split(
        dataset,
        [num_train, num_val, num_test],
        generator=generator
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader