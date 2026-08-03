import os
import torch

from torch.utils.data import Dataset


class CrystalGraphDataset(Dataset):
    """按需读取已经预处理好的 .pt 晶体图文件。"""

    def __init__(self, graph_dir, max_samples=None):
        if not os.path.isdir(graph_dir):
            raise FileNotFoundError(f"找不到图数据目录：{graph_dir}")

        graph_files = [
            filename
            for filename in os.listdir(graph_dir)
            if filename.endswith(".pt")
        ]

        # 让 0.pt、1.pt、2.pt 按数字顺序排列
        graph_files.sort(
            key=lambda filename: int(os.path.splitext(filename)[0])
        )

        if max_samples is not None:
            graph_files = graph_files[:max_samples]

        if len(graph_files) == 0:
            raise RuntimeError(f"目录中没有找到 .pt 文件：{graph_dir}")

        self.graph_dir = graph_dir
        self.graph_files = graph_files

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, index):
        graph_path = os.path.join(
            self.graph_dir,
            self.graph_files[index]
        )

        return torch.load(
            graph_path,
            weights_only=False
        )


def build_graph_list(graph_dir, max_samples=None):
    """保留原来的函数名，返回可被 DataLoader 使用的数据集。"""

    return CrystalGraphDataset(
        graph_dir=graph_dir,
        max_samples=max_samples
    )