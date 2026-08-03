import os

from crystal_gnn.dataset import build_graph_list
from crystal_gnn.data_loader import create_data_loaders


# 当前 scripts 文件夹
scripts_dir = os.path.dirname(
    os.path.abspath(__file__)
)

# 项目根目录 learn_gnn
project_dir = os.path.dirname(scripts_dir)

# 读取已经预处理好的 .pt 晶体图
graph_dir = os.path.join(
    project_dir,
    "graph_data_20000"
)

# 创建按需读取 .pt 文件的数据集
dataset = build_graph_list(
    graph_dir=graph_dir
)

# 划分训练集、验证集和测试集
train_loader, val_loader, test_loader = create_data_loaders(
    dataset=dataset,
    batch_size=32,
    train_ratio=0.8,
    val_ratio=0.1,
    random_seed=42,
    num_workers=0
)

print("图数据目录：", graph_dir)
print("全部样本数：", len(dataset))

print("\n训练集：")
print("样本数：", len(train_loader.dataset))
print("批次数：", len(train_loader))

print("\n验证集：")
print("样本数：", len(val_loader.dataset))
print("批次数：", len(val_loader))

print("\n测试集：")
print("样本数：", len(test_loader.dataset))
print("批次数：", len(test_loader))