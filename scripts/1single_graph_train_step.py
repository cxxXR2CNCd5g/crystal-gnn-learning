import os
import torch
import torch.nn.functional as F
from ase import io
import torch.nn as nn
import pandas as pd

scripts_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(
    scripts_dir,
    "..",
    "bulk_data_demo180"
)

data_dir = os.path.normpath(data_dir)
cif_path = os.path.join(
    data_dir,
    "0.cif"
)

atoms = io.read(cif_path)

atomic_numbers = atoms.get_atomic_numbers()

z = torch.tensor(
    atomic_numbers,
    dtype=torch.long
)

element_indices = z - 1

x = F.one_hot(
    element_indices,
    num_classes = 100
).float()


recovered_z = torch.argmax(x, dim=1) + 1

distance_matrix = atoms.get_all_distances(mic=True)
distance_tensor = torch.tensor(
    distance_matrix,
    dtype=torch.float32
)

cut_off = 4
neighbor_mask = (
    (distance_tensor > 0)
    & (distance_tensor <= 4)
)

source_nodes, target_nodes = torch.where(neighbor_mask)

edge_index = torch.stack(
    [source_nodes, target_nodes],
    dim=0
)

edge_weight = distance_tensor[
    source_nodes,
    target_nodes
]


# 高斯基函数的参数
distance_min = 0.0
distance_max = cut_off
distance_step = 0.2
gaussian_width = 0.2

# 生成高斯基函数的中心
gaussian_centers = torch.arange(
    distance_min,
    distance_max + gaussian_width,
    gaussian_width,
    dtype=edge_weight.dtype
)

# 将【E】改成【E，1】
distance_column = edge_weight.view(-1, 1)

# 将每个距离展开成多个高斯特征
edge_attr = torch.exp(
    -(
        (distance_column - gaussian_centers.view(1, -1)) ** 2
    )
    / gaussian_width ** 2
)

x_source = x[source_nodes]
x_target = x[target_nodes]

x_source = x_source.float()
x_target = x_target.float()
edge_attr = edge_attr.float()

# 沿着特征维度拼接每条边的信息
edge_inputs = torch.cat(
    [x_target, x_source, edge_attr],
    dim=1
)

# 记录各个部分的特征维度
node_feature_dim = x.shape[1]
edge_feature_dim = edge_attr.shape[1]

target_part = edge_inputs[:, :node_feature_dim]
source_part = edge_inputs[
    :,
    node_feature_dim:2 * node_feature_dim
]
distance_part = edge_inputs[
    :,
    2 * node_feature_dim:
]

torch.manual_seed(42)

# 自动读取输入和输出特征维度
num_nodes = x.shape[0]

node_feature_dim = x.shape[1]

edge_inputs_dim = edge_inputs.shape[1]
# 消息生成层
message_layer = nn.Linear(
    in_features=edge_inputs_dim,
    out_features=node_feature_dim
)

# 图级预测层
# 将整个晶体的100维度特征转化成一个性质的预测值

readout_layer = nn.Linear(
    in_features=node_feature_dim,
    out_features=1
)
# 为每条边生成消息
edge_messages = torch.relu(
    message_layer(edge_inputs)
)
# 创建一个全零系欸但消息矩阵形状与x相同【n， 100】
aggregated_messages = torch.zeros_like(x)

# 根据target_nodes,把边消息累加到接收节点
aggregated_messages.index_add_(
    dim=0,
    index=target_nodes,
    source=edge_messages
)

# 统计每个节点接受了多少条边的消息
neighbor_counts = torch.bincount(
    target_nodes,
    minlength=num_nodes
)

# 转换成浮点数，并调整成【n，1】
neighbor_counts = neighbor_counts.float().view(-1, 1)

# 防止某个节点没有邻居是出现除以0
neighbor_counts = neighbor_counts.clamp(min=1)
# 将消息求和改成消息pingjun
aggregated_messages = (
    aggregated_messages / neighbor_counts
)
new_x = x + aggregated_messages

# 对所有节点求平均，得到整个晶体的特征  【n，100】 - 【100】
graph_feature = new_x.mean(dim=0)

prediction = readout_layer(
    graph_feature
).squeeze()

target_path = os.path.join(
    data_dir,
    "targets.csv"
)
# 读取标签的表格，没有表头，所以手动指定列名
target_df = pd.read_csv(
    target_path,
    header=None,
    names=["structure_id","target"]
)

# 从0.cif 中提取样本编号“0”\\
sample_id = os.path.splitext(
    os.path.basename(cif_path)
)[0]

# 从target_df中找到structure_id等于sample_id的那一行
target_row =target_df[
    target_df["structure_id"].astype(str) == sample_id
]

# 没有找到标签，主动报错

if target_row.empty:
    raise ValueError(
        f"没有找到结构{sample_id}对应的标签"
    )

# 取出标签，并转化为float32 标量

target_value = float(
    target_row.iloc[0]["target"]
)
target = torch.tensor(
    target_value,
    dtype=torch.float32
)
# 均方误差损失函数
criterion = nn.MSELoss()

# 优化器同时管理消息层和预测层的参数
optimizer = torch.optim.Adam(
    list(message_layer.parameters())
    + list(readout_layer.parameters()),
    lr=0.001
)
# 保存更新前的消息层权重
weight_before = (
    message_layer.weight
    .detach()
    .clone()
)

# 清除上一次保留的梯度

optimizer.zero_grad()

loss = criterion(
    prediction,
    target
)

loss.backward()

weight_gradient = (
    message_layer.weight.grad
    .detach()
    .clone()
)

optimizer.step()

weight_after = (
    message_layer.weight
    .detach()
    .clone()
)

print("\n样本编号：", sample_id)
print("真实标签：", target)
print("预测值：", prediction)
print("损失值：", loss)
print("预测值形状：", prediction.shape)
print("标签形状：", target.shape)

print("\n消息层权重梯度形状：")
print(weight_gradient.shape)

print("\n梯度绝对值平均值：")
print(weight_gradient.abs().mean())

print("\n权重更新前后是否完全相同：")
print(torch.equal(weight_before, weight_after))

print("\n权重最大变化量：")
print(
    (weight_after - weight_before)
    .abs()
    .max()
)



