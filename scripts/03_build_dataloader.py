import os

from torch_geometric.loader import DataLoader

from crystal_gnn.dataset import build_graph_list


scripts_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(scripts_dir)

graph_dir = os.path.join(
    project_dir,
    "graph_data_20000"
)

# 这里返回的是按需读取 .pt 文件的数据集
# 不会一次把 20000 张图全部放入内存
dataset = build_graph_list(
    graph_dir=graph_dir
)

batch_size = 16

data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

print("图数据目录：", graph_dir)
print("晶体图总数：", len(dataset))
print("每个批次最多包含：", batch_size)
print("批次数量：", len(data_loader))

# 只读取并检查第一个批次
for batch_index, batch in enumerate(data_loader):
    print("\n当前批次编号：", batch_index)
    print("当前批次晶体数：", batch.num_graphs)
    print("节点特征形状：", batch.x.shape)
    print("边连接形状：", batch.edge_index.shape)
    print("边特征形状：", batch.edge_attr.shape)
    print("标签形状：", batch.y.shape)
    print("节点所属晶体：", batch.batch.shape)
    print("晶体节点边界：", batch.ptr)

    break