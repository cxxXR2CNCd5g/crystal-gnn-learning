import os
import pandas as pd
import torch
import torch.nn.functional as F
from ase import io
from torch_geometric.data import Data

scripts_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(scripts_dir, "..", "bulk_data_demo180"))
cif_path = os.path.join(data_dir, "0.cif")
target_path = os.path.join(data_dir, "targets.csv")

atoms = io.read(cif_path)
z = torch.tensor(
    atoms.get_atomic_numbers(),
    dtype=torch.long
)

x = F.one_hot(z - 1, num_classes=100).float()
distance_tensor = torch.tensor(
    atoms.get_all_distances(mic=True),
    dtype=torch.float32
)

cut_off = 4.0
neighbor_mask = (
    (distance_tensor > 0)
    & (distance_tensor <= cut_off)
)
source_nodes, target_nodes = torch.where(neighbor_mask)
edge_index = torch.stack([source_nodes, target_nodes], dim=0)
edge_weight = distance_tensor[source_nodes, target_nodes]

distance_step = 0.2
gaussian_width = 0.2
gaussian_centers = torch.arange(
    0.0,
    cut_off + gaussian_width, distance_step
)

distance_column = edge_weight.view(-1, 1)
edge_attr = torch.exp(
    -(distance_column - gaussian_centers.view(1, -1)) ** 2
    / gaussian_width ** 2 
)
target_df = pd.read_csv(
    target_path,
    header=None,
    names=["structure_id", "target"]
)

sample_id = os.path.splitext(os.path.basename(cif_path))[0]
target_row = target_df[
    target_df["structure_id"].astype(str) == sample_id
]
if target_row.empty:
    raise ValueError(f"没有找到{sample_id}的标签")
target_value = float(target_row.iloc[0]["target"])
y = torch.tensor([target_value], dtype=torch.float32)

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=y,
    z=z
    
)

data.structure_id = sample_id
