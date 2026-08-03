import os
import torch
import torch.nn.functional as F
from ase import io
from torch_geometric.data import Data

def cif_to_graph(
        cif_path,
        target,
        cut_off=4.0,
        distance_step=0.2,
        gaussian_width=0.2
):
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"找不到CIF文件：{cif_path}")

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

    neighbor_mask = (
        (distance_tensor > 0)
        & (distance_tensor <= cut_off)
    )

    source_nodes, target_nodes = torch.where(neighbor_mask)
    edge_index = torch.stack(
        [source_nodes, target_nodes],
        dim = 0
    )

    edge_weight = distance_tensor[source_nodes, target_nodes]

    gaussian_centors = torch.arange(
        0.0,
        cut_off + distance_step,
        gaussian_width,
        dtype=edge_weight.dtype
    )

    distance_column = edge_weight.view(-1, 1)

    edge_attr = torch.exp(
        - (distance_column - gaussian_centors.view(1, -1)) ** 2
        / gaussian_width ** 2
    )

    y = torch.tensor([float(target)], dtype=torch.float32)

    structure_id  = os.path.splitext(os.path.basename(cif_path))[0]

    data = Data(
        x=x,
        z=z,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )
    data.structure_id = [structure_id]

    return data
