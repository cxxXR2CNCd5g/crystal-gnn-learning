import os
import csv
import math

import torch
import torch.nn as nn

from crystal_gnn.dataset import build_graph_list
from crystal_gnn.data_loader import create_data_loaders
from crystal_gnn.model import CrystalGNN


def train_one_epoch(
        model,
        data_loader,
        optimizer,
        loss_function,
        device
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        batch = batch.to(device)
        targets = batch.y.float().view(-1)

        optimizer.zero_grad()

        predictions = model(batch)
        loss = loss_function(predictions, targets)

        loss.backward()
        optimizer.step()

        num_samples = targets.size(0)
        total_loss += loss.item() * num_samples
        total_samples += num_samples

    average_loss = total_loss / total_samples

    return average_loss


def evaluate(
        model,
        data_loader,
        loss_function,
        device
):
    model.eval()

    total_loss = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            targets = batch.y.float().view(-1)

            predictions = model(batch)
            loss = loss_function(predictions, targets)

            num_samples = targets.size(0)

            total_loss += loss.item() * num_samples

            total_absolute_error += (
                torch.abs(
                    predictions - targets
                ).sum().item()
            )

            total_samples += num_samples

    average_loss = total_loss / total_samples

    mean_absolute_error = (
        total_absolute_error / total_samples
    )

    return average_loss, mean_absolute_error


def collect_predictions(
        model,
        data_loader,
        device
):
    model.eval()

    rows = []
    sample_number = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            targets = batch.y.float().view(-1)
            predictions = model(batch)

            raw_ids = getattr(batch, "structure_id", None)
            structure_ids = []

            def flatten_ids(value):
                if isinstance(value, str):
                    structure_ids.append(value)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        flatten_ids(item)

            flatten_ids(raw_ids)

            if len(structure_ids) != targets.size(0):
                structure_ids = [
                    f"test_sample_{sample_number + i}"
                    for i in range(targets.size(0))
                ]

            for structure_id, target, prediction in zip(
                structure_ids,
                targets.cpu().tolist(),
                predictions.cpu().tolist()
            ):
                error = prediction - target

                rows.append(
                    {
                        "structure_id": structure_id,
                        "true_value": target,
                        "predicted_value": prediction,
                        "error": error,
                        "absolute_error": abs(error),
                    }
                )

            sample_number += targets.size(0)

    return rows


scripts_dir = os.path.dirname(
    os.path.abspath(__file__)
)

project_dir = os.path.dirname(scripts_dir)

graph_dir = os.path.join(
    project_dir,
    "graph_data_20000"
)

checkpoint_dir = os.path.join(
    project_dir,
    "checkpoint"
)

os.makedirs(
    checkpoint_dir,
    exist_ok=True
)

checkpoint_path = os.path.join(
    checkpoint_dir,
    "best_model_20000.pt"
)


results_dir = os.path.join(
    project_dir,
    "results_20000"
)

os.makedirs(
    results_dir,
    exist_ok=True
)

history_path = os.path.join(
    results_dir,
    "training_history.csv"
)

predictions_path = os.path.join(
    results_dir,
    "test_predictions.csv"
)

metrics_path = os.path.join(
    results_dir,
    "test_metrics.txt"
)


torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


dataset = build_graph_list(
    graph_dir=graph_dir
)


train_loader, val_loader, test_loader = create_data_loaders(
    dataset=dataset,
    batch_size=32,
    train_ratio=0.8,
    val_ratio=0.1,
    random_seed=42,
    num_workers=0
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("图数据目录：", graph_dir)
print("计算设备：", device)
print("全部样本数：", len(dataset))
print("训练集样本数：", len(train_loader.dataset))
print("验证集样本数：", len(val_loader.dataset))
print("测试集样本数：", len(test_loader.dataset))


model = CrystalGNN(
    node_input_dim=100,
    edge_dim=21,
    hidden_dim=64
)

model = model.to(device)


loss_function = nn.MSELoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


num_epochs = 100

best_val_loss = float("inf")

training_history = []


for epoch in range(
    1,
    num_epochs + 1
):
    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        loss_function,
        device
    )

    val_loss, val_mae = evaluate(
        model,
        val_loader,
        loss_function,
        device
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

    print(
        f"Epoch {epoch:03d} | "
        f"Train MSE: {train_loss:.6f} | "
        f"Val MSE: {val_loss:.6f} | "
        f"Val MAE: {val_mae:.6f}"
    )

    training_history.append(
        {
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "val_mae": val_mae,
        }
    )



with open(
    history_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "train_mse",
            "val_mse",
            "val_mae",
        ]
    )

    writer.writeheader()
    writer.writerows(training_history)


model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)


test_loss, test_mae = evaluate(
    model,
    test_loader,
    loss_function,
    device
)


prediction_rows = collect_predictions(
    model,
    test_loader,
    device
)

with open(
    predictions_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "structure_id",
            "true_value",
            "predicted_value",
            "error",
            "absolute_error",
        ]
    )

    writer.writeheader()
    writer.writerows(prediction_rows)


true_values = [
    row["true_value"]
    for row in prediction_rows
]

predicted_values = [
    row["predicted_value"]
    for row in prediction_rows
]

test_rmse = math.sqrt(test_loss)

true_mean = sum(true_values) / len(true_values)

ss_res = sum(
    (prediction - target) ** 2
    for target, prediction in zip(
        true_values,
        predicted_values
    )
)

ss_tot = sum(
    (target - true_mean) ** 2
    for target in true_values
)

test_r2 = (
    1.0 - ss_res / ss_tot
    if ss_tot > 0
    else float("nan")
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:
    file.write(f"best_val_mse: {best_val_loss}\n")
    file.write(f"test_mse: {test_loss}\n")
    file.write(f"test_rmse: {test_rmse}\n")
    file.write(f"test_mae: {test_mae}\n")
    file.write(f"test_r2: {test_r2}\n")


print("\n训练完成")
print("最佳验证集 MSE：", best_val_loss)
print("测试集 MSE：", test_loss)
print("测试集 MAE：", test_mae)
print("最佳模型路径：", checkpoint_path)
print("训练历史路径：", history_path)
print("测试预测路径：", predictions_path)
print("测试指标路径：", metrics_path)
print("测试集 RMSE：", test_rmse)
print("测试集 R²：", test_r2)