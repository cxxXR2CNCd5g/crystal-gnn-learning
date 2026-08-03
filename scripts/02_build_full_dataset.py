import csv
import json
import os
import time
from datetime import datetime

import torch

from crystal_gnn.graph_builder import cif_to_graph


# ============================================================
# 1. 路径设置
# ============================================================

scripts_dir = os.path.dirname(
    os.path.abspath(__file__)
)

project_dir = os.path.dirname(scripts_dir)

# 原始数据目录：包含 0.cif、1.cif、... 和 targets.csv
# 目前仍沿用原来的目录名，但其中可以放 20000 个样本。
data_dir = os.path.join(
    project_dir,
    "bulk_data_demo180"
)

# 预处理后的晶体图保存目录
graph_dir = os.path.join(
    project_dir,
    "graph_data_20000"
)

# 标签文件
target_path = os.path.join(
    data_dir,
    "targets.csv"
)

# 成功图索引与失败记录
index_path = os.path.join(
    graph_dir,
    "graph_index.csv"
)

failed_path = os.path.join(
    graph_dir,
    "failed_graphs.csv"
)

metadata_path = os.path.join(
    graph_dir,
    "metadata.json"
)


# ============================================================
# 2. 构图参数
# ============================================================

# 与当前 graph_builder.py 的默认参数保持一致
CUT_OFF = 4.0
DISTANCE_STEP = 0.2
GAUSSIAN_WIDTH = 0.2

# 每处理多少个样本打印一次进度
PRINT_EVERY = 100

# False：已有 .pt 文件就跳过，支持断点续跑
# True：重新生成并覆盖已有 .pt 文件
OVERWRITE_EXISTING = False


# ============================================================
# 3. 检查输入文件
# ============================================================

if not os.path.isdir(data_dir):
    raise FileNotFoundError(
        f"找不到原始数据目录：{data_dir}"
    )

if not os.path.isfile(target_path):
    raise FileNotFoundError(
        f"找不到标签文件：{target_path}"
    )

os.makedirs(graph_dir, exist_ok=True)


# ============================================================
# 4. 读取 targets.csv
# ============================================================

samples = []

with open(
    target_path,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:
    reader = csv.reader(file)

    for line_number, row in enumerate(reader, start=1):
        if not row:
            continue

        if len(row) < 2:
            raise ValueError(
                f"targets.csv 第 {line_number} 行列数不足：{row}"
            )

        structure_id = row[0].strip()
        target_text = row[1].strip()

        if not structure_id:
            continue

        try:
            target = float(target_text)
        except ValueError as error:
            raise ValueError(
                f"targets.csv 第 {line_number} 行目标值无法转换为浮点数："
                f"{target_text}"
            ) from error

        samples.append((structure_id, target))


num_samples = len(samples)

if num_samples == 0:
    raise RuntimeError(
        f"{target_path} 中没有可处理的样本。"
    )


print("=" * 70, flush=True)
print("开始构建完整晶体图数据集", flush=True)
print("原始数据目录：", data_dir, flush=True)
print("图数据目录：", graph_dir, flush=True)
print("样本总数：", num_samples, flush=True)
print("截断半径：", CUT_OFF, "Å", flush=True)
print("距离步长：", DISTANCE_STEP, flush=True)
print("高斯宽度：", GAUSSIAN_WIDTH, flush=True)
print("覆盖已有图：", OVERWRITE_EXISTING, flush=True)
print("=" * 70, flush=True)


# ============================================================
# 5. 逐个 CIF 构图并保存为独立 .pt 文件
# ============================================================

success_count = 0
skipped_count = 0
failed_count = 0

successful_rows = []
failed_rows = []

start_time = time.time()

for sample_number, (structure_id, target) in enumerate(
    samples,
    start=1
):
    cif_path = os.path.join(
        data_dir,
        f"{structure_id}.cif"
    )

    graph_filename = f"{structure_id}.pt"
    graph_path = os.path.join(
        graph_dir,
        graph_filename
    )

    # 已存在的图不重复构建，用于断点续跑
    if (
        not OVERWRITE_EXISTING
        and os.path.isfile(graph_path)
        and os.path.getsize(graph_path) > 0
    ):
        skipped_count += 1
        successful_rows.append(
            [structure_id, target, graph_filename, "existing"]
        )
    else:
        try:
            data = cif_to_graph(
                cif_path=cif_path,
                target=target,
                cut_off=CUT_OFF,
                distance_step=DISTANCE_STEP,
                gaussian_width=GAUSSIAN_WIDTH
            )

            # 先写临时文件，再原子替换，避免中断后留下不完整 .pt
            temporary_path = graph_path + ".tmp"
            torch.save(data, temporary_path)
            os.replace(temporary_path, graph_path)

            success_count += 1
            successful_rows.append(
                [structure_id, target, graph_filename, "created"]
            )

        except Exception as error:
            failed_count += 1
            failed_rows.append(
                [structure_id, target, type(error).__name__, str(error)]
            )

            # 清理可能残留的临时文件
            temporary_path = graph_path + ".tmp"
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

            print(
                f"处理失败：{structure_id}.cif | {error}",
                flush=True
            )

    if (
        sample_number == 1
        or sample_number % PRINT_EVERY == 0
        or sample_number == num_samples
    ):
        elapsed_seconds = time.time() - start_time
        speed = sample_number / elapsed_seconds if elapsed_seconds > 0 else 0.0
        remaining_samples = num_samples - sample_number
        remaining_seconds = (
            remaining_samples / speed
            if speed > 0
            else 0.0
        )

        print(
            f"进度 {sample_number}/{num_samples} | "
            f"新建 {success_count} | "
            f"跳过 {skipped_count} | "
            f"失败 {failed_count} | "
            f"速度 {speed:.2f} 个/秒 | "
            f"预计剩余 {remaining_seconds / 60:.1f} 分钟",
            flush=True
        )


# ============================================================
# 6. 保存索引、失败记录和元数据
# ============================================================

with open(
    index_path,
    "w",
    encoding="utf-8",
    newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow(
        ["structure_id", "target", "graph_file", "status"]
    )
    writer.writerows(successful_rows)


with open(
    failed_path,
    "w",
    encoding="utf-8",
    newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow(
        ["structure_id", "target", "error_type", "error_message"]
    )
    writer.writerows(failed_rows)


elapsed_seconds = time.time() - start_time

total_available = success_count + skipped_count

metadata = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "source_data_dir": os.path.abspath(data_dir),
    "graph_data_dir": os.path.abspath(graph_dir),
    "target_file": os.path.abspath(target_path),
    "total_samples_in_targets": num_samples,
    "newly_created_graphs": success_count,
    "existing_graphs_skipped": skipped_count,
    "total_available_graphs": total_available,
    "failed_graphs": failed_count,
    "cut_off": CUT_OFF,
    "distance_step": DISTANCE_STEP,
    "gaussian_width": GAUSSIAN_WIDTH,
    "edge_feature_dimension": int(
        torch.arange(
            0.0,
            CUT_OFF + DISTANCE_STEP,
            GAUSSIAN_WIDTH
        ).numel()
    ),
    "elapsed_seconds": elapsed_seconds
}

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        metadata,
        file,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 7. 最终结果
# ============================================================

print("\n" + "=" * 70, flush=True)
print("晶体图预处理完成", flush=True)
print("targets.csv 样本数：", num_samples, flush=True)
print("本次新建图数量：", success_count, flush=True)
print("已有并跳过数量：", skipped_count, flush=True)
print("可用图总数：", total_available, flush=True)
print("失败数量：", failed_count, flush=True)
print("总耗时：", f"{elapsed_seconds / 60:.2f} 分钟", flush=True)
print("图数据目录：", graph_dir, flush=True)
print("图索引文件：", index_path, flush=True)
print("失败记录文件：", failed_path, flush=True)
print("元数据文件：", metadata_path, flush=True)
print("=" * 70, flush=True)

if failed_count > 0:
    print(
        "存在构图失败的样本，请检查 failed_graphs.csv。",
        flush=True
    )