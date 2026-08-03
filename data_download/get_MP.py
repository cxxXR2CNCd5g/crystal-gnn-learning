import csv
import os
import shutil
import time

from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter


# ============================================================
# 1. Materials Project API Key
# ============================================================

# 直接在代码中使用 API Key，不再读取环境变量
API_KEY = "Jwrmx3fSCtg7cnwonHcpBmXWhSC39T48"


# ============================================================
# 2. 文件路径
# ============================================================

# 当前脚本所在目录：
# /root/autodl-tmp/MatDeepLearn/learn_gnn/data_download
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Materials Project ID 文件
ID_FILE = os.path.join(
    SCRIPT_DIR,
    "mp-ids-46744.csv"
)

# 最终数据集目录：
# /root/autodl-tmp/MatDeepLearn/learn_gnn/bulk_data_demo180
FINAL_DIR = os.path.normpath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "bulk_data_demo180"
    )
)

# 临时下载目录
# 下载成功满 20000 个后，再用它覆盖原来的数据集
TEMP_DIR = FINAL_DIR + "_tmp"


# ============================================================
# 3. 下载参数
# ============================================================

# 最终需要成功保存的材料数量
TARGET_COUNT = 20000

# 每次向 Materials Project 请求的 ID 数量
# 200 比较稳妥，避免一次请求过大
BATCH_SIZE = 200

# 每一批请求失败后的最大重试次数
MAX_RETRIES = 5

# 请求失败后等待时间，单位：秒
RETRY_WAIT_SECONDS = 10


# ============================================================
# 4. 检查路径和 API Key
# ============================================================

if not API_KEY.strip():
    raise RuntimeError("API_KEY 为空，请检查代码中的 API_KEY。")

if not os.path.exists(ID_FILE):
    raise FileNotFoundError(
        f"没有找到 Materials Project ID 文件：\n{ID_FILE}"
    )

print("=" * 70)
print("脚本目录：", SCRIPT_DIR)
print("ID 文件：", ID_FILE)
print("最终数据目录：", FINAL_DIR)
print("临时数据目录：", TEMP_DIR)
print("目标下载数量：", TARGET_COUNT)
print("每批查询数量：", BATCH_SIZE)
print("=" * 70)


# ============================================================
# 5. 读取 mp-ids-46744.csv
# ============================================================

material_ids = []
seen_ids = set()

with open(ID_FILE, "r", encoding="utf-8-sig") as file:
    reader = csv.reader(file)

    for row in reader:
        # 跳过空行
        if not row:
            continue

        material_id = row[0].strip()

        # 跳过空字符串
        if not material_id:
            continue

        # 跳过可能存在的表头
        if material_id.lower() in {
            "material_id",
            "materials_id",
            "mp_id",
            "id",
        }:
            continue

        # 只保留类似 mp-149 这样的 ID
        if not material_id.startswith("mp-"):
            print("跳过无法识别的内容：", material_id)
            continue

        # 去除重复 ID
        if material_id not in seen_ids:
            material_ids.append(material_id)
            seen_ids.add(material_id)


print("读取到的唯一 Materials Project ID 数量：", len(material_ids))

if len(material_ids) < TARGET_COUNT:
    raise RuntimeError(
        f"ID 文件中只有 {len(material_ids)} 个有效且不重复的 ID，"
        f"不足以下载 {TARGET_COUNT} 个材料。"
    )


# ============================================================
# 6. 创建临时下载目录
# ============================================================

# 如果上一次下载留下了临时目录，先删除
if os.path.exists(TEMP_DIR):
    print("发现旧的临时目录，正在删除：", TEMP_DIR)
    shutil.rmtree(TEMP_DIR)

os.makedirs(TEMP_DIR, exist_ok=True)

targets_path = os.path.join(TEMP_DIR, "targets.csv")
mapping_path = os.path.join(TEMP_DIR, "id_mapping.csv")
failed_path = os.path.join(TEMP_DIR, "failed_ids.csv")


# ============================================================
# 7. 分批查询并保存 CIF
# ============================================================

success_count = 0
checked_count = 0
failed_count = 0

with (
    open(
        targets_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as targets_file,
    open(
        mapping_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as mapping_file,
    open(
        failed_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as failed_file,
    MPRester(API_KEY) as mpr,
):
    targets_writer = csv.writer(targets_file)
    mapping_writer = csv.writer(mapping_file)
    failed_writer = csv.writer(failed_file)

    # targets.csv 不写表头
    # MatDeepLearn 需要：
    # 0,目标值
    # 1,目标值
    # 2,目标值

    # 映射表写表头，方便以后找到编号对应的 MP-ID
    mapping_writer.writerow(
        [
            "structure_id",
            "material_id",
            "formation_energy_per_atom",
        ]
    )

    failed_writer.writerow(
        [
            "material_id",
            "reason",
        ]
    )

    total_batches = (
        len(material_ids) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for batch_start in range(
        0,
        len(material_ids),
        BATCH_SIZE
    ):
        # 成功下载满 20000 个后停止
        if success_count >= TARGET_COUNT:
            break

        batch_ids = material_ids[
            batch_start:batch_start + BATCH_SIZE
        ]

        batch_number = batch_start // BATCH_SIZE + 1

        print()
        print("-" * 70)
        print(
            f"开始处理第 {batch_number}/{total_batches} 批，"
            f"本批包含 {len(batch_ids)} 个 ID"
        )

        docs = None
        last_error = None

        # 请求失败时自动重试
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                docs = mpr.materials.summary.search(
                    material_ids=batch_ids,
                    fields=[
                        "material_id",
                        "structure",
                        "formation_energy_per_atom",
                    ],
                    all_fields=False,
                    chunk_size=BATCH_SIZE,
                )

                print(
                    f"第 {batch_number} 批请求成功，"
                    f"API 返回 {len(docs)} 个材料"
                )

                break

            except Exception as error:
                last_error = error

                print(
                    f"第 {batch_number} 批请求失败，"
                    f"第 {attempt}/{MAX_RETRIES} 次尝试："
                )
                print(error)

                if attempt < MAX_RETRIES:
                    print(
                        f"{RETRY_WAIT_SECONDS} 秒后重新请求……"
                    )
                    time.sleep(RETRY_WAIT_SECONDS)

        # 整批请求失败
        if docs is None:
            print(
                f"第 {batch_number} 批连续失败 "
                f"{MAX_RETRIES} 次，跳过这一批。"
            )

            for material_id in batch_ids:
                failed_writer.writerow(
                    [
                        material_id,
                        f"batch query failed: {last_error}",
                    ]
                )
                failed_count += 1

            failed_file.flush()
            continue

        # 将返回结果按照 material_id 保存成字典
        # 因为 API 返回顺序不一定与输入顺序一致
        docs_by_id = {}

        for doc in docs:
            docs_by_id[str(doc.material_id)] = doc

        # 按原始 ID 顺序处理
        for material_id in batch_ids:
            if success_count >= TARGET_COUNT:
                break

            checked_count += 1

            doc = docs_by_id.get(material_id)

            # API 没有返回这个 ID
            if doc is None:
                failed_writer.writerow(
                    [
                        material_id,
                        "API did not return this material",
                    ]
                )
                failed_count += 1
                continue

            structure = doc.structure
            target = doc.formation_energy_per_atom

            # 检查结构
            if structure is None:
                failed_writer.writerow(
                    [
                        material_id,
                        "structure is None",
                    ]
                )
                failed_count += 1
                continue

            # 检查形成能
            if target is None:
                failed_writer.writerow(
                    [
                        material_id,
                        "formation_energy_per_atom is None",
                    ]
                )
                failed_count += 1
                continue

            # 只按照成功下载的顺序编号
            # 最终得到 0.cif 到 19999.cif
            structure_id = success_count

            cif_filename = f"{structure_id}.cif"
            cif_path = os.path.join(
                TEMP_DIR,
                cif_filename
            )

            try:
                CifWriter(structure).write_file(cif_path)

            except Exception as error:
                failed_writer.writerow(
                    [
                        material_id,
                        f"CIF write failed: {error}",
                    ]
                )
                failed_count += 1
                continue

            # 写入 MatDeepLearn 使用的 targets.csv
            targets_writer.writerow(
                [
                    structure_id,
                    float(target),
                ]
            )

            # 保存编号、MP-ID 和目标值之间的对应关系
            mapping_writer.writerow(
                [
                    structure_id,
                    material_id,
                    float(target),
                ]
            )

            success_count += 1

        # 每处理完一批，立即将内容写入硬盘
        targets_file.flush()
        mapping_file.flush()
        failed_file.flush()

        print(
            f"当前进度：成功 {success_count}/{TARGET_COUNT}，"
            f"已检查 {checked_count} 个 ID，"
            f"失败 {failed_count} 个"
        )


# ============================================================
# 8. 检查下载结果
# ============================================================

print()
print("=" * 70)
print("下载阶段结束")
print("成功保存数量：", success_count)
print("失败数量：", failed_count)
print("目标数量：", TARGET_COUNT)
print("=" * 70)

if success_count < TARGET_COUNT:
    raise RuntimeError(
        f"只成功下载了 {success_count} 个材料，"
        f"未达到目标数量 {TARGET_COUNT}。\n"
        f"原来的 bulk_data_demo180 不会被删除。\n"
        f"当前已下载的数据保存在：\n{TEMP_DIR}"
    )


# ============================================================
# 9. 成功下载满 20000 个后覆盖旧数据集
# ============================================================

if os.path.exists(FINAL_DIR):
    print("正在删除原来的数据集目录：")
    print(FINAL_DIR)
    shutil.rmtree(FINAL_DIR)

# 将临时目录改名为正式目录
os.rename(TEMP_DIR, FINAL_DIR)

print()
print("=" * 70)
print("20000 个材料下载完成")
print("最终数据目录：", FINAL_DIR)
print("CIF 文件范围：0.cif 到 19999.cif")
print("目标文件：", os.path.join(FINAL_DIR, "targets.csv"))
print("映射文件：", os.path.join(FINAL_DIR, "id_mapping.csv"))
print("失败记录：", os.path.join(FINAL_DIR, "failed_ids.csv"))
print("=" * 70)