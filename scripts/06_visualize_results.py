import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. 路径
# ============================================================

scripts_dir = os.path.dirname(
    os.path.abspath(__file__)
)

project_dir = os.path.dirname(scripts_dir)

results_dir = os.path.join(
    project_dir,
    "results_20000"
)

history_path = os.path.join(
    results_dir,
    "training_history.csv"
)

predictions_path = os.path.join(
    results_dir,
    "test_predictions.csv"
)

figure_dir = os.path.join(
    results_dir,
    "figures"
)

os.makedirs(
    figure_dir,
    exist_ok=True
)


# ============================================================
# 2. 检查结果文件
# ============================================================

if not os.path.exists(history_path):
    raise FileNotFoundError(
        f"找不到训练历史文件：{history_path}"
    )

if not os.path.exists(predictions_path):
    raise FileNotFoundError(
        f"找不到测试集预测文件：{predictions_path}"
    )


# ============================================================
# 3. 读取数据
# ============================================================

history_df = pd.read_csv(history_path)
prediction_df = pd.read_csv(predictions_path)

true_values = prediction_df["true_value"].to_numpy()
predicted_values = prediction_df["predicted_value"].to_numpy()
errors = prediction_df["error"].to_numpy()
absolute_errors = prediction_df["absolute_error"].to_numpy()


# ============================================================
# 4. 计算测试集指标
# ============================================================

mse = np.mean(
    (predicted_values - true_values) ** 2
)

rmse = np.sqrt(mse)

mae = np.mean(
    np.abs(predicted_values - true_values)
)

ss_res = np.sum(
    (predicted_values - true_values) ** 2
)

ss_tot = np.sum(
    (true_values - np.mean(true_values)) ** 2
)

r2 = (
    1.0 - ss_res / ss_tot
    if ss_tot > 0
    else np.nan
)


# ============================================================
# 5. 训练曲线
# ============================================================

plt.figure(figsize=(7.0, 5.2))

plt.plot(
    history_df["epoch"],
    history_df["train_mse"],
    label="Train MSE"
)

plt.plot(
    history_df["epoch"],
    history_df["val_mse"],
    label="Validation MSE"
)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training and validation loss")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

training_curve_path = os.path.join(
    figure_dir,
    "01_training_curve.png"
)

plt.savefig(
    training_curve_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. 真实值与预测值散点图
# ============================================================

value_min = min(
    true_values.min(),
    predicted_values.min()
)

value_max = max(
    true_values.max(),
    predicted_values.max()
)

margin = 0.05 * (
    value_max - value_min
)

plot_min = value_min - margin
plot_max = value_max + margin


plt.figure(figsize=(6.2, 6.0))

plt.scatter(
    true_values,
    predicted_values,
    s=18,
    alpha=0.55,
    edgecolors="none"
)

plt.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--",
    linewidth=1.5,
    label="Ideal prediction"
)

plt.xlim(plot_min, plot_max)
plt.ylim(plot_min, plot_max)

plt.xlabel("DFT formation energy (eV/atom)")
plt.ylabel("Predicted formation energy (eV/atom)")
plt.title("Test set prediction")

metric_text = (
    f"$R^2$ = {r2:.3f}\n"
    f"MAE = {mae:.3f} eV/atom\n"
    f"RMSE = {rmse:.3f} eV/atom"
)

plt.text(
    0.05,
    0.95,
    metric_text,
    transform=plt.gca().transAxes,
    va="top",
    ha="left",
    bbox={
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.85
    }
)

plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

parity_plot_path = os.path.join(
    figure_dir,
    "02_test_parity_plot.png"
)

plt.savefig(
    parity_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. 绝对误差分布
# ============================================================

plt.figure(figsize=(7.0, 5.2))

plt.hist(
    absolute_errors,
    bins=40,
    edgecolor="black",
    linewidth=0.5
)

plt.axvline(
    mae,
    linestyle="--",
    linewidth=1.5,
    label=f"MAE = {mae:.3f} eV/atom"
)

plt.xlabel("Absolute error (eV/atom)")
plt.ylabel("Count")
plt.title("Absolute error distribution")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

error_histogram_path = os.path.join(
    figure_dir,
    "03_absolute_error_distribution.png"
)

plt.savefig(
    error_histogram_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. 残差图
# ============================================================

plt.figure(figsize=(7.0, 5.2))

plt.scatter(
    true_values,
    errors,
    s=18,
    alpha=0.55,
    edgecolors="none"
)

plt.axhline(
    0.0,
    linestyle="--",
    linewidth=1.5
)

plt.xlabel("DFT formation energy (eV/atom)")
plt.ylabel("Prediction error (eV/atom)")
plt.title("Residual plot")
plt.grid(alpha=0.25)
plt.tight_layout()

residual_plot_path = os.path.join(
    figure_dir,
    "04_residual_plot.png"
)

plt.savefig(
    residual_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. 输出
# ============================================================

print("绘图完成")
print("训练曲线：", training_curve_path)
print("预测散点图：", parity_plot_path)
print("绝对误差分布：", error_histogram_path)
print("残差图：", residual_plot_path)

print("\n测试集指标")
print("MSE：", mse)
print("RMSE：", rmse)
print("MAE：", mae)
print("R²：", r2)