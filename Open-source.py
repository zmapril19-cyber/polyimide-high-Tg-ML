import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# 固定所有的随机种子以保证结果可复现 (上一轮增加的内容)
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.titleweight': 'bold',
    'axes.labelsize': 24,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'axes.formatter.limits': (-3, 3)
})

# 1. 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 数据加载
descriptor_names = pd.read_csv("1.csv", header=None)
feature_mapping = dict(zip(descriptor_names.iloc[0], descriptor_names.iloc[1]))

df = pd.read_csv("2.csv")

descriptor_columns = [col for col in df.columns if '描述符' in col]
target_column = [col for col in df.columns if 'Tg' in col][0]

df = df.dropna(subset=descriptor_columns + [target_column])
X = df[descriptor_columns].values
y = df[target_column].values
feature_names = [feature_mapping.get(col, col) for col in descriptor_columns]

# 顺序划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# 转为张量（不需要unsqueeze(1)的维度变换）
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)  # (B, D)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# 数据加载器
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 3. MLP 模型构建（与MLP.py一致）
class MLPRegressor(nn.Module):
    def __init__(self, input_size):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        return self.output(x)


# 初始化模型
input_size = X_train.shape[1]
model = MLPRegressor(input_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)

# 4. 训练模型
num_epochs = 500
train_losses = []
test_losses = []
test_r2_scores = []
train_mae_curve = []
test_mae_curve = []

# 记录最佳模型状态
best_epoch = 0
best_train_r2 = -np.inf
best_test_r2 = -np.inf
best_train_rmse = np.inf
best_test_rmse = np.inf
best_train_mae = np.inf
best_test_mae = np.inf
best_model_state = None
best_train_predictions = None
best_test_predictions = None
best_y_train_actual = None
best_y_test_actual = None

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    total_samples = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    avg_train_loss = epoch_train_loss / total_samples
    train_losses.append(avg_train_loss)

    # 验证阶段
    model.eval()
    with torch.no_grad():
        epoch_test_loss = 0
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            epoch_test_loss += loss.item() * batch_x.size(0)
        test_losses.append(epoch_test_loss / len(test_loader.dataset))

        train_predictions = model(X_train)
        test_predictions = model(X_test)

        # 反标准化
        train_predictions_np = scaler_y.inverse_transform(train_predictions.cpu().numpy())
        test_predictions_np = scaler_y.inverse_transform(test_predictions.cpu().numpy())
        y_train_actual = scaler_y.inverse_transform(y_train.cpu().numpy().reshape(-1, 1))
        y_test_actual = scaler_y.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

        # 计算各种指标
        train_r2 = r2_score(y_train_actual, train_predictions_np)
        test_r2 = r2_score(y_test_actual, test_predictions_np)
        test_r2_scores.append(test_r2)
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions_np))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions_np))
        train_mae = mean_absolute_error(y_train_actual, train_predictions_np)
        test_mae = mean_absolute_error(y_test_actual, test_predictions_np)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, "
              f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, "
              f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}, "
              f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")

        train_mae_curve.append(train_mae)
        test_mae_curve.append(test_mae)

        if test_r2 > best_test_r2:
            best_epoch = epoch
            best_test_r2 = test_r2
            best_train_r2 = train_r2
            best_test_rmse = test_rmse
            best_train_rmse = train_rmse
            best_test_mae = test_mae
            best_train_mae = train_mae
            best_model_state = model.state_dict()
            best_train_predictions = train_predictions_np
            best_test_predictions = test_predictions_np
            best_y_train_actual = y_train_actual
            best_y_test_actual = y_test_actual

# 5. 加载最佳模型
model.load_state_dict(best_model_state)
model.eval()

# 6. 打印最终结果
print("\n" + "=" * 80)
print("最佳模型性能汇总:")
print("=" * 80)
print(f"最佳轮次: {best_epoch + 1}")
print(f"训练集 R²: {best_train_r2:.6f}")
print(f"测试集 R²: {best_test_r2:.6f}")
print(f"训练集 RMSE: {best_train_rmse:.6f}")
print(f"测试集 RMSE: {best_test_rmse:.6f}")
print(f"训练集 MAE: {best_train_mae:.6f}")
print(f"测试集 MAE: {best_test_mae:.6f}")
print("=" * 80)

# ------------------ 主要修改的位置 (第7和第7.5部分) ------------------
# 计算全局的最小值和最大值，强制左右两图坐标轴一致
global_min = min(best_y_train_actual.min(), best_train_predictions.min(),
                 best_y_test_actual.min(), best_test_predictions.min())
global_max = max(best_y_train_actual.max(), best_train_predictions.max(),
                 best_y_test_actual.max(), best_test_predictions.max())

# 加一点留白边界让点不至于贴边
padding = (global_max - global_min) * 0.05
axis_min = global_min - padding
axis_max = global_max + padding

# 7. 拟合图
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 18

plt.figure(figsize=(16, 6))

# 训练集
plt.subplot(1, 2, 1)
plt.scatter(best_y_train_actual, best_train_predictions, color='blue', alpha=0.6, s=50)
plt.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', linewidth=2)
plt.xlim(axis_min, axis_max)  # 固定 x 轴范围
plt.ylim(axis_min, axis_max)  # 固定 y 轴范围，保证左右高度一致
plt.xlabel("Actual T$_{g}$ (K)", fontweight='bold', fontsize=22)
plt.ylabel("Predicted T$_{g}$ (K)", fontweight='bold', fontsize=22)
plt.title("Train Fit Model", fontweight='bold', fontsize=22)

# 测试集
plt.subplot(1, 2, 2)
plt.scatter(best_y_test_actual, best_test_predictions, color='green', alpha=0.6, s=50)
plt.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', linewidth=2)
plt.xlim(axis_min, axis_max)  # 固定 x 轴范围
plt.ylim(axis_min, axis_max)  # 固定 y 轴范围，保证左右高度一致
plt.xlabel("Actual T$_{g}$ (K)", fontweight='bold', fontsize=22)
plt.ylabel("Predicted T$_{g}$ (K)", fontweight='bold', fontsize=22)
plt.title("Test Fit Model", fontweight='bold', fontsize=22)

plt.tight_layout()
plt.show()

# 7.5 预测结果图
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(best_y_train_actual, best_train_predictions, color='blue', alpha=0.6, s=80)
plt.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', linewidth=3)
plt.xlim(axis_min, axis_max)  # 固定 x 轴范围
plt.ylim(axis_min, axis_max)  # 固定 y 轴范围，保证左右高度一致
plt.xlabel("Actual T$_{g}$ (K)", fontweight='bold', fontsize=22)
plt.ylabel("Predicted T$_{g}$ (K)", fontweight='bold', fontsize=22)
plt.title(f"Train Set (R²={best_train_r2:.3f})", fontweight='bold', fontsize=22)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(best_y_test_actual, best_test_predictions, color='green', alpha=0.6, s=80)
plt.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', linewidth=3)
plt.xlim(axis_min, axis_max)  # 固定 x 轴范围
plt.ylim(axis_min, axis_max)  # 固定 y 轴范围，保证左右高度一致
plt.xlabel("Actual T$_{g}$ (K)", fontweight='bold', fontsize=22)
plt.ylabel("Predicted T$_{g}$ (K)", fontweight='bold', fontsize=22)
plt.title(f"Test Set (R²={best_test_r2:.3f})", fontweight='bold', fontsize=22)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# ----------------------------------------------------------------------


# 8. 训练过程图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', linewidth=2)
plt.xlabel('Epoch', fontweight='bold', fontsize=24)
plt.ylabel('Loss', fontweight='bold', fontsize=24)
plt.title('Training Loss over Epochs', fontweight='bold', fontsize=24)
plt.xticks(fontsize=24, fontweight='bold', fontname='Times New Roman')
plt.yticks(fontsize=24, fontweight='bold', fontname='Times New Roman')
plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 24})
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(test_losses, label='Test Loss', linewidth=2, color='orange')
plt.xlabel('Epoch', fontweight='bold', fontsize=24)
plt.ylabel('Loss', fontweight='bold', fontsize=24)
plt.title('Test Loss over Epochs', fontweight='bold', fontsize=24)
plt.xticks(fontsize=24, fontweight='bold', fontname='Times New Roman')
plt.yticks(fontsize=24, fontweight='bold', fontname='Times New Roman')
plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 24})
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# 9. SHAP可视化
def mlp_predict(X):
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy().flatten()
    return predictions


print("Calculating SHAP values...")

X_test_np = X_test.cpu().numpy()
n_background_samples = min(100, len(X_train.cpu().numpy()))
if n_background_samples > 10:
    np.random.seed(42)
    background_indices = np.random.choice(len(X_train.cpu().numpy()), n_background_samples, replace=False)
    background_data = X_train.cpu().numpy()[background_indices]

    explainer = shap.KernelExplainer(
        model=mlp_predict,
        data=background_data,
        link="identity"
    )

    n_shap_samples = min(100, len(X_test_np))
    shap_values = explainer.shap_values(X_test_np[:n_shap_samples])

    test_df = pd.DataFrame(X_test_np[:n_shap_samples], columns=feature_names)

    # 使用 rc_context 局部改变 SHAP 图的字体大小参数，防止 y 轴特征名称拥挤
    with plt.rc_context({
        'font.size': 14,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.titlesize': 24
    }):
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values=shap_values,
            features=test_df,
            feature_names=feature_names,
            plot_type="dot",
            show=False,
            max_display=min(20, len(feature_names))
        )

        fig = plt.gcf()
        axes = fig.get_axes()
        main_ax = axes[0]

        main_ax.set_xlabel(
            "SHAP Value (Impact On Model Output)",
            fontsize=18,
            fontweight='bold',
            fontfamily='Times New Roman'
        )

        if len(axes) > 1:
            cb_ax = axes[-1]
            cb_ax.set_ylabel(
                "Feature Value",
                fontsize=16,
                fontweight='bold',
                fontfamily='Times New Roman'
            )
            cb_ax.tick_params(
                axis='y',
                labelsize=14,
                width=2,
                length=5,
                direction='in'
            )

        # 将所有字体统一为 Times New Roman 加粗，但不再强制覆盖其大小以保持合适的比例
        for text_obj in fig.findobj(match=plt.Text):
            text_obj.set_fontname('Times New Roman')
            text_obj.set_fontweight('bold')

        plt.title("SHAP Feature Importance - MLP", pad=20, fontsize=24, fontweight='bold', fontfamily='Times New Roman')
        plt.tight_layout()
        plt.show()

    # 特征重要性分析
    if shap_values is not None:
        shap_importance = np.mean(np.abs(shap_values), axis=0)

        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(shap_importance)],
            'SHAP_Importance': shap_importance
        }).sort_values('SHAP_Importance', ascending=False)

        plt.figure(figsize=(12, 8))
        plt.barh(range(min(15, len(importance_df))),
                 importance_df['SHAP_Importance'].head(15),
                 color='steelblue')
        plt.yticks(range(min(15, len(importance_df))),
                   importance_df['Feature'].head(15))
        plt.xlabel('Mean |SHAP value|', fontweight='bold', fontsize=22)
        plt.title('Top 15 Feature Importance (MLP)', fontweight='bold', fontsize=22)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()

        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))

# 10. RMSE vs Epoch 图
train_rmse_curve = np.sqrt(np.array(train_losses)) * scaler_y.scale_[0]

test_r2_arr = np.array(test_r2_scores)
y_test_std = np.std(y_test.cpu().numpy())
val_rmse_curve = y_test_std * np.sqrt(np.maximum(0, (1 - test_r2_arr)))

plt.figure(figsize=(12, 8))

plt.plot(range(1, len(train_rmse_curve) + 1), train_rmse_curve,
         label='Training Error', color='#8DB48E', linestyle='--', linewidth=3)
plt.plot(range(1, len(val_rmse_curve) + 1), val_rmse_curve,
         label='Validation Error', color='#D85140', linestyle='-', linewidth=3)

plt.xlim(-len(train_rmse_curve) * 0.05, len(train_rmse_curve) * 1.05)
y_min = min(np.min(train_rmse_curve), np.min(val_rmse_curve))
y_max = max(np.max(train_rmse_curve[:10]), np.max(val_rmse_curve[:10]))
plt.ylim(y_min * 0.8, y_max * 1.2)

plt.xlabel('Epoch', fontweight='bold', fontsize=28, family='Times New Roman')
plt.ylabel('RMSE (℃)', fontweight='bold', fontsize=28, family='Times New Roman')
plt.legend(loc='upper right', frameon=True, fontsize=20, prop={'family': 'Times New Roman', 'weight': 'bold'})
plt.text(0.02, 0.95, '(a)', transform=plt.gca().transAxes, fontsize=35, fontweight='bold', family='Times New Roman')

plt.tick_params(axis='both', which='major', labelsize=24, direction='in', length=10, width=2)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontweight('bold')

plt.tight_layout()
plt.show()

# 11. MAE vs Epoch 图
plt.figure(figsize=(12, 8))

plt.plot(range(1, len(train_mae_curve) + 1), train_mae_curve,
         label='Training MAE', color='#8DB48E', linestyle='--', linewidth=3)
plt.plot(range(1, len(test_mae_curve) + 1), test_mae_curve,
         label='Validation MAE', color='#D85140', linestyle='-', linewidth=3)

plt.xlim(-len(train_mae_curve) * 0.05, len(train_mae_curve) * 1.05)
y_min = min(np.min(train_mae_curve), np.min(test_mae_curve))
y_max = max(np.max(train_mae_curve[:10]), np.max(test_mae_curve[:10]))
plt.ylim(y_min * 0.8, y_max * 1.2)

plt.xlabel('Epoch', fontweight='bold', fontsize=28, family='Times New Roman')
plt.ylabel('MAE (℃)', fontweight='bold', fontsize=28, family='Times New Roman')
plt.legend(loc='upper right', frameon=True, fontsize=20, prop={'family': 'Times New Roman', 'weight': 'bold'})
plt.text(0.02, 0.95, '(b)', transform=plt.gca().transAxes, fontsize=35, fontweight='bold', family='Times New Roman')

plt.tick_params(axis='both', which='major', labelsize=24, direction='in', length=10, width=2)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontweight('bold')

plt.tight_layout()
plt.show()

# 11. 模型结构信息
print("\n=== MLP Model Architecture ===")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")