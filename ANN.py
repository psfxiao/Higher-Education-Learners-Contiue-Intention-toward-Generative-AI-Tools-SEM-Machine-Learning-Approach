import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==================== 1. 加载数据 ====================
df = pd.read_csv("ANN_input.csv")
print(f"样本量：{len(df)}")
if 'TTF' in df.columns and 'TTF' not in df.columns:
    df.rename(columns={'TTF': 'TTF'}, inplace=True)
features = ['PEU', 'PEC', 'PTF', 'TTF']

# ==================== 2. 模型构建 ====================
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model

# ==================== 3. 训练评估函数（固定划分 + 10次重复）====================
def train_evaluate(X, y, n_repeats=10, test_size=0.2, val_ratio=0.1875, random_state=42):
    """
    固定划分，重复训练n_repeats次。
    返回：
        stats: (r2_mean, r2_std, rmse_mean, rmse_std, mae_mean, mae_std, imp_mean, imp_std)
        details: dict 包含每次的train_rmse, test_rmse, test_r2, test_mae, importance
    """
    # 归一化
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # 固定划分（只做一次）
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state)

    details = {
        'train_rmse': [], 'test_rmse': [], 'test_r2': [], 'test_mae': [], 'importance': []
    }

    for _ in range(n_repeats):
        model = build_model(X.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=200, batch_size=32, callbacks=[early_stop], verbose=0)

        # 训练集预测
        y_train_pred = scaler_y.inverse_transform(model.predict(X_train, verbose=0)).ravel()
        y_train_true = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
        train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))

        # 测试集预测
        y_test_pred = scaler_y.inverse_transform(model.predict(X_test, verbose=0)).ravel()
        y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
        test_r2 = r2_score(y_test_true, y_test_pred)
        test_mae = mean_absolute_error(y_test_true, y_test_pred)

        details['train_rmse'].append(train_rmse)
        details['test_rmse'].append(test_rmse)
        details['test_r2'].append(test_r2)
        details['test_mae'].append(test_mae)

        # Olden 变量重要性
        weights = [layer.get_weights()[0] for layer in model.layers if layer.get_weights()]
        if len(weights) >= 3:
            w1, w2, w3 = weights[0], weights[1], weights[2]
            imp = np.zeros(w1.shape[0])
            for i in range(w1.shape[0]):
                total = 0.0
                for j in range(w1.shape[1]):
                    for k in range(w2.shape[1]):
                        total += abs(w1[i, j] * w2[j, k] * w3[k, 0])
                imp[i] = total
        else:
            w1, w2 = weights[0], weights[1]
            imp = np.sum(np.abs(w1) * np.abs(w2.T), axis=1)
        imp = imp / np.sum(imp)
        details['importance'].append(imp)

    # 汇总统计
    r2_mean = np.mean(details['test_r2'])
    r2_std = np.std(details['test_r2'])
    rmse_mean = np.mean(details['test_rmse'])
    rmse_std = np.std(details['test_rmse'])
    mae_mean = np.mean(details['test_mae'])
    mae_std = np.std(details['test_mae'])
    imp_mean = np.mean(details['importance'], axis=0)
    imp_std = np.std(details['importance'], axis=0)

    stats = (r2_mean, r2_std, rmse_mean, rmse_std, mae_mean, mae_std, imp_mean, imp_std)
    return stats, details

# ==================== 4. 运行三种模型 ====================
X_main = df[features].values
y_bi = df['BI'].values
y_att = df['ATT'].values

print("\n主模型 (4特征 → BI)")
stats_main, det_main = train_evaluate(X_main, y_bi)
r2_m, r2_std_m, rmse_m, rmse_std_m, mae_m, mae_std_m, imp_m, imp_std_m = stats_main
print(f"R² = {r2_m:.4f} ± {r2_std_m:.4f}")

print("\n辅助模型 (4特征 → ATT)")
stats_att, det_att = train_evaluate(X_main, y_att)
r2_a, r2_std_a, rmse_a, rmse_std_a, mae_a, mae_std_a, imp_a, imp_std_a = stats_att
print(f"R² = {r2_a:.4f} ± {r2_std_a:.4f}")

print("\n简化模型 (PEU, PTF, TTF → BI)")
X_red = df[['PEU', 'PTF', 'TTF']].values
stats_red, det_red = train_evaluate(X_red, y_bi)
r2_r, r2_std_r, rmse_r, rmse_std_r, mae_r, mae_std_r, _, _ = stats_red
print(f"R² = {r2_r:.4f} ± {r2_std_r:.4f}")
print(f"完整模型 R² = {r2_m:.4f}, 简化模型 R² = {r2_r:.4f}, 提升 = {r2_m - r2_r:.4f}")

# ==================== 5. 构建Excel输出 ====================
# 性能汇总
perf_df = pd.DataFrame({
    '模型': ['主模型(PEU,PEC,PTF,TTF→BI)', '辅助模型(→ATT)', '简化模型(无PEC→BI)'],
    'R²均值': [r2_m, r2_a, r2_r],
    'R²标准差': [r2_std_m, r2_std_a, r2_std_r],
    'RMSE均值': [rmse_m, rmse_a, rmse_r],
    'RMSE标准差': [rmse_std_m, rmse_std_a, rmse_std_r],
    'MAE均值': [mae_m, mae_a, mae_r],
    'MAE标准差': [mae_std_m, mae_std_a, mae_std_r]
})

# 变量重要性汇总
imp_df = pd.DataFrame({
    '变量': features,
    '预测BI重要性均值': imp_m,
    '预测BI重要性标准差': imp_std_m,
    '预测ATT重要性均值': imp_a,
    '预测ATT重要性标准差': imp_std_a
})

# 主模型详细
main_rmse_detail = pd.DataFrame({
    'Run': range(1, 11),
    'Training RMSE': det_main['train_rmse'],
    'Testing RMSE': det_main['test_rmse'],
    'Testing R²': det_main['test_r2'],
    'Testing MAE': det_main['test_mae']
})
main_imp_detail = pd.DataFrame(det_main['importance'], columns=features)
main_imp_detail.insert(0, 'Run', range(1, 11))

# 辅助模型详细
att_rmse_detail = pd.DataFrame({
    'Run': range(1, 11),
    'Training RMSE': det_att['train_rmse'],
    'Testing RMSE': det_att['test_rmse'],
    'Testing R²': det_att['test_r2'],
    'Testing MAE': det_att['test_mae']
})
att_imp_detail = pd.DataFrame(det_att['importance'], columns=features)
att_imp_detail.insert(0, 'Run', range(1, 11))

# 简化模型详细（仅RMSE）
red_rmse_detail = pd.DataFrame({
    'Run': range(1, 11),
    'Training RMSE': det_red['train_rmse'],
    'Testing RMSE': det_red['test_rmse'],
    'Testing R²': det_red['test_r2'],
    'Testing MAE': det_red['test_mae']
})

# 保存到Excel
output = "ANN_Complete_Detailed.xlsx"
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    perf_df.to_excel(writer, sheet_name="模型性能汇总", index=False)
    imp_df.to_excel(writer, sheet_name="变量重要性汇总", index=False)
    main_rmse_detail.to_excel(writer, sheet_name="主模型_Detail_RMSE", index=False)
    main_imp_detail.to_excel(writer, sheet_name="主模型_Detail_Importance", index=False)
    att_rmse_detail.to_excel(writer, sheet_name="辅助模型_Detail_RMSE", index=False)
    att_imp_detail.to_excel(writer, sheet_name="辅助模型_Detail_Importance", index=False)
    red_rmse_detail.to_excel(writer, sheet_name="简化模型_Detail_RMSE", index=False)

print(f"\n所有结果已保存至: {output}")
print("注意：本代码采用固定划分（random_state=42），汇总结果应与您表8完全一致。")