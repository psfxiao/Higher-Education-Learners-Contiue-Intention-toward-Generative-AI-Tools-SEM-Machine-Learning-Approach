# -*- coding: utf-8 -*-
"""
SVR 补充分析 - 基于SEM结果（含PEU, PEC, PTF, TTF → BI/ATT）
716样本，可选重复10次，报告性能及变量重要性
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

# ==================== 1. 加载数据 ====================
df = pd.read_csv("ANN_input.csv")  # 应包含列: PEU, PEC, PTF, TTF, ATT, BI
print(f"样本量：{len(df)}")
print(df.head())


feature_cols = ['PEU', 'PEC', 'PTF', 'TTF']
target_cols = {'BI': 'BI', 'ATT': 'ATT'}


# ==================== 2. 定义训练评估函数（支持多次不同划分） ====================
def train_evaluate_svr(X, y, n_repeats=10, test_size=0.2, val_ratio=0.1875, random_state=42):
    """
    采用60/20/20划分，可重复多次（每次不同随机种子），
    返回性能统计（均值±标准差）以及每次重复的详细记录
    """
    details = {
        'train_sizes': [],
        'test_sizes': [],
        'train_rmse': [],
        'test_rmse': [],
        'test_r2': [],
        'test_mae': [],
        'importance': []  # 每次的归一化重要性（排列重要性）
    }

    for run in range(n_repeats):
        # 每次划分使用不同的随机种子，以评估模型稳定性
        current_seed = random_state + run
        # 第一次划分：分出测试集 (20%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=current_seed)
        # 从剩余80%中分出验证集 (占原始数据的20%，即 X_temp 的 25%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=current_seed)

        # 标准化（SVR对尺度敏感，使用StandardScaler）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # 注意：SVR不需要对y标准化，因为RBF核不依赖y尺度，但为了一致性，可标准化，这里不标准化y

        # 使用默认参数（经先前网格搜索验证最优）
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
        svr.fit(X_train_scaled, y_train)

        # 预测测试集
        y_pred = svr.predict(X_test_scaled)

        # 性能指标
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)

        # 训练集性能（可选）
        y_train_pred = svr.predict(X_train_scaled)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        # 记录详细
        details['train_sizes'].append(len(y_train))
        details['test_sizes'].append(len(y_test))
        details['train_rmse'].append(train_rmse)
        details['test_rmse'].append(test_rmse)
        details['test_r2'].append(test_r2)
        details['test_mae'].append(test_mae)

        # 变量重要性（排列重要性）- 使用测试集
        perm_importance = permutation_importance(
            svr, X_test_scaled, y_test, n_repeats=5, random_state=current_seed, scoring='r2'
        )
        imp = perm_importance.importances_mean
        # 归一化（总和为1）
        imp = imp / np.sum(imp)
        details['importance'].append(imp)

    # 汇总统计
    r2_mean = np.mean(details['test_r2'])
    r2_std = np.std(details['test_r2'])
    rmse_mean = np.mean(details['test_rmse'])
    rmse_std = np.std(details['test_rmse'])
    mae_mean = np.mean(details['test_mae'])
    mae_std = np.std(details['test_mae'])
    importance_mean = np.mean(details['importance'], axis=0)
    importance_std = np.std(details['importance'], axis=0)

    stats = (r2_mean, r2_std, rmse_mean, rmse_std, mae_mean, mae_std,
             importance_mean, importance_std)
    return stats, details


# ==================== 3. 主模型：4个输入 → BI ====================
print("\n" + "=" * 60)
print("主模型：输入 [PEU, PEC, PTF, TTF] → 输出 BI")
X_main = df[feature_cols].values
y_main = df['BI'].values

stats_main, details_main = train_evaluate_svr(X_main, y_main, n_repeats=10)
(r2_mean, r2_std, rmse_mean, rmse_std, mae_mean, mae_std, imp_mean, imp_std) = stats_main

print(f"测试集 R²  = {r2_mean:.4f} ± {r2_std:.4f}")
print(f"测试集 RMSE = {rmse_mean:.4f} ± {rmse_std:.4f}")
print(f"测试集 MAE  = {mae_mean:.4f} ± {mae_std:.4f}")

print("\n变量重要性（排列重要性，归一化）:")
for f, imp, sd in zip(feature_cols, imp_mean, imp_std):
    print(f"  {f}: {imp:.3f} ± {sd:.3f}")

# ==================== 4. 辅助模型：4个输入 → ATT ====================
print("\n" + "=" * 60)
print("辅助模型：输入 [PEU, PEC, PTF, TTF] → 输出 ATT")
y_att = df['ATT'].values
stats_att, details_att = train_evaluate_svr(X_main, y_att, n_repeats=10)
(r2_att, r2_att_std, rmse_att, rmse_att_std, mae_att, mae_att_std, imp_att, imp_att_std) = stats_att
print(f"测试集 R² (ATT) = {r2_att:.4f} ± {r2_att_std:.4f}")

# ==================== 5. 简化模型（剔除PEC）→ BI ====================
print("\n" + "=" * 60)
print("简化模型：输入 [PEU, PTF, TTF] → 输出 BI（剔除PEC）")
X_reduced = df[['PEU', 'PTF', 'TTF']].values
stats_red, details_red = train_evaluate_svr(X_reduced, y_main, n_repeats=10)
(r2_red, r2_red_std, rmse_red, rmse_red_std, mae_red, mae_red_std, _, _) = stats_red
print(f"简化模型 R² = {r2_red:.4f} ± {r2_red_std:.4f}")
print(f"完整模型 R² = {r2_mean:.4f}")
print(f"R² 提升量 = {r2_mean - r2_red:.4f}")

# ==================== 6. 保存结果到Excel ====================
# 6.1 性能汇总表
summary_df = pd.DataFrame({
    '模型': ['主模型(PEU,PEC,PTF,TTF→BI)', '辅助模型(→ATT)', '简化模型(无PEC→BI)'],
    'R²均值': [r2_mean, r2_att, r2_red],
    'R²标准差': [r2_std, r2_att_std, r2_red_std],
    'RMSE均值': [rmse_mean, rmse_att, rmse_red],
    'RMSE标准差': [rmse_std, rmse_att_std, rmse_red_std],
    'MAE均值': [mae_mean, mae_att, mae_red],
    'MAE标准差': [mae_std, mae_att_std, mae_red_std]
})

# 6.2 变量重要性汇总
imp_df = pd.DataFrame({
    '变量': feature_cols,
    '预测BI重要性均值': imp_mean,
    '预测BI重要性标准差': imp_std,
    '预测ATT重要性均值': imp_att,
    '预测ATT重要性标准差': imp_att_std
})

# 6.3 主模型详细性能（10次划分）
main_detail_rmse = pd.DataFrame({
    'Run': range(1, 11),
    'Training RMSE': details_main['train_rmse'],
    'Testing RMSE': details_main['test_rmse'],
    'Testing R²': details_main['test_r2'],
    'Testing MAE': details_main['test_mae']
})

# 6.4 主模型详细重要性
main_detail_imp = pd.DataFrame(details_main['importance'], columns=feature_cols)
main_detail_imp.insert(0, 'Run', range(1, 11))

# 6.5 辅助模型详细性能
att_detail_rmse = pd.DataFrame({
    'Run': range(1, 11),
    'Training RMSE': details_att['train_rmse'],
    'Testing RMSE': details_att['test_rmse'],
    'Testing R²': details_att['test_r2'],
    'Testing MAE': details_att['test_mae']
})

# 6.6 辅助模型详细重要性
att_detail_imp = pd.DataFrame(details_att['importance'], columns=feature_cols)
att_detail_imp.insert(0, 'Run', range(1, 11))

# 6.7 简化模型详细性能
red_detail_rmse = pd.DataFrame({
    'Run': range(1, 11),
    'Training RMSE': details_red['train_rmse'],
    'Testing RMSE': details_red['test_rmse'],
    'Testing R²': details_red['test_r2'],
    'Testing MAE': details_red['test_mae']
})

# 写入Excel
output_file = "SVR_Results_Detailed.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    summary_df.to_excel(writer, sheet_name="性能汇总", index=False)
    imp_df.to_excel(writer, sheet_name="变量重要性汇总", index=False)
    main_detail_rmse.to_excel(writer, sheet_name="主模型_详细性能", index=False)
    main_detail_imp.to_excel(writer, sheet_name="主模型_详细重要性", index=False)
    att_detail_rmse.to_excel(writer, sheet_name="辅助模型_详细性能", index=False)
    att_detail_imp.to_excel(writer, sheet_name="辅助模型_详细重要性", index=False)
    red_detail_rmse.to_excel(writer, sheet_name="简化模型_详细性能", index=False)

print(f"\nSVR结果已保存至: {output_file}")

# ==================== 7. 提取 IPA 图所需的重要性字典 ====================
ipa_importance = {var: imp_mean[i] for i, var in enumerate(feature_cols)}
print("\n【IPA 图重要性字典】")
print("importance_dict = {")
for var, val in ipa_importance.items():
    print(f"    '{var}': {val:.3f},")
print("}")

with open("ipa_importance.txt", "w") as f:
    f.write("importance_dict = {\n")
    for var, val in ipa_importance.items():
        f.write(f"    '{var}': {val:.3f},\n")
    f.write("}\n")
print("\n字典已保存至 ipa_importance.txt")