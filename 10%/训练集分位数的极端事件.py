import numpy as np
import pandas as pd
import os

# ====================== 配置 ======================
DATA_DIR = r"D:\基于因果尾部系数的金属期货极端风险方向性传导与预警研究"
OUTPUT_DIR = os.path.join(DATA_DIR, "10%阈值")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 9种金属代码
ALL_METALS = ['PB', 'CU', 'ZN', 'AL', 'NI', 'AG', 'AU', 'RB', 'SN']

TAU = 0.10             # 极端事件分位数阈值
SPLIT_RATIO = 0.8       # 训练集比例

# ====================== 数据加载 ======================
price_files = {
    'PB': 'SHFE_PB_Main_Contract_Daily_20150327_20251231.csv',
    'CU': 'SHFE_CU_Main_Contract_Daily_20150327_20251231.csv',
    'ZN': 'SHFE_ZN_Main_Contract_Daily_20150327_20251231.csv',
    'AG': 'SHFE_AG_Main_Contract_Daily_20150327_20251231.csv',
    'AU': 'SHFE_AU_Main_Contract_Daily_20150327_20251231.csv',
    'AL': 'SHFE_AL_Main_Contract_Daily_20150327_20251231.csv',
    'NI': 'SHFE_NI_Main_Contract_Daily_20150327_20251231.csv',
    'RB': 'SHFE_RB_Main_Contract_Daily_20150327_20251231.csv',
    'SN': 'SHFE_SN_Main_Contract_Daily_20150327_20251231.csv'
}

returns_dict = {}
for name, fname in price_files.items():
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    returns_dict[name] = df[['date', 'return']].dropna().set_index('date')['return']

returns = pd.DataFrame(returns_dict)
common_dates = returns.dropna().index
n = len(common_dates)
split_idx = int(n * SPLIT_RATIO)
train_dates = common_dates[:split_idx]
test_dates  = common_dates[split_idx:]

print(f"公共日期: {common_dates[0]} 至 {common_dates[-1]}, 共 {n} 天")
print(f"训练集:   {train_dates[0]} 至 {train_dates[-1]}, 共 {len(train_dates)} 天")
print(f"测试集:   {test_dates[0]} 至 {test_dates[-1]}, 共 {len(test_dates)} 天")

# ====================== 生成极端事件 ======================
stats_rows = []
for name in ALL_METALS:
    ret = returns[name]

    # 训练集上拟合阈值
    train_ret = ret.loc[train_dates].dropna()
    left_thresh  = np.quantile(train_ret, TAU)
    right_thresh = np.quantile(train_ret, 1 - TAU)

    # 用训练集阈值标记整个公共日期区间
    # （训练集区间的事件数≈训练集15%，测试集用同样的阈值，才符合"训练集分位数"逻辑）
    full_df = pd.DataFrame({
        'date': common_dates,
        'return': ret.loc[common_dates].values
    })
    full_df['extreme_left']  = (full_df['return'] <= left_thresh).astype(int)
    full_df['extreme_right'] = (full_df['return'] >= right_thresh).astype(int)
    full_df['extreme_event'] = 0
    full_df.loc[full_df['extreme_left'] == 1, 'extreme_event'] = -1
    full_df.loc[full_df['extreme_right'] == 1, 'extreme_event'] = 1

    # 保存
    out_file = os.path.join(OUTPUT_DIR, f'SHFE_{name}_Extreme_Events_Threshold_Method_TrainBased.csv')
    full_df.to_csv(out_file, index=False, encoding='utf-8-sig')

    # 统计训练集和测试集的极端事件次数
    train_df = full_df[full_df['date'].isin(train_dates)]
    test_df  = full_df[full_df['date'].isin(test_dates)]

    stats_rows.append({
        '品种': name,
        '左尾阈值': round(left_thresh, 6),
        '右尾阈值': round(right_thresh, 6),
        '训练集左尾': int(train_df['extreme_left'].sum()),
        '训练集右尾': int(train_df['extreme_right'].sum()),
        '训练集频率-左': f"{train_df['extreme_left'].mean():.2%}",
        '训练集频率-右': f"{train_df['extreme_right'].mean():.2%}",
        '测试集左尾': int(test_df['extreme_left'].sum()),
        '测试集右尾': int(test_df['extreme_right'].sum()),
    })

# ====================== 打印验证表 ======================
df_stats = pd.DataFrame(stats_rows)
print("\n基于训练集分位数的极端事件统计（验证）")
print("="*100)
print(df_stats.to_string(index=False))
print("="*100)

# 保存
stats_file = os.path.join(OUTPUT_DIR, '基于训练集分位数的极端事件统计.csv')
df_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
print(f"\n统计表已保存至: {stats_file}")
print("\n完成。新建的训练集阈值极端事件CSV已保存到 OUTPUT_DIR 中。")