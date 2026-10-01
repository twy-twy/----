import numpy as np
import pandas as pd
import os
import scipy.stats as stats

# ====================== 配置 ======================
DATA_DIR = r"D:\基于因果尾部系数的金属期货极端风险方向性传导与预警研究"
OUTPUT_DIR = r"D:\基于因果尾部系数的金属期货极端风险方向性传导与预警研究\10%阈值"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LAG = 5          # 最大滞后阶数
SPLIT_RATIO = 0.8    # 训练集比例


NAME_MAP = {
    '锡': 'SN', '铝': 'AL', '白银': 'AG', '黄金': 'AU',
    '镍': 'NI', '螺纹钢': 'RB', '铅': 'PB', '锌': 'ZN',
    '铜': 'CU'
}

# ===== 需要检验的方向：10% 阈值、窗口 (1,3) 下的 12 条显著方向 =====
DIRECTIONS = [
    ('铜左尾', '铅右尾'),        # CU左尾→PB右尾
    ('铜左尾', '铝左尾'),        # CU左尾→AL左尾
    ('镍右尾', '铝右尾'),        # NI右尾→AL右尾
    ('锡左尾', '铝左尾'),        # SN左尾→AL左尾
    ('铅左尾', '镍左尾'),        # PB左尾→NI左尾
    ('白银右尾', '镍右尾'),      # AG右尾→NI右尾
    ('锡右尾', '镍左尾'),        # SN右尾→NI左尾
    ('锡右尾', '镍右尾'),        # SN右尾→NI右尾
    ('锡右尾', '白银右尾'),      # SN右尾→AG右尾
    ('镍右尾', '黄金左尾'),      # NI右尾→AU左尾
    ('白银右尾', '黄金右尾'),    # AG右尾→AU右尾
    ('铅左尾', '螺纹钢左尾'),    # PB左尾→RB左尾
]

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
print(f"公共日期区间: {common_dates[0]} 至 {common_dates[-1]}, 共 {len(common_dates)} 天")

# 80/20 分割
n = len(common_dates)
split_idx = int(n * SPLIT_RATIO)
train_dates = common_dates[:split_idx]
train_returns = returns.loc[train_dates]

print(f"训练集: {train_dates[0]} 至 {train_dates[-1]}, 共 {len(train_dates)} 天")


# ====================== 格兰杰因果检验函数 ======================
def granger_causality_pvalue(X, Y, lag):
    """
    检验 X 是否 Granger-cause Y（滞后 lag 阶）
    返回: F统计量, p值
    """
    n = len(X)
    Y_curr = Y[lag:]
    Y_lags = np.column_stack([Y[lag - i:n - i] for i in range(1, lag + 1)])
    X_lags = np.column_stack([X[lag - i:n - i] for i in range(1, lag + 1)])

    # 受限模型
    X_r = np.column_stack([np.ones(len(Y_curr)), Y_lags])
    beta_r = np.linalg.lstsq(X_r, Y_curr, rcond=None)[0]
    rss_r = np.sum((Y_curr - X_r @ beta_r) ** 2)

    # 非受限模型
    X_u = np.column_stack([np.ones(len(Y_curr)), Y_lags, X_lags])
    beta_u = np.linalg.lstsq(X_u, Y_curr, rcond=None)[0]
    rss_u = np.sum((Y_curr - X_u @ beta_u) ** 2)

    df1 = lag
    df2 = len(Y_curr) - X_u.shape[1]
    if df2 <= 0 or rss_u == 0:
        return np.nan, np.nan
    F = ((rss_r - rss_u) / df1) / (rss_u / df2)
    p = 1 - stats.f.cdf(F, df1, df2)
    return F, p


# ====================== 解析方向 ======================
def parse_metal(full_str):
    """从完整描述中提取金属名称，例如 '锡左尾' -> '锡'"""
    metals = ['螺纹钢', '白银', '黄金', '锡', '铝', '镍', '铅', '锌', '铜']
    for m in metals:
        if full_str.startswith(m):
            return m
    return None


# ====================== 执行检验 ======================
print("\n均值格兰杰因果检验结果（10% 阈值，窗口 [1,3] 下的显著方向）")
print("基于训练集收益率")
print("=" * 100)
print(f"{'方向':<25} {'L1':<10} {'L2':<10} {'L3':<10} {'L4':<10} {'L5':<10}")
print("-" * 100)

results = []
for cause_full, effect_full in DIRECTIONS:
    cause_name = parse_metal(cause_full)
    effect_name = parse_metal(effect_full)
    if cause_name is None or effect_name is None:
        print(f"无法解析方向: {cause_full} → {effect_full}")
        continue
    cause_code = NAME_MAP.get(cause_name)
    effect_code = NAME_MAP.get(effect_name)
    if cause_code is None or effect_code is None:
        print(f"未知金属: {cause_name} 或 {effect_name}")
        continue

    X = train_returns[cause_code].values
    Y = train_returns[effect_code].values

    row = [f"{cause_full}→{effect_full}"]
    for lag in range(1, MAX_LAG + 1):
        F, p = granger_causality_pvalue(X, Y, lag)
        row.append(f"{p:.4f}" if not np.isnan(p) else "NaN")
    results.append(row)
    print(f"{row[0]:<25} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10}")

# 保存结果
df_res = pd.DataFrame(results, columns=['方向', 'L1', 'L2', 'L3', 'L4', 'L5'])
output_file = os.path.join(
    OUTPUT_DIR,
    '均值格兰杰因果检验结果_10%阈值_窗口1_3.csv'
)
df_res.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n结果已保存至: {output_file}")