import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ====================== 配置 =====================
DATA_DIR = r"D:\基于因果尾部系数的金属期货极端风险方向性传导与预警研究"
OUTPUT_DIR = os.path.join(DATA_DIR, "10%阈值")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 9 种金属
ALL_METALS = ['PB', 'CU', 'ZN', 'AL', 'NI', 'AG', 'AU', 'RB', 'SN']

# 是否使用完整期（True = 全样本；False = 仅测试集）
USE_FULL_SAMPLE = True

# 训练/测试分割比例（仅当 USE_FULL_SAMPLE=False 时生效）
SPLIT_RATIO = 0.8

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

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

# 合并为宽表并取 9 种金属的公共日期
returns = pd.DataFrame(returns_dict)
common_dates = returns.dropna().index
print(f"公共日期区间: {common_dates[0]} 至 {common_dates[-1]}, 共 {len(common_dates)} 天")

# ====================== 确定绘图区间 ======================
if USE_FULL_SAMPLE:
    plot_dates = common_dates
    period_label = "完整期"
else:
    n = len(common_dates)
    split_idx = int(n * SPLIT_RATIO)
    plot_dates = common_dates[split_idx:]
    period_label = "测试集"

print(f"绘图区间: {plot_dates[0]} 至 {plot_dates[-1]}, 共 {len(plot_dates)} 天")

# ====================== 循环绘图 ======================
for metal in ALL_METALS:
    series = returns.loc[plot_dates, metal]

    plt.figure(figsize=(12, 5))
    plt.plot(series.index, series.values, color='steelblue', linewidth=0.8)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
    plt.title(f'{period_label}{metal}期货日收益率时序图')
    plt.xlabel('日期')
    plt.ylabel('收益率')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f'{period_label}{metal}期货收益率时序图.png'
    )
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"已保存: {output_file}")

print("\n全部完成。")