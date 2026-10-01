import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

input_file_path = r"D:\基于因果尾部系数的金属期货极端风险方向性传导与预警研究\SHFE_CU_Main_Contract_Daily_20150327_20251231.csv"
output_dir = r"D:\基于因果尾部系数的金属期货极端风险方向性传导与预警研究"
left_tail_percentile = 15   
right_tail_percentile = 85  


# 检查并创建输出目录
os.makedirs(output_dir, exist_ok=True)
print("正在使用阈值法识别极端风险事件...")
print(f"输入文件: {input_file_path}")
print(f"左尾阈值: 收益率分布的 {left_tail_percentile}% 分位数 (极端下跌)")
print(f"右尾阈值: 收益率分布的 {right_tail_percentile}% 分位数 (极端上涨)")
print("-" * 60)

# 1. 读取数据并准备
df = pd.read_csv(input_file_path, parse_dates=['date'])
df.sort_values('date', inplace=True)  # 确保按日期排序
df.reset_index(drop=True, inplace=True)

# 2. 计算对数收益率 (如果尚未计算)
if 'log_return' not in df.columns:
    print("计算对数收益率序列...")
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    # 删除第一行的NaN
    df.dropna(subset=['log_return'], inplace=True)

print(f"用于分析的收益率数据行数: {len(df)}")

# 3. 计算阈值
left_threshold = np.percentile(df['log_return'], left_tail_percentile)
right_threshold = np.percentile(df['log_return'], right_tail_percentile)

print(f"左尾(暴跌)阈值: {left_threshold:.6f}")
print(f"右尾(暴涨)阈值: {right_threshold:.6f}")

# 4. 应用阈值法，生成极端事件信号
# 1 表示发生极端事件，0 表示未发生
df['extreme_left'] = (df['log_return'] <= left_threshold).astype(int)
df['extreme_right'] = (df['log_return'] >= right_threshold).astype(int)

df['extreme_event'] = 0
df.loc[df['extreme_left'] == 1, 'extreme_event'] = -1  # 用-1标记下跌
df.loc[df['extreme_right'] == 1, 'extreme_event'] = 1   # 用1标记上涨

# 5. 统计结果
left_event_count = df['extreme_left'].sum()
right_event_count = df['extreme_right'].sum()
total_event_count = left_event_count + right_event_count

print("\n=== 极端事件识别结果 ===")
print(f"识别出的极端下跌事件(左尾)数量: {left_event_count}")
print(f"识别出的极端上涨事件(右尾)数量: {right_event_count}")
print(f"极端事件总计: {total_event_count}")
print(f"极端事件占比: {total_event_count/len(df)*100:.2f}%")

# 6. 保存结果到CSV
output_filename = "SHFE_CU_Extreme_Events_Threshold_Method.csv"
output_path = os.path.join(output_dir, output_filename)

# 选择要保存的列
columns_to_save = ['date', 'symbol', 'close', 'log_return', 
                   'extreme_left', 'extreme_right', 'extreme_event']
df_result = df[columns_to_save].copy()

df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n极端事件识别完成！")
print(f"结果已保存至: {output_path}")

print("\n正在生成诊断图表...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 7.1 收益率分布与阈值
axes[0, 0].hist(df['log_return'].values, bins=80, alpha=0.7, density=True, color='skyblue', edgecolor='black')
axes[0, 0].axvline(x=left_threshold, color='red', linestyle='--', linewidth=2, label=f'左尾阈值 ({left_threshold:.4f})')
axes[0, 0].axvline(x=right_threshold, color='green', linestyle='--', linewidth=2, label=f'右尾阈值 ({right_threshold:.4f})')
axes[0, 0].axvspan(df['log_return'].min(), left_threshold, alpha=0.3, color='red', label='极端下跌区域')
axes[0, 0].axvspan(right_threshold, df['log_return'].max(), alpha=0.3, color='green', label='极端上涨区域')
axes[0, 0].set_title('铅期货主力合约收益率分布与阈值')
axes[0, 0].set_xlabel('对数收益率')
axes[0, 0].set_ylabel('密度')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 7.2 价格序列与极端事件标记
ax2 = axes[0, 1]
ax2.plot(df['date'], df['close'], color='black', alpha=0.7, linewidth=1, label='收盘价')
# 标记极端下跌事件
left_dates = df[df['extreme_left'] == 1]['date']
left_prices = df[df['extreme_left'] == 1]['close']
ax2.scatter(left_dates, left_prices, color='red', s=25, zorder=5, label=f'极端下跌 ({left_event_count}次)')
# 标记极端上涨事件
right_dates = df[df['extreme_right'] == 1]['date']
right_prices = df[df['extreme_right'] == 1]['close']
ax2.scatter(right_dates, right_prices, color='green', s=25, zorder=5, label=f'极端上涨 ({right_event_count}次)')
ax2.set_title('铅期货价格与极端事件发生点')
ax2.set_xlabel('日期')
ax2.set_ylabel('价格')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 7.3 极端事件时间线
axes[1, 0].plot(df['date'], df['extreme_event'], drawstyle='steps-post', linewidth=1.5, color='purple')
axes[1, 0].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
axes[1, 0].fill_between(df['date'], 0, df['extreme_event'], where=df['extreme_event']>0, color='green', alpha=0.5, step='post')
axes[1, 0].fill_between(df['date'], 0, df['extreme_event'], where=df['extreme_event']<0, color='red', alpha=0.5, step='post')
axes[1, 0].set_title('极端事件信号时间线 (-1:下跌, 1:上涨)')
axes[1, 0].set_xlabel('日期')
axes[1, 0].set_ylabel('事件信号')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([-1.5, 1.5])

# 7.4 极端事件样本预览表格
# 选取最近5次极端事件作为样例展示在图上
sample_events = df[df['extreme_event'] != 0].tail(5)[['date', 'log_return', 'extreme_event']]
cell_text = []
for _, row in sample_events.iterrows():
    event_type = "下跌" if row['extreme_event'] == -1 else "上涨"
    cell_text.append([row['date'].strftime('%Y-%m-%d'), f"{row['log_return']:.4f}", event_type])
axes[1, 1].axis('tight')
axes[1, 1].axis('off')
axes[1, 1].set_title('最近5次极端事件样例')
table = axes[1, 1].table(cellText=cell_text, 
                         colLabels=['日期', '收益率', '类型'], 
                         loc='center', 
                         cellLoc='center',
                         colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.tight_layout()

# 保存图表
chart_path = os.path.join(output_dir, "SHFE_CU_Extreme_Events_Analysis_Chart.png")
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"诊断图表已保存至: {chart_path}")

# 显示图表 (如果环境支持)
try:
    plt.show()
except:
    print("(图表已保存，如需查看请打开文件)")

print("\n" + "="*60)
print("下一步建议:")
print("1. 检查生成的CSV文件，确认极端事件识别是否符合预期。")
print("2. 查看PNG图表，直观验证阈值设定是否合理。")
print("3. 对铜(CU)、锌(ZN)数据重复此流程，为后续CTC分析准备数据。")
print("="*60)