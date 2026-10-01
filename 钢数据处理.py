import pandas as pd
import os

# 读取数据
df = pd.read_csv('SHFE_RB_Raw_Daily_20150327_20251231.csv')

# 确保日期列是datetime类型
df['date'] = pd.to_datetime(df['date'])

# 方法一：使用 idxmax（推荐）
# 找出每个日期中成交量最大的行索引
idx = df.groupby('date')['volume'].idxmax()
main_contracts = df.loc[idx].reset_index(drop=True)

# 方法二：如果成交量为0或缺失，可先过滤，再用 idxmax
# valid = df[df['volume'].notna() & (df['volume'] > 0)]
# idx = valid.groupby('date')['volume'].idxmax()
# main_contracts = valid.loc[idx].reset_index(drop=True)

# 保存输出文件夹路径
output_folder = r'D:\基于因果尾部系数的金属期货极端风险方向性传导与预警研究'
os.makedirs(output_folder, exist_ok=True)

# 保存主力合约数据
output_file = os.path.join(output_folder, 'SHFE_RB_Main_Contract_Daily_20150327_20251231.csv')
main_contracts.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"数据处理完成!")
print(f"原始数据记录数: {len(df)}")
print(f"主力合约记录数: {len(main_contracts)}")
print(f"保存路径: {output_file}")
print(main_contracts.head())