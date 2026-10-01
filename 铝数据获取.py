import akshare as ak
import pandas as pd
import os
import time

target_variety = 'AL'         
start_date_str = "20150327"   
end_date_str = "20251231"     
save_dir = r"D:\基于因果尾部系数的金属期货极端风险方向性传导与预警研究"   

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 构造输出文件完整路径
output_filename = f"SHFE_{target_variety}_Raw_Daily_{start_date_str}_{end_date_str}.csv"
output_path = os.path.join(save_dir, output_filename)

print(f"目标：保存 {target_variety} 数据至 {output_path}")
print(f"时间范围：{start_date_str} 至 {end_date_str}")
print("开始获取数据...")

all_data_frames = []
start_year = int(start_date_str[:4])
end_year = int(end_date_str[:4])

for year in range(start_year, end_year + 1):
    print(f"{year}...", end=' ', flush=True)
    
    year_start = f"{year}0101"
    year_end = f"{year}1231"
    
    try:
        df_year = ak.get_futures_daily(start_date=year_start, end_date=year_end, market="SHFE")
        if not df_year.empty:
            df_variety = df_year[df_year['variety'] == target_variety].copy()
            if not df_variety.empty:
                df_variety['date'] = pd.to_datetime(df_variety['date'])
                all_data_frames.append(df_variety)
    except Exception:
        pass  # 静默失败，继续下一年
    
    time.sleep(0.3)

print("\n数据获取完毕，正在处理...")

if not all_data_frames:
    print("错误：未获取到任何有效数据。")
else:
    # 合并、排序并筛选至精确日期范围
    df_combined = pd.concat(all_data_frames, ignore_index=True)
    df_combined.sort_values(by='date', inplace=True)
    
    final_start = pd.Timestamp(start_date_str)
    final_end = pd.Timestamp(end_date_str)
    df_final = df_combined[(df_combined['date'] >= final_start) & 
                           (df_combined['date'] <= final_end)].copy()
    
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # 最终确认信息
    print("=" * 50)
    print(f"数据已保存！")
    print(f"文件位置：{output_path}")
    print(f"数据行数：{len(df_final)}")
    print(f"日期范围：{df_final['date'].min().date()} 至 {df_final['date'].max().date()}")
    print("=" * 50)