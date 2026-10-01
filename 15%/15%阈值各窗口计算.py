import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import genpareto, kstest
import os
import warnings
import random

from statsmodels.tsa.stattools import adfuller

random.seed(57)
np.random.seed(57)
warnings.filterwarnings('ignore')

# ====================== 配置 ======================
DATA_DIR = r"D:\基于因果尾部系数的金属期货极端风险方向性传导与预警研究"
OUTPUT_DIR = r"D:\基于因果尾部系数的金属期货极端风险方向性传导与预警研究\15%阈值"
EXTREME_DIR = OUTPUT_DIR   # 极端事件CSV所在目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_METALS = ['PB', 'CU', 'ZN', 'AL', 'NI', 'AG', 'AU', 'RB', 'SN']

# 20 个窗口：起始均为 1，终点从 1 到 20
WINDOWS_TO_TEST = [(1, i) for i in range(2, 21)]

try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


# ====================== 数据加载 ======================
def load_price_data():
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
    return pd.DataFrame(returns_dict)


# ====================== 读取极端事件 ======================
def load_extreme_events():
    extreme_events = {}
    for name in ALL_METALS:
        fname = f'SHFE_{name}_Extreme_Events_Threshold_Method_TrainBased.csv'
        path = os.path.join(EXTREME_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"未找到极端事件文件: {path}\n"
                f"请先运行生成极端事件的脚本（TAU=0.15）。"
            )
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        extreme_events[(name, 'left')] = df['extreme_left'].astype(int)
        extreme_events[(name, 'right')] = df['extreme_right'].astype(int)
    return extreme_events


# ====================== ADF 单位根检验 + Hill 尾指数 ======================
def hill_estimate(data, tail='right', k_ratio=0.15):
    n = len(data)
    k = int(n * k_ratio)
    if k < 10:
        return np.nan, np.nan, np.nan

    if tail == 'right':
        sorted_data = np.sort(data)
        tail_data = sorted_data[-k:]
        threshold = sorted_data[-k-1] if n > k else sorted_data[0]
        if np.min(tail_data) <= 0:
            shift = -np.min(tail_data) + 1e-8
            tail_data = tail_data + shift
            threshold = threshold + shift
    else:
        sorted_data = np.sort(data)
        tail_data = -sorted_data[:k]
        threshold = -sorted_data[k] if n > k else -sorted_data[-1]
        if np.min(tail_data) <= 0:
            shift = -np.min(tail_data) + 1e-8
            tail_data = tail_data + shift
            threshold = threshold + shift

    if threshold <= 0:
        threshold = np.min(tail_data) * 0.99

    logs = np.log(tail_data / threshold)
    alpha = 1.0 / np.mean(logs)
    se = alpha / np.sqrt(k)
    ci_lower = alpha - 1.96 * se
    ci_upper = alpha + 1.96 * se
    return alpha, ci_lower, ci_upper


def run_descriptive_tests(returns):
    results = []
    for name in returns.columns:
        data = returns[name].dropna().values
        if len(data) < 20:
            continue

        adf_stat, adf_p, _, _, crit_values, _ = adfuller(data, autolag='AIC')
        left_alpha, left_ci_low, left_ci_high = hill_estimate(data, tail='left', k_ratio=0.15)
        right_alpha, right_ci_low, right_ci_high = hill_estimate(data, tail='right', k_ratio=0.15)

        results.append({
            '品种': name,
            'ADF统计量': adf_stat,
            'ADF-p值': f"{adf_p:.10e}",
            '左尾指数': left_alpha,
            '左尾95%CI下界': left_ci_low,
            '左尾95%CI上界': left_ci_high,
            '右尾指数': right_alpha,
            '右尾95%CI下界': right_ci_low,
            '右尾95%CI上界': right_ci_high,
        })

    df = pd.DataFrame(results)
    print("\nADF 单位根检验与 Hill 尾指数估计结果：")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUTPUT_DIR, 'ADF_Hill_检验结果.csv'), index=False, encoding='utf-8-sig')
    return df


# ====================== CTC 统计量 ======================
def ctc_window_statistic(X, Y, start_lag, end_lag, v=0.5):
    n = len(X)
    k = int(np.floor(n ** v))
    threshold = np.sort(X)[-k]
    extreme_indices = np.where(X >= threshold)[0]
    stat_sum = 0
    valid = 0
    for idx in extreme_indices:
        start = idx + start_lag
        end = idx + end_lag
        if start < n:
            window_end = min(end, n-1)
            window_max = np.max(Y[start:window_end+1])
            cdf = np.mean(Y <= window_max)
            stat_sum += cdf
            valid += 1
    return stat_sum / valid if valid > 0 else 0.0


def subsampling_pvalue(X, Y, start_lag, end_lag, v=0.5, subsample_ratio=0.3, n_subsample=200):
    n = len(X)
    m = int(n * subsample_ratio)
    orig_stat = ctc_window_statistic(X, Y, start_lag, end_lag, v)
    max_start = n - m
    if max_start <= 0:
        return orig_stat, 1.0, 0.0
    possible_starts = list(range(max_start + 1))
    if n_subsample > len(possible_starts):
        n_subsample = len(possible_starts)
    selected_starts = np.random.choice(possible_starts, size=n_subsample, replace=False)
    subsample_stats = []
    for start in selected_starts:
        X_sub = X[start:start+m]
        Y_sub = Y[start:start+m]
        stat_sub = ctc_window_statistic(X_sub, Y_sub, start_lag, end_lag, v)
        subsample_stats.append(stat_sub)
    subsample_stats = np.array(subsample_stats)
    p_value = (np.sum(subsample_stats >= orig_stat) + 1) / (n_subsample + 1)
    critical = np.quantile(subsample_stats, 0.95)
    return orig_stat, p_value, critical


# ====================== 无条件事件基准 ======================
def calculate_unconditional_baseline(effect_extreme_test, window_size):
    T0 = len(effect_extreme_test)
    if T0 <= window_size:
        return 0.0
    total_starts = T0 - window_size
    hit_count = 0
    for t in range(total_starts):
        if np.any(effect_extreme_test[t+1 : t+window_size+1] == 1):
            hit_count += 1
    return hit_count / total_starts


# ====================== 规则生成 ======================
def get_cross_rules(effect_metal):
    cause_metals = [m for m in ALL_METALS if m != effect_metal]
    rules = []
    for cause in cause_metals:
        for cause_tail in ['left', 'right']:
            for effect_tail in ['left', 'right']:
                rule_name = f"{cause}_{cause_tail}_{effect_metal}_{effect_tail}"
                rule_desc = f"{cause}{'左尾' if cause_tail=='left' else '右尾'}→{effect_metal}{'左尾' if effect_tail=='left' else '右尾'}"
                rules.append({
                    'name': rule_name,
                    'cause': f"{cause}_{cause_tail}",
                    'effect': f"{effect_metal}_{effect_tail}",
                    'desc': rule_desc
                })
    return rules


# ====================== 单条规则评估 ======================
def evaluate_rule_global(rule, returns, extreme_events, train_dates, test_dates,
                         tau=0.15, alpha=0.05, window=(1,20),
                         subsample_ratio=0.3, n_subsample=200):
    cause = rule['cause'].split('_')[0]
    cause_tail = rule['cause'].split('_')[1]
    effect_metal = rule['effect'].split('_')[0]
    effect_tail = rule['effect'].split('_')[1]

    X_train = returns.loc[train_dates, cause].dropna().values
    Y_train = returns.loc[train_dates, effect_metal].dropna().values
    X_test = returns.loc[test_dates, cause].dropna().values
    Y_test = returns.loc[test_dates, effect_metal].dropna().values

    if len(X_train) < 20:
        return False, None

    X_train_adj = X_train.copy()
    Y_train_adj = Y_train.copy()
    if cause_tail == 'left':
        X_train_adj = -X_train_adj
    if effect_tail == 'left':
        Y_train_adj = -Y_train_adj

    orig_stat, p_value, critical = subsampling_pvalue(
        X_train_adj, Y_train_adj, window[0], window[1], v=0.5,
        subsample_ratio=subsample_ratio, n_subsample=n_subsample
    )
    significant = p_value < alpha

    result = {
        '规则': rule['desc'],
        'CTC统计量': orig_stat,
        'p值': p_value,
        '临界值': critical,
        '显著': significant,
        '训练集样本数': len(X_train),
        '测试集样本数': len(X_test),
        '训练集起始': train_dates[0],
        '训练集结束': train_dates[-1],
        '测试集起始': test_dates[0],
        '测试集结束': test_dates[-1],
    }

    if not significant:
        return False, result

    cause_series = extreme_events[(cause, cause_tail)]
    effect_series = extreme_events[(effect_metal, effect_tail)]

    test_index = pd.DatetimeIndex(test_dates)
    cause_extreme_test = cause_series.reindex(test_index).fillna(0).astype(int).values
    effect_extreme_test = effect_series.reindex(test_index).fillna(0).astype(int).values

    a, b = window
    trigger_indices = np.where(cause_extreme_test == 1)[0]
    hits = 0
    delay_list = []
    for idx in trigger_indices:
        start = idx + a
        end = idx + b
        if start < len(test_dates):
            win_end = min(end, len(test_dates)-1)
            if np.any(effect_extreme_test[start:win_end+1] == 1):
                hits += 1
                for lag in range(a, b+1):
                    if idx+lag < len(test_dates) and effect_extreme_test[idx+lag] == 1:
                        delay_list.append(lag)
                        break
    total_warnings = len(trigger_indices)
    hit_rate = hits / total_warnings if total_warnings > 0 else 0
    avg_delay = np.mean(delay_list) if delay_list else 0

    window_size = b - a + 1
    unconditional_baseline = calculate_unconditional_baseline(effect_extreme_test, window_size)

    absolute_increment = hit_rate - unconditional_baseline
    relative_increment = hit_rate / unconditional_baseline if unconditional_baseline > 0 else np.nan

    result.update({
        '预警次数': total_warnings,
        '命中次数': hits,
        '命中率': hit_rate,
        '平均延迟': avg_delay,
        '测试集效果事件总次数': int(np.sum(effect_extreme_test)),
        '无条件基准': unconditional_baseline,
        '绝对增量': absolute_increment,
        '相对增量': relative_increment,
        '增量是否为正': bool(absolute_increment > 0)
    })
    return True, result


# ====================== 主程序 ======================
def main():
    print("="*70)
    print("多金属极端风险预警系统 (ADF + Hill，20个窗口，10%阈值)")
    print("="*70)

    ALPHA = 0.05
    SUBSAMPLE_RATIO = 0.3
    N_SUBSAMPLE = 200

    returns = load_price_data()
    print("数据加载完成。")

    extreme_events = load_extreme_events()
    print("极端事件标记加载完成（来自 10%阈值 目录的 CSV）。")

    common_dates = returns.dropna().index
    print(f"公共日期区间: {common_dates[0]} 至 {common_dates[-1]}, 共 {len(common_dates)} 个交易日")

    print("\n正在进行 ADF 单位根检验与 Hill 尾指数估计...")
    run_descriptive_tests(returns.loc[common_dates])

    n = len(common_dates)
    split_idx = int(n * 0.8)
    train_dates = common_dates[:split_idx]
    test_dates = common_dates[split_idx:]
    print(f"训练集: {train_dates[0]} ~ {train_dates[-1]}, 共 {len(train_dates)} 天")
    print(f"测试集: {test_dates[0]} ~ {test_dates[-1]}, 共 {len(test_dates)} 天")

    all_window_records = []

    for window in WINDOWS_TO_TEST:
        window_str = f"窗口_{window[0]}_{window[1]}"
        window_dir = os.path.join(OUTPUT_DIR, window_str)
        os.makedirs(window_dir, exist_ok=True)

        print("\n" + "="*90)
        print(f"开始处理窗口: {window}")
        print("="*90)

        window_significant = []

        for target in ALL_METALS:
            target_dir = os.path.join(window_dir, f"{target}_results")
            os.makedirs(target_dir, exist_ok=True)

            rules = get_cross_rules(target)
            all_results = []
            significant_rules_info = []

            for rule in rules:
                is_sig, res = evaluate_rule_global(
                    rule, returns, extreme_events, train_dates, test_dates,
                    tau=0.15, alpha=ALPHA, window=window,
                    subsample_ratio=SUBSAMPLE_RATIO,
                    n_subsample=N_SUBSAMPLE
                )
                if res is not None:
                    all_results.append(res)
                if is_sig:
                    significant_rules_info.append(res)

            if all_results:
                pd.DataFrame(all_results).to_csv(
                    os.path.join(target_dir, f"{target}_所有规则CTC检验结果.csv"),
                    index=False, encoding='utf-8-sig'
                )
            if significant_rules_info:
                pd.DataFrame(significant_rules_info).to_csv(
                    os.path.join(target_dir, f"{target}_显著规则结果.csv"),
                    index=False, encoding='utf-8-sig'
                )
                for r in significant_rules_info:
                    r_copy = dict(r)
                    r_copy['效果变量'] = target
                    r_copy['窗口'] = f"[{window[0]},{window[1]}]"
                    r_copy['窗口起点'] = window[0]
                    r_copy['窗口终点'] = window[1]
                    window_significant.append(r_copy)

        if not window_significant:
            print(f"窗口 {window}：未发现显著规则。")
            continue

        df_win = pd.DataFrame(window_significant)

        preferred_cols = [
            '窗口', '效果变量', '规则', 'CTC统计量', 'p值', '临界值',
            '命中率', '平均延迟', '无条件基准', '绝对增量', '相对增量',
            '预警次数', '命中次数', '测试集效果事件总次数', '测试集样本数'
        ]
        cols = [c for c in preferred_cols if c in df_win.columns]
        df_win = df_win[cols]

        print(f"\n窗口 {window}：共 {len(df_win)} 条显著规则")
        print("-"*90)
        print(df_win.to_string(index=False))

        win_out = os.path.join(window_dir, f"窗口_{window[0]}_{window[1]}_全部显著规则.csv")
        df_win.to_csv(win_out, index=False, encoding='utf-8-sig')
        print(f"\n该窗口汇总已保存至: {win_out}")

        all_window_records.extend(window_significant)

    print("\n" + "="*90)
    print("跨窗口汇总")
    print("="*90)

    if all_window_records:
        df_all = pd.DataFrame(all_window_records)
        df_all = df_all.sort_values(['窗口终点', '效果变量', '规则'])

        cols_all = [
            '窗口', '效果变量', '规则', 'CTC统计量', 'p值', '临界值',
            '命中率', '平均延迟', '无条件基准', '绝对增量', '相对增量',
            '预警次数', '命中次数', '测试集效果事件总次数', '测试集样本数'
        ]
        cols_all = [c for c in cols_all if c in df_all.columns]
        df_all = df_all[cols_all]

        all_out = os.path.join(OUTPUT_DIR, "跨窗口_全部显著规则汇总.csv")
        df_all.to_csv(all_out, index=False, encoding='utf-8-sig')

        print(f"\n共发现 {len(df_all)} 条显著规则，涉及 {df_all['规则'].nunique()} 个方向。")
        print(f"跨窗口汇总已保存至: {all_out}")

        print("\n按窗口统计：")
        summary = df_all.groupby('窗口').agg(
            显著规则数=('规则', 'count'),
            平均命中率=('命中率', 'mean'),
            平均绝对增量=('绝对增量', 'mean'),
            平均相对增量=('相对增量', 'mean')
        ).reset_index()
        print(summary.to_string(index=False))

        print("\n按方向统计（出现窗口数）：")
        direction_summary = df_all.groupby('规则').agg(
            显著窗口数=('窗口', 'count'),
            显著窗口列表=('窗口', lambda x: ', '.join(x)),
            平均命中率=('命中率', 'mean'),
            平均绝对增量=('绝对增量', 'mean'),
            平均相对增量=('相对增量', 'mean'),
            最小p值=('p值', 'min')
        ).reset_index().sort_values('显著窗口数', ascending=False)
        print(direction_summary.to_string(index=False))

        dir_out = os.path.join(OUTPUT_DIR, "跨窗口_方向汇总.csv")
        direction_summary.to_csv(dir_out, index=False, encoding='utf-8-sig')
        print(f"\n方向汇总已保存至: {dir_out}")

    else:
        print("所有窗口均未发现显著规则。")

    print("\n" + "="*90)
    print("全部分析完成！")
    print("="*90)


if __name__ == "__main__":
    main()