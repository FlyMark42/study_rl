import os
import pandas as pd
from collections import Counter
import shutil

# === 路径配置 ===
YEAR = 2023
# START_DATE = f"{YEAR}-01-01"
START_DATE = f"{YEAR}-01-01"

# SRC_FOLDER = "/root/autodl-tmp/datasets/csp_data/raw/"
# DST_FOLDER = "/root/autodl-tmp/datasets/csp_data"
SRC_FOLDER = "/root/autodl-tmp/datasets/csp_817_data/raw/"
DST_FOLDER = "/root/autodl-tmp/datasets/csp_817_data"
COPY_FOLDER = os.path.join(DST_FOLDER, "raw_filtered")
AUX_DIR = os.path.join(DST_FOLDER, "aux_stocks_files")
CLEANED_AUX_DIR = os.path.join(DST_FOLDER, "aux_stocks_files_cleaned")
FILTERED_FILE = os.path.join(DST_FOLDER, f"stocks_filtered_{YEAR}.txt")
os.makedirs(COPY_FOLDER, exist_ok=True)
os.makedirs(CLEANED_AUX_DIR, exist_ok=True)

# 需要删除的列
EXCLUDE_COLUMNS = [
    'daily_ret_6m', 'mom_3m_1m', 'mom_3m', 'daily_ret_3m',
    'mom_1m', 'daily_ret_1m', 'close20_close', 'price_delta_indicator_9'
]

def process_year(year):
    survivors = []
    row_counts = {}

    files = [f for f in os.listdir(SRC_FOLDER) if f.lower().endswith('.csv')]
    if not files:
        print(f"❌ No CSV files under {SRC_FOLDER}")
        return []

    for fname in files:
        try:
            path = os.path.join(SRC_FOLDER, fname)
            df = pd.read_csv(path, parse_dates=['trade_date'])

            # 去除指定列（如果存在）
            df = df.drop(columns=[col for col in EXCLUDE_COLUMNS if col in df.columns], errors='ignore')

            # 保证 trade_date 列存在
            if 'trade_date' not in df.columns:
                print(f"❌ {fname} 缺少 'trade_date' 列，跳过")
                continue

            # 按时间裁剪
            df_cut = df[df['trade_date'] >= START_DATE]

            # NaN 检查（任何行或列含 NaN 都剔除）
            if not df_cut.isna().any().any():
                survivors.append(os.path.splitext(fname)[0])
                row_counts[fname] = len(df_cut)
            else:
                print(f"❌ {fname} 剪裁后存在 NaN，跳过")
        except Exception as e:
            print(f"[!] 处理 {fname} 出错: {e}")

    print(f"\n✅ 存活股票数: {len(survivors)}")
    for fname, cnt in row_counts.items():
        print(f"{fname}: {cnt} rows")

    # 保存 survivors 名单
    with open(FILTERED_FILE, 'w', encoding='utf-8') as f:
        for stock in sorted(survivors):
            f.write(stock + '\n')
    print(f"➡️ 已保存至: {FILTERED_FILE}")

    # 拷贝裁剪过的 CSV 文件
    for stock in survivors:
        try:
            raw_path = os.path.join(SRC_FOLDER, stock + ".csv")
            df = pd.read_csv(raw_path, parse_dates=['trade_date'])
            df = df.drop(columns=[col for col in EXCLUDE_COLUMNS if col in df.columns], errors='ignore')
            df_cut = df[df['trade_date'] >= START_DATE]
            df_cut.to_csv(os.path.join(COPY_FOLDER, stock + ".csv"), index=False)
        except Exception as e:
            print(f"复制 {stock}.csv 时出错: {e}")

    return set(survivors)

def clean_aux_files(survivors: set):
    print(f"\n🧹 清洗 aux_stocks_files 中...")
    for fname in os.listdir(AUX_DIR):
        if not fname.endswith(".txt"):
            continue
        in_path = os.path.join(AUX_DIR, fname)
        out_path = os.path.join(CLEANED_AUX_DIR, fname)

        with open(in_path, "r") as f:
            stocks = [line.strip() for line in f if line.strip()]

        kept = [s for s in stocks if s in survivors]
        removed = [s for s in stocks if s not in survivors]

        with open(out_path, "w") as f:
            for s in kept:
                f.write(s + "\n")

        print(f"📄 {fname}: 保留 {len(kept)} 支，移除 {len(removed)} 支")

    print(f"\n✅ 清洗完成，保存至: {CLEANED_AUX_DIR}\n")

def main():
    survivors = process_year(YEAR)
    if survivors:
        clean_aux_files(survivors)

if __name__ == '__main__':
    main()
