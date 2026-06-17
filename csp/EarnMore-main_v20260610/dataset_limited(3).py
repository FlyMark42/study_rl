import os
import shutil
import random

# ========== 配置 ==========
SOURCE_ROOT = '/root/autodl-tmp/datasets/csp_data'
TARGET_ROOT = SOURCE_ROOT + '_cut'
AUX_FOLDER = 'aux_stocks_files_cleaned'
FEATURE_FOLDER = 'features'
FILTERED_FILE = 'stocks_filtered_2021.txt'
STOCKS_CUT_FILE = 'stocks_cut.txt'
NUM_TOTAL_STOCKS = 1000  # 你可以根据需要修改总挑选数
# ===========================

# 1. 创建目标文件夹
os.makedirs(TARGET_ROOT, exist_ok=True)
os.makedirs(os.path.join(TARGET_ROOT, FEATURE_FOLDER), exist_ok=True)

# 2. 统计 aux_stocks_files 中所有股票
aux_dir_src = os.path.join(SOURCE_ROOT, AUX_FOLDER)
aux_dir_dst = os.path.join(TARGET_ROOT, AUX_FOLDER)
os.makedirs(aux_dir_dst, exist_ok=True)

aux_stocks = set()
for fname in os.listdir(aux_dir_src):
    if fname.endswith('.txt'):
        with open(os.path.join(aux_dir_src, fname), 'r') as f:
            for line in f:
                stock = line.strip()
                if stock:
                    aux_stocks.add(stock)
        # 同时复制该 txt 文件到目标文件夹
        shutil.copy(os.path.join(aux_dir_src, fname), os.path.join(aux_dir_dst, fname))

print(f"从 {AUX_FOLDER} 中提取股票共 {len(aux_stocks)} 个。")

# 3. 随机挑选 N 支股票（包含 aux_stocks 中所有股票）
filtered_path = os.path.join(SOURCE_ROOT, FILTERED_FILE)
with open(filtered_path, 'r') as f:
    all_stocks = [line.strip() for line in f if line.strip()]

# 补充数量 = 目标数 - 必含的 aux_stocks
remaining_stocks = list(set(all_stocks) - aux_stocks)
num_to_sample = NUM_TOTAL_STOCKS - len(aux_stocks)

if num_to_sample < 0:
    raise ValueError(f"aux_stocks 数量 {len(aux_stocks)} 已超过目标总数 {NUM_TOTAL_STOCKS}，请增加目标数。")

random_selected = random.sample(remaining_stocks, num_to_sample)
final_stocks = sorted(aux_stocks.union(random_selected))

# 写入新 stocks_cut.txt 文件
stocks_cut_path = os.path.join(TARGET_ROOT, STOCKS_CUT_FILE)
with open(stocks_cut_path, 'w') as f:
    for stock in final_stocks:
        f.write(stock + '\n')

print(f"已生成 {len(final_stocks)} 支股票，保存至 {STOCKS_CUT_FILE}")

# 4. 复制对应 features 下的 csv 文件
features_src = os.path.join(SOURCE_ROOT, FEATURE_FOLDER)
features_dst = os.path.join(TARGET_ROOT, FEATURE_FOLDER)

missing_count = 0
for stock in final_stocks:
    src_file = os.path.join(features_src, stock + '.csv')
    dst_file = os.path.join(features_dst, stock + '.csv')
    if os.path.exists(src_file):
        shutil.copy(src_file, dst_file)
    else:
        print(f" 缺失 csv 文件: {stock}.csv")
        missing_count += 1

print(f"完成复制 {len(final_stocks) - missing_count} 个 csv 文件至 {features_dst}")
if missing_count > 0:
    print(f"共缺失 {missing_count} 个 csv 文件，建议检查源 features 文件夹。")
