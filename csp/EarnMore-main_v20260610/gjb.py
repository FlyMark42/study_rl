import os
import pandas as pd

ROOT = "/root/autodl-tmp/datasets/csp_data/raw_filtered"
bad_files = []

for fname in os.listdir(ROOT):
    if not fname.endswith('.csv'):
        continue
    fpath = os.path.join(ROOT, fname)
    df = pd.read_csv(fpath)
    if df.isna().any().any():
        bad_files.append(fname)

if bad_files:
    print("❌ 以下文件仍含 NaN：")
    print(len(bad_files))
    for f in bad_files:
        print(f" - {f}")
else:
    print("✅ 所有文件均不含 NaN。")
