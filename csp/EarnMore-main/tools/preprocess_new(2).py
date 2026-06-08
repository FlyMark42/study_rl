import os
from glob import glob
import pathlib
import pandas as pd

START_DATE = '2023-01-01'
END_DATE = '2025-09-30'

def add_temporal_and_target(df):
    # ensure Date column as datetime index
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    # add temporal features
    df['weekday'] = df.index.weekday
    df['day'] = df.index.day
    df['month'] = df.index.month
    # compute next-day return and movement label
    df['ret1'] = df['close'].pct_change(1).shift(-1)
    df['mov1'] = (df['ret1'] > 0).astype(int)
    return df

def preprocess_and_filter(df):
    # Keep rows 60 to -1 (dropping first 60 and last row)
    #df = df.iloc[60:-1].reset_index()
    df = df.reset_index()
    # Filter by date range
    df = df[df['trade_date'] >= START_DATE]
    df = df[df['trade_date'] <= END_DATE]
    # Reset index after filtering
    df = df.reset_index(drop=True)
    return df


def main(inpath, outpath):
    os.makedirs(outpath, exist_ok=True)
    for path in glob(os.path.join(inpath, '*.csv')):
        name = os.path.basename(path)
        df = pd.read_csv(path)
        # rename required OHLCV columns if they differ
        if df.isna().any().any():
            raise ValueError(f"❌ 检测到 NaN 值，终止执行！出错文件: {name}")

        df.rename({
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Vol': 'volume'
        }, axis=1, inplace=True)
        # add temporal features and targets
        df = add_temporal_and_target(df)
        # apply original crop and date filtering logic
        df = preprocess_and_filter(df)
        # save
        df.to_csv(os.path.join(outpath, name), index=True)

if __name__ == '__main__':
    # ROOT = '/root/autodl-tmp/datasets/csp_data'
    ROOT = '/root/autodl-tmp/datasets/csp_817_data'
    raw = os.path.join(ROOT, 'raw_filtered')
    feats = os.path.join(ROOT, 'features')
    main(raw, feats)
