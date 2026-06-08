import numpy as np
import pandas as pd
import quantstats as qs

from tools.tools_utils import check_operating_system


def ARR(ret):
    """
    年化收益率 (CAGR)
    """
    series = pd.Series(ret)
    return qs.stats.cagr(series, periods=365)

def VOL(ret):
    """
    年化波动率
    """
    series = pd.Series(ret)
    return qs.stats.volatility(series)

def DD(ret):
    res = np.std(ret[np.where(ret<0, True, False)])
    return res

def MDD(ret):
    """
    最大回撤 (max drawdown)
    """
    series = pd.Series(ret)
    return qs.stats.max_drawdown(series)

def SR(ret):
    """
    年化夏普比率 (Sharpe ratio)
    """
    series = pd.Series(ret)
    return qs.stats.sharpe(series)

def ASR(ret):
    """
    Annualized Sharpe Ratio —— 与 SR 等价
    """
    series = pd.Series(ret)
    return qs.stats.sharpe(series, rf=0.02)

def CR(ret):
    """
    Calmar 比率 (CAGR / 最大回撤)
    """
    series = pd.Series(ret)
    return qs.stats.calmar(series)

def SOR(ret):
    """
    Sortino 比率
    """
    series = pd.Series(ret)
    return qs.stats.sortino(series)

def IR(ret):
    """
    信息比率 (Information Ratio)，相对于 benchmark_returns.csv
    假设 benchmark_returns.csv 第一列是日期，第二列是基准日收益率
    """
    series = pd.Series(ret)
    strCSV="/root/autodl-tmp/datasets/benchmark_returns.csv"
    if(check_operating_system()=="Windows"):
        strCSV="./datasets/benchmark_returns.csv"
    # 读取基准
    bench = pd.read_csv(
        strCSV,
        index_col=0,
        parse_dates=True
    ).squeeze("columns")
    # 对齐：取最后 len(series) 天
    bench = bench.iloc[-len(series) :]
    return qs.stats.information_ratio(series, bench)
