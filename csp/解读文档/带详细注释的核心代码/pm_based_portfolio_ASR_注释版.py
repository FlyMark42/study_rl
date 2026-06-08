# -*- coding: utf-8 -*-
# ============================================================================
#  文件: pm/environment/pm_based_portfolio_ASR.py  （带详细中文注释版）
#  作用: 强化学习"环境(Environment)" —— 对应论文第 3.2 节 MDP 建模。
#        它实现了标准 gym 接口 reset()/step(), 负责:
#          1) 提供"状态"(各股票近 days 天的特征);
#          2) 接收智能体的"动作"(投资组合权重);
#          3) 推进一天, 计算"收益"和"奖励", 返回新状态。
#  注意: ASR = Adjusted Sharpe Reward, 本环境用"夏普比率"当奖励,
#        这与论文原版(用市值变化 V_t - V_{t-1} 当奖励)不同, 是本仓库的改动。
# ============================================================================
import numpy as np
from typing import List, Any
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd
import gym
from quantstats import stats as qs_stats
from pm.registry import ENVIRONMENT


@ENVIRONMENT.register_module()   # 注册进 ENVIRONMENT 电话簿, 名字 "EnvironmentASR"
class EnvironmentASR(gym.Env):
    def __init__(self,
                 mode: str = "train",                 # train / val / test
                 dataset: Any = None,                 # 数据集对象(PortfolioManagementDataset)
                 if_norm: bool = True,                # 是否标准化特征
                 if_norm_temporal: bool = True,       # 是否也标准化时序特征
                 scaler: List[StandardScaler] = None, # 标准化器(验证/测试要复用训练集的)
                 days: int = 10,                      # 状态回看的天数 D
                 start_date: str = None,
                 end_date: str = None,
                 initial_amount: int = 1e3,           # 初始资金 V_0
                 transaction_cost_pct: float = 1e-3   # 交易成本比例(本实现里实际未扣, 留作扩展)
                 ):
        super(EnvironmentASR, self).__init__()

        self.mode = mode
        self.dataset = dataset
        self.if_norm = if_norm
        self.if_norm_temporal = if_norm_temporal
        self.scaler = scaler
        self.days = days
        self.start_date = start_date
        self.end_date = end_date
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct

        if end_date is not None:
            assert end_date > start_date, "结束日期必须晚于开始日期"

        # 从数据集取出: 股票列表、股票↔id 映射、子池定义(aux_stocks)
        self.stocks = self.dataset.stocks
        self.stocks2id = self.dataset.stocks2id
        self.id2stocks = self.dataset.id2stocks
        self.aux_stocks = self.dataset.aux_stocks   # ← 各"可定制子池 CSP"的定义(含掩码)

        self.features_name = self.dataset.features_name
        self.prices_name = ['OPEN', 'HIGH', 'LOW', 'CLOSE']  # 用收盘价算收益
        self.temporals_name = self.dataset.temporals_name
        self.labels_name = self.dataset.labels_name
        self.stocks_df = []
        self._ret_history = []     # 收益历史(算夏普用)
        self._date_history = []    # 日期历史

        prices = []
        if if_norm:
            print("normalize datasets")
            if self.mode == "train":
                # 训练模式: 现场 fit 一个标准化器, 并保存(验证/测试要复用)
                self.scaler = []
                for df in self.dataset.stocks_df:
                    # 按日期切片
                    if end_date is not None:
                        df = df.loc[start_date:end_date].copy()
                    else:
                        df = df.loc[start_date:].copy()
                    # 整理价格列(大写列名)
                    df[self.prices_name] = df[[name.lower() for name in self.prices_name]]
                    price_df = df[self.prices_name]
                    prices.append(price_df.values)
                    # 标准化特征(StandardScaler: 减均值除标准差)
                    scaler = StandardScaler()
                    if self.if_norm_temporal:
                        df[self.features_name + self.temporals_name] = scaler.fit_transform(
                            df[self.features_name + self.temporals_name])
                    else:
                        df[self.features_name] = scaler.fit_transform(df[self.features_name])
                    self.scaler.append(scaler)
                    self.stocks_df.append(df)
            else:
                # 验证/测试模式: 必须用训练集 fit 好的 scaler 来 transform (防止数据泄漏)
                assert self.scaler is not None, "验证/测试模式必须传入训练好的 scaler"
                for index, df in enumerate(self.dataset.stocks_df):
                    if end_date is not None:
                        df = df.loc[start_date:end_date].copy()
                    else:
                        df = df.loc[start_date:].copy()
                    df[self.prices_name] = df[[name.lower() for name in self.prices_name]]
                    price_df = df[self.prices_name]
                    prices.append(price_df.values)
                    scaler = self.scaler[index]
                    if self.if_norm_temporal:
                        df[self.features_name + self.temporals_name] = scaler.transform(
                            df[self.features_name + self.temporals_name])
                    else:
                        df[self.features_name] = scaler.transform(df[self.features_name])
                    self.stocks_df.append(df)
        else:
            print("no normalize datasets")

        # 把所有股票的特征堆成一个大数组: (股票数, 总天数, 特征数)
        self.features = []
        for df in self.stocks_df:
            df = df[self.features_name + self.temporals_name]
            self.features.append(df.values)
        self.features = np.stack(self.features)

        # 价格数组: (股票数, 总天数, 4)  4=OHLC
        self.prices = np.stack(prices)

        # 标签数组(本环境没直接用, 留作扩展)
        self.labels = []
        for df in self.stocks_df:
            df = df[self.labels_name]
            self.labels.append(df.values)
        self.labels = np.stack(self.labels)

        print("features shape {}, prices shape {}, labels shape {}, num days {}".format(
            self.features.shape, self.prices.shape, self.labels.shape, self.features.shape[1]))

        self.num_days = self.features.shape[1]          # 总天数
        self.dates = pd.to_datetime(self.stocks_df[0].index)
        print("dates range:", self.dates[0], "to", self.dates[-1])

        # 初始化"当前是第几天"
        if self.mode == "train":
            # 训练时随机选起点(数据增强, 让模型见到不同起始行情); 上限是前 3/4 处
            self.day = random.randint(self.days - 1, 3 * (self.num_days // 4))
        else:
            self.day = self.days - 1  # 验证/测试从最早能凑齐 days 天的位置开始

    def get_current_date(self):
        """返回当前交易日的日期。"""
        return self.stocks_df[0].index[self.day]

    def reset(self, *, seed=None, options=None):
        """
        重置环境到一个新回合(episode)的开始。返回初始状态和 info。
        对应论文 MDP 的"初始状态 s_0"。
        """
        self._ret_history = []
        self._date_history = []

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # 重新选起始日
        if self.mode == "train":
            self.day = random.randint(self.days - 1, 3 * (self.num_days // 4))
        else:
            self.day = self.days - 1

        # 状态 = 所有股票 [当前日往前数 days 天] 的特征, 形状 (股票数, days, 特征数)
        state = self.features[:, self.day - self.days + 1: self.day + 1, :]
        self.state = state
        self.protfolio_value = self.initial_amount   # 重置组合市值为初始资金(注:原代码拼写为 protfolio)

        info = {"current_date": self.get_current_date()}
        return state, info

    def get_prices(self):
        """取当前交易日所有股票的 OHLC 四个价格, 各返回一个数组。"""
        prices = self.prices[:, self.day, :]
        o, h, l, c = prices[:, 0], prices[:, 1], prices[:, 2], prices[:, 3]
        return o, h, l, c

    def step(self, action: np.array = None):
        """
        ★ 环境的核心: 执行一个动作, 推进一天 ★
        输入: action = 投资组合权重 (长度 N+1, 第0位是现金), 和为1。
        返回: (下一状态, 奖励, terminated, truncated, info)  —— gym 新版接口。
        """
        state = self.state
        weights = action.flatten()    # 拉平成一维: [现金, 股票1, 股票2, ...]

        # 记下"今天"的价格
        pre_o, pre_h, pre_l, pre_c = self.get_prices()
        # 推进到"明天"
        self.day += 1
        done = self.day >= self.num_days - 1   # 到数据末尾就结束
        post_o, post_h, post_l, post_c = self.get_prices()

        # ★ 组合收益率 = Σ (每只股票的收盘价涨跌幅 × 该股票权重) ★
        #   weights[1:] 是股票权重(跳过第0位现金); 现金部分不涨不跌
        portfolio_ret = np.sum(((post_c - pre_c) / pre_c) * weights[1:])

        # ★ 更新组合市值 ★  对应论文公式(1):
        #   新市值 = 现金部分(不变) + 股票部分(按组合收益率增长)
        new_value = (1 - weights[0]) * self.protfolio_value * (1 + portfolio_ret) \
                    + weights[0] * self.protfolio_value
        self.protfolio_value = new_value

        # 记录当期收益和日期(用来算夏普)
        self._ret_history.append(portfolio_ret)
        current_date = self.get_current_date()
        self._date_history.append(current_date)

        # ★ 计算奖励 = 截至当前的"年化夏普比率"(本仓库特色, 非论文原版) ★
        #   论文原版奖励 = V_t - V_{t-1}(市值变化); 这里换成夏普, 更看重"风险调整后收益"
        date_index = pd.to_datetime(self._date_history)
        rets = pd.Series(self._ret_history, index=date_index)
        reward = qs_stats.sharpe(rets, rf=0.02)   # rf=0.02 是无风险利率

        # 数值保护: 夏普可能是 nan/inf(序列太短时), 兜底为 0; 并把奖励截断为非负
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        reward = max(reward, 0.0)

        info = {
            "state": state,
            "action": action,
            "portfolio_ret": portfolio_ret,        # 当期收益率
            "portfolio_value": self.protfolio_value,  # 当前市值
            "date": current_date
        }

        # 下一状态
        next_state = self.features[:, self.day - self.days + 1: self.day + 1, :]
        self.state = next_state

        terminated = done    # 是否自然结束(到末尾)
        truncated = False     # 是否被强制截断(本环境不用)

        return next_state, reward, terminated, truncated, info

# ============================================================================
#  小结(对照论文 MDP 五元组):
#    - 状态 S : 所有股票近 days 天的特征 (reset/step 返回的 state)
#    - 动作 A : 投资组合权重向量 weights (和为1, 含现金)
#    - 转移 T : self.day += 1, 用真实历史行情推进(确定性, 无随机)
#    - 奖励 R : 本版=夏普比率; 论文原版=市值变化
#    - 折扣 γ : 不在环境里, 在智能体 AgentMaskSAC 中(gamma=0.99)
#
#  "可定制股票池(CSP)"如何体现? —— 见 dataset 里的 aux_stocks[i]["mask"]:
#    验证时, 为每个子池建一个环境, 并在智能体决策时传入该子池的 mask,
#    把池外股票的权重压到≈0(见 helpers.py 的重加权)。
# ============================================================================
