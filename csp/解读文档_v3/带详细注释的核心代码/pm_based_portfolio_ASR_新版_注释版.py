# -*- coding: utf-8 -*-
"""
==========================================================================================
EarnMore 新版环境 EnvironmentASR —— 逐行详细中文注释版（学习用）
对应原文件：EarnMore-main_v20260610/pm/environment/pm_based_portfolio_ASR.py
==========================================================================================

【这份文件是新版相对旧版唯一的重大逻辑改动】
- 旧版环境（EarnMore-main，约 213 行）：只做"权重 → 组合日收益 → 夏普奖励"，是一个理想化的、
  无摩擦的简化环境（不考虑手续费细节、不考虑 A 股交易规则）。
- 新版环境（本文件，约 620 行）：把作者另一套交易系统 "AGDRL_calmar_v2" 的**真实 A 股交易执行逻辑**
  移植进来，变成一个**贴近中国 A 股市场的高保真交易模拟器**。新增：
    ① t+N 调仓（每 N 天才真正换仓，其余日子只持有）
    ② T+1 规则（当天买入的股票当天不能卖）
    ③ 涨跌停限制（根据股票代码推断涨跌停幅度，封板时禁买/禁卖）
    ④ 整手约束（A 股按"手"=100 股交易）
    ⑤ 完整交易成本（佣金+最低5元、印花税仅卖出、过户费仅沪市、滑点）
    ⑥ 现金留存（保留 1% 现金缓冲，避免满仓买不进）
    ⑦ 股数级账本（显式记录 shares_held / shares_frozen / cash，而非只记权重）
- 奖励仍然是：年化夏普比率（ASR = Annualized Sharpe Reward）。

【与论文的关系】
- 论文原版奖励是"市值变化 V_t - V_{t-1}"；这套魔改版改用"夏普比率"做奖励（更看重风险调整收益）。
- 论文的 MDP 状态/动作定义不变：状态仍是 [股票数, 天数, 特征数]，动作仍是 [现金, 各股票权重]。
- 论文里没有 A 股交易细节（论文用美股 SP500/DJ30）；这些细节是工程落地时为对接 A 股加的。

【阅读建议】先看 step() 总入口（第 ~598 行）理解主循环骨架，再回头看各 _xxx 辅助方法。
==========================================================================================
"""

import numpy as np
from typing import List, Any
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd
import gym
from quantstats import stats as qs_stats        # quantstats：金融指标库，这里用它算夏普比率
from pm.registry import ENVIRONMENT             # 注册器：把本类登记进"电话簿"，配置里用字符串 "EnvironmentASR" 即可创建


@ENVIRONMENT.register_module()                  # ← 登记进注册器。配置 environment=dict(type="EnvironmentASR", ...) 就会造它
class EnvironmentASR(gym.Env):                  # 继承 gym.Env，实现标准的 reset()/step() 接口，便于被 gym 向量环境包装
    def __init__(self,
                 mode: str = "train",            # "train"/"val"/"test"：训练模式会随机选起始日做数据增强
                 dataset: Any = None,            # 数据集对象（PortfolioManagementDataset），提供股票列表、特征、子池掩码等
                 if_norm: bool = True,           # 是否对特征做标准化（StandardScaler）
                 if_norm_temporal: bool = True,  # 是否把"时序特征(日期等)"也一起标准化
                 scaler: List[StandardScaler] = None,  # 验证/测试时复用训练集拟合好的 scaler（避免数据泄漏）
                 days: int = 10,                 # 回看窗口长度：状态取最近 days 天（论文 D=10）
                 start_date: str = None,
                 end_date: str = None,
                 initial_amount: int = 1e5,      # 初始本金（新版默认 10 万元；旧版配置里曾是 1e3）
                 transaction_cost_pct: float = 1e-3,  # 旧版用的笼统交易成本率（新版改用下面更细的费用项，此参数基本不再主导）
                 # ============ 以下是新版新增的 A 股交易参数（配置里若不传则全部走这些默认值） ============
                 rebalance_period: int = 5,      # t+N 调仓周期：每 5 个交易日才真正调一次仓，其余日子只"持有"
                 enable_t1_rule: bool = True,     # T+1 规则开关：当天买入的股票当天不可卖出（A 股规则）
                 enable_price_limit: bool = True, # 涨跌停限制开关：封涨停不能买、封跌停不能卖
                 enable_lot_size: bool = True,    # 整手约束开关：买卖股数必须是 lot_size 的整数倍
                 lot_size: int = 100,             # 一手 = 100 股（A 股规则）
                 commission_rate: float = 0.0000855,  # 佣金费率（万分之 0.855），买卖都收
                 commission_min: float = 5.0,     # 佣金最低 5 元（不足 5 元按 5 元收）
                 stamp_tax_rate: float = 0.0005,  # 印花税（万分之 5），**仅卖出时收**
                 sh_transfer_fee_rate: float = 0.00002,  # 沪市过户费（万分之 0.2），**仅沪市股票收**
                 sz_transfer_fee_rate: float = 0.0,        # 深市过户费（这里设为 0）
                 slip_perc: float = 0.001,        # 滑点比例：实际成交价比理论价差 0.1%（买入更贵、卖出更便宜）
                 enable_slippage: bool = True,    # 滑点开关
                 cash_reserve_ratio: float = 0.01 # 现金留存比例：永远保留 1% 现金，避免满仓后买不进/算账穿仓
                 ):
        super(EnvironmentASR, self).__init__()

        # ---- 把所有配置参数存为成员变量 ----
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

        # 交易参数（A 股规则）
        self.rebalance_period = rebalance_period
        self.enable_t1_rule = enable_t1_rule
        self.enable_price_limit = enable_price_limit
        self.enable_lot_size = enable_lot_size
        self.lot_size = lot_size
        self.commission_rate = commission_rate
        self.commission_min = commission_min
        self.stamp_tax_rate = stamp_tax_rate
        self.sh_transfer_fee_rate = sh_transfer_fee_rate
        self.sz_transfer_fee_rate = sz_transfer_fee_rate
        self.slip_perc = slip_perc
        self.enable_slippage = enable_slippage
        self.cash_reserve_ratio = cash_reserve_ratio

        if end_date is not None:
            assert end_date > start_date, "start date {}, end date {}, end date should be greater than start date".format(
                start_date, end_date)

        # ---- 从 dataset 取出股票池信息 ----
        self.stocks = self.dataset.stocks            # 全局股票池 GSP 的股票代码列表
        self.n_stocks = len(self.stocks)             # 股票总数 N
        self.stocks2id = self.dataset.stocks2id      # 股票代码 → 索引
        self.id2stocks = self.dataset.id2stocks      # 索引 → 股票代码
        self.aux_stocks = self.dataset.aux_stocks    # 各 CSP 子池定义（含 mask）

        # ---- 特征/价格/标签的列名 ----
        self.features_name = self.dataset.features_name      # 技术指标等特征列名
        self.prices_name = ['OPEN', 'HIGH', 'LOW', 'CLOSE']  # 价格列：开/高/低/收
        self.temporals_name = self.dataset.temporals_name    # 时序特征列名（星期、月份等）
        self.labels_name = self.dataset.labels_name          # 标签列（如未来收益，本环境用得少）
        self.stocks_df = []
        self._ret_history = []      # 组合日收益率历史（用于算夏普奖励）
        self._date_history = []     # 对应日期历史

        # ===================== 数据预处理：标准化 + 切片 =====================
        prices = []
        if if_norm:
            print("normalize datasets")

            if self.mode == "train":
                # 训练模式：对每只股票各自 fit 一个 StandardScaler，并把它存起来供验证/测试复用
                self.scaler = []
                for df in self.dataset.stocks_df:
                    # 按起止日期切片
                    if end_date is not None:
                        df = df.loc[start_date:end_date].copy()
                    else:
                        df = df.loc[start_date:].copy()

                    # 把小写价格列(open/high/low/close)复制到大写列(OPEN/HIGH/LOW/CLOSE)
                    df[self.prices_name] = df[[name.lower() for name in self.prices_name]]
                    price_df = df[self.prices_name]
                    prices.append(price_df.values)       # 价格**不做标准化**（交易要用真实价）

                    # 只对特征(+可选时序)做标准化
                    scaler = StandardScaler()
                    if self.if_norm_temporal:
                        df[self.features_name + self.temporals_name] = scaler.fit_transform(
                            df[self.features_name + self.temporals_name])
                    else:
                        df[self.features_name] = scaler.fit_transform(df[self.features_name])

                    self.scaler.append(scaler)           # 记下每只股票的 scaler
                    self.stocks_df.append(df)
            else:
                # 验证/测试模式：必须复用训练集的 scaler（用 transform 而非 fit_transform，防止数据泄漏）
                assert self.scaler is not None, "val mode or test mode is not None."
                for index, df in enumerate(self.dataset.stocks_df):
                    if end_date is not None:
                        df = df.loc[start_date:end_date].copy()
                    else:
                        df = df.loc[start_date:].copy()

                    df[self.prices_name] = df[[name.lower() for name in self.prices_name]]
                    price_df = df[self.prices_name]
                    prices.append(price_df.values)

                    scaler = self.scaler[index]          # 用训练时拟合好的 scaler
                    if self.if_norm_temporal:
                        df[self.features_name + self.temporals_name] = scaler.transform(
                            df[self.features_name + self.temporals_name])
                    else:
                        df[self.features_name] = scaler.transform(df[self.features_name])

                    self.stocks_df.append(df)
        else:
            print("no normalize datasets")

        # ---- 把 DataFrame 堆叠成 numpy 张量，方便按 [股票, 天, 特征] 索引 ----
        self.features = []
        for df in self.stocks_df:
            df = df[self.features_name + self.temporals_name]
            self.features.append(df.values)
        self.features = np.stack(self.features)          # shape: [N股票, T天, F特征]

        self.prices = np.stack(prices)                   # shape: [N股票, T天, 4(OHLC)]

        self.labels = []
        for df in self.stocks_df:
            df = df[self.labels_name]
            self.labels.append(df.values)
        self.labels = np.stack(self.labels)

        self.num_days = self.features.shape[1]           # 总交易天数 T
        self.dates = pd.to_datetime(self.stocks_df[0].index)
        print("features shape {}, prices shape {}, labels shape {}, num days {}".format(
            self.features.shape, self.prices.shape, self.labels.shape, self.features.shape[1]))
        print("dates range:", self.dates[0], "to", self.dates[-1])

        # ===================== 交易状态初始化（股数级账本） =====================
        self.shares_held = np.zeros(self.n_stocks, dtype=np.float64)   # 各股票当前持仓股数
        self.shares_frozen = np.zeros(self.n_stocks, dtype=np.float64) # 当天买入、被 T+1 冻结(当天不可卖)的股数
        self.cash = self.initial_amount                               # 现金余额
        self.capital_history = []                                     # 总资产历史
        self.last_rebalance_day = -self.rebalance_period             # 上次调仓日（初始设为负，保证第一天可调仓）
        self.can_buy_mask = np.ones(self.n_stocks, dtype=bool)        # 可买掩码（封涨停的股票为 False）
        self.can_sell_mask = np.ones(self.n_stocks, dtype=bool)       # 可卖掩码（封跌停的股票为 False）

        # ---- 从股票代码推断"交易所"和"涨跌停幅度" ----
        self.stock_markets = []                                       # 每只股票所属市场（'SH'/'SZ'）
        self.stock_limit_ratios = np.zeros(self.n_stocks, dtype=np.float64)  # 每只股票的涨跌停幅度
        for i, stock in enumerate(self.stocks):
            market, limit_ratio = self._infer_market_and_limit(stock)
            self.stock_markets.append(market)
            self.stock_limit_ratios[i] = limit_ratio

        print("stock markets inferred, n_stocks={}, rebalance_period={}".format(self.n_stocks, self.rebalance_period))

        # ---- 选定起始日 ----
        if self.mode == "train":
            # 训练：在 [days-1, 总天数的3/4] 间随机选起始日 → 数据增强(每个 episode 起点不同)
            self.day = random.randint(self.days - 1, 3 * (self.num_days // 4))
        else:
            # 验证/测试：固定从第 days-1 天开始（保证有足够回看窗口）
            self.day = self.days - 1

    def _infer_market_and_limit(self, stock):
        """从股票代码推断交易所和涨跌停幅度（A 股规则）。"""
        if '.' in stock:
            code, suffix = stock.split('.')        # 形如 "600000.SH" → code="600000", suffix="SH"
        else:
            code = stock
            suffix = 'SH'

        # 后缀判断交易所
        if suffix == 'SH':
            market = 'SH'                           # 上交所
        else:
            market = 'SZ'                           # 深交所

        # 按代码前缀/特征判断涨跌停幅度
        if 'ST' in stock.upper():
            limit_ratio = 0.05                      # ST 股：±5%
        elif code.startswith('688'):
            limit_ratio = 0.20                      # 科创板：±20%
        elif code.startswith('300') or code.startswith('301'):
            limit_ratio = 0.20                      # 创业板：±20%
        elif code.startswith('8') or code.startswith('4'):
            limit_ratio = 0.30                      # 北交所：±30%
        else:
            limit_ratio = 0.10                      # 主板：±10%

        return market, limit_ratio

    def get_current_date(self):
        """返回当前交易日的日期。"""
        return self.stocks_df[0].index[self.day]

    def reset(self, *, seed=None, options=None):
        """重置环境到一个新 episode 的起点，返回初始状态。gym 标准接口。"""
        self._ret_history = []
        self._date_history = []
        self.capital_history = []

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # 重新选起始日（训练随机、验证/测试固定）
        if self.mode == "train":
            self.day = random.randint(self.days - 1, 3 * (self.num_days // 4))
        else:
            self.day = self.days - 1

        # 重置账本
        self.shares_held = np.zeros(self.n_stocks, dtype=np.float64)
        self.shares_frozen = np.zeros(self.n_stocks, dtype=np.float64)
        self.cash = self.initial_amount
        self.last_rebalance_day = self.day - self.rebalance_period   # 保证起始日当天就是一个可调仓日
        self.can_buy_mask = np.ones(self.n_stocks, dtype=bool)
        self.can_sell_mask = np.ones(self.n_stocks, dtype=bool)
        self.protfolio_value = self.initial_amount                   # (原代码拼写 protfolio，保留)

        # 初始状态 = 最近 days 天的特征窗口，shape [N, days, F]
        state = self.features[:, self.day - self.days + 1: self.day + 1, :]
        self.state = state

        info = {"current_date": self.get_current_date()}
        print("EnvironmentASR reset, day={}, cash={:.2f}, initial_amount={}".format(
            self.day, self.cash, self.initial_amount))
        return state, info

    def get_prices(self):
        """取当前交易日的开/高/低/收价，以及前一日收盘价(用于算涨跌停)。"""
        prices = self.prices[:, self.day, :]
        o, h, l, c = prices[:, 0], prices[:, 1], prices[:, 2], prices[:, 3]
        if self.day > 0:
            prev_c = self.prices[:, self.day - 1, 3]    # 前一日收盘
        else:
            prev_c = o
        return o, h, l, c, prev_c

    def _get_total_capital(self):
        """当前总资产 = 现金 + 持仓市值(按当日开盘价估值)。"""
        current_prices = self.prices[:, self.day, 0]    # 用开盘价(列0)估值
        stock_value = np.sum(self.shares_held * current_prices)
        return self.cash + stock_value

    def _normalize_action_weights(self, action):
        """把智能体输出的原始动作清洗成合法的权重分布(非负、和为1)。
        action[0]=现金权重，action[1:]=各股票权重。"""
        clean_action = np.maximum(action, 0)            # 负权重截为 0(不允许做空)
        total = clean_action.sum()
        if total > 1e-8:
            target_weights = clean_action / total       # 归一化到和为 1
        else:
            target_weights = np.ones(len(clean_action)) / len(clean_action)  # 全 0 时均分(兜底)
        return target_weights

    def _weights_to_shares(self, weights, capital, prices):
        """把'目标权重'换算成'目标持仓股数'。weights 是各股票权重(不含现金)。"""
        target_shares = np.zeros(self.n_stocks, dtype=np.float64)
        safe_capital = capital * (1 - self.cash_reserve_ratio)      # 预留 1% 现金，用 99% 资金买股

        for i in range(self.n_stocks):
            if weights[i] <= 1e-8 or prices[i] < 1e-8:              # 权重≈0 或价格异常 → 不持有
                target_shares[i] = 0
                continue
            target_amount = weights[i] * safe_capital              # 分配给该股的金额
            target_shares[i] = target_amount / prices[i]           # 金额 / 价格 = 股数(此处还未取整手)
        return target_shares

    def _apply_t1_constraint(self, target_shares):
        """T+1 约束：当天买入的股票当天不能卖。
        做法：若目标股数低于'可卖股数'(=持仓-冻结)，说明想卖出被冻结的部分，强制把目标抬回可卖下限。"""
        constrained_shares = target_shares.copy()
        sellable_shares = np.maximum(self.shares_held - self.shares_frozen, 0)  # 可卖 = 持仓 - 冻结
        for i in range(self.n_stocks):
            if target_shares[i] < sellable_shares[i]:
                constrained_shares[i] = sellable_shares[i]         # 不允许卖到低于可卖下限
        return constrained_shares

    def _update_price_limit_masks(self, prev_close, open_price, high, low):
        """根据前收盘价和当日高低价，判断每只股票是否封涨停/跌停，更新可买/可卖掩码。"""
        for i in range(self.n_stocks):
            if prev_close[i] < 1e-8:                                # 价格异常 → 都允许
                self.can_buy_mask[i] = True
                self.can_sell_mask[i] = True
                continue
            limit_ratio = self.stock_limit_ratios[i]
            upper_limit = prev_close[i] * (1 + limit_ratio)        # 涨停价
            lower_limit = prev_close[i] * (1 - limit_ratio)        # 跌停价
            if high[i] >= upper_limit - 0.01:                      # 摸到涨停 → 禁买(买不进)、可卖
                self.can_buy_mask[i] = False
                self.can_sell_mask[i] = True
            elif low[i] <= lower_limit + 0.01:                     # 摸到跌停 → 可买、禁卖(卖不出)
                self.can_buy_mask[i] = True
                self.can_sell_mask[i] = False
            else:                                                  # 正常 → 都允许
                self.can_buy_mask[i] = True
                self.can_sell_mask[i] = True

    def _apply_price_limit_constraint(self, target_shares):
        """涨跌停约束：封涨停时不能加仓(目标≤现有持仓)，封跌停时不能减仓(目标≥现有持仓)。"""
        constrained_shares = target_shares.copy()
        for i in range(self.n_stocks):
            if not self.can_buy_mask[i]:                           # 不能买 → 目标股数不得超过现有持仓
                constrained_shares[i] = min(target_shares[i], self.shares_held[i])
            if not self.can_sell_mask[i]:                          # 不能卖 → 目标股数不得低于现有持仓
                constrained_shares[i] = max(target_shares[i], self.shares_held[i])
        return constrained_shares

    def _apply_lot_size_constraint(self, target_shares):
        """整手约束：把目标股数向下取整到 lot_size(100股)的整数倍。"""
        if not self.enable_lot_size:
            return target_shares.copy()
        constrained_shares = target_shares.copy()
        for i in range(self.n_stocks):
            constrained_shares[i] = (constrained_shares[i] // self.lot_size) * self.lot_size
        return constrained_shares

    def _apply_cash_constraint(self, target_shares, prices, old_shares):
        """现金约束：保证'买入所需现金'不超过'现金+卖出回款'，否则按比例缩减买入量。
        这一步会预估卖出回款和买入花费(含各项费用和滑点)，确保不穿仓。"""
        # ---- 先估卖出能回多少现金 ----
        sell_shares = np.maximum(0, old_shares - target_shares)    # 需要卖出的股数
        if self.enable_slippage:
            sell_prices = prices * (1 - self.slip_perc)            # 卖出滑点：成交价更低
        else:
            sell_prices = prices
        sell_amounts = sell_shares * sell_prices
        sell_costs = np.zeros(len(sell_amounts))
        for i in range(len(sell_amounts)):
            if sell_amounts[i] > 0:
                commission = max(sell_amounts[i] * self.commission_rate, self.commission_min)  # 佣金(有最低)
                stamp_tax = sell_amounts[i] * self.stamp_tax_rate                              # 印花税(仅卖)
                market = self.stock_markets[i]
                transfer_fee = sell_amounts[i] * self.sh_transfer_fee_rate if market == 'SH' else 0  # 过户费(仅沪)
                sell_costs[i] = commission + stamp_tax + transfer_fee
        net_sell_proceeds = np.sum(sell_amounts - sell_costs)      # 卖出净回款
        available_cash = self.cash + net_sell_proceeds            # 可用现金 = 当前现金 + 卖出回款

        # ---- 再估买入要花多少现金 ----
        buy_shares = np.maximum(0, target_shares - old_shares)     # 需要买入的股数
        if self.enable_slippage:
            buy_prices = prices * (1 + self.slip_perc)            # 买入滑点：成交价更高
        else:
            buy_prices = prices
        buy_amounts = buy_shares * buy_prices
        buy_costs = np.zeros(len(buy_amounts))
        for i in range(len(buy_amounts)):
            if buy_amounts[i] > 0:
                commission = max(buy_amounts[i] * self.commission_rate, self.commission_min)
                market = self.stock_markets[i]
                transfer_fee = buy_amounts[i] * self.sh_transfer_fee_rate if market == 'SH' else 0
                buy_costs[i] = commission + transfer_fee          # 买入不收印花税
        total_buy_needed = np.sum(buy_amounts + buy_costs)        # 买入总花费

        # ---- 若买入超过可用现金，按比例缩减买入量(再取整手) ----
        if total_buy_needed > available_cash + 1e-8:
            scale_ratio = available_cash / total_buy_needed       # 缩减比例
            buy_shares = buy_shares * scale_ratio
            if self.enable_lot_size:
                buy_shares = (buy_shares // self.lot_size) * self.lot_size
            buy_amounts = buy_shares * buy_prices
            buy_costs = np.zeros(len(buy_amounts))
            for i in range(len(buy_amounts)):
                if buy_amounts[i] > 0:
                    commission = max(buy_amounts[i] * self.commission_rate, self.commission_min)
                    market = self.stock_markets[i]
                    transfer_fee = buy_amounts[i] * self.sh_transfer_fee_rate if market == 'SH' else 0
                    buy_costs[i] = commission + transfer_fee

        # 最终目标持仓 = 旧持仓 - 实际卖出 + 实际买入
        constrained_shares = old_shares - sell_shares + buy_shares
        return constrained_shares

    def _execute_trades(self, old_shares, target_shares, prices, highs, lows):
        """真正执行交易：逐股计算成交价(含滑点、限制在当日高低价内)、各项费用，并更新现金。
        返回交易明细(新持仓、各类成本等)。"""
        stock_value = np.sum(old_shares * prices)
        pre_trade_capital = self.cash + stock_value               # 交易前总资产

        sell_shares = np.maximum(0, old_shares - target_shares)   # 实际卖出
        buy_shares = np.maximum(0, target_shares - old_shares)    # 实际买入

        total_commission = 0.0
        total_stamp_tax = 0.0
        total_transfer_fee = 0.0
        total_slippage_cost = 0.0

        # ---- 执行卖出 ----
        for i in range(self.n_stocks):
            if sell_shares[i] > 1e-8:
                execution_price = prices[i]
                if self.enable_slippage:
                    execution_price = execution_price * (1 - self.slip_perc)  # 卖出滑点
                execution_price = np.clip(execution_price, lows[i], highs[i]) # 成交价限制在当日[低,高]内
                slippage_cost = sell_shares[i] * (prices[i] - execution_price)
                total_slippage_cost += slippage_cost
                sell_amount = sell_shares[i] * execution_price
                commission = max(sell_amount * self.commission_rate, self.commission_min)
                total_commission += commission
                stamp_tax = sell_amount * self.stamp_tax_rate                  # 卖出收印花税
                total_stamp_tax += stamp_tax
                market = self.stock_markets[i]
                transfer_fee = 0.0
                if market == 'SH':
                    transfer_fee = sell_amount * self.sh_transfer_fee_rate
                    total_transfer_fee += transfer_fee
                self.cash += (sell_amount - commission - stamp_tax - transfer_fee)  # 卖出回款打进现金(扣费后)

        # ---- 执行买入 ----
        for i in range(self.n_stocks):
            if buy_shares[i] > 1e-8:
                execution_price = prices[i]
                if self.enable_slippage:
                    execution_price = execution_price * (1 + self.slip_perc)  # 买入滑点
                execution_price = np.clip(execution_price, lows[i], highs[i])
                slippage_cost = buy_shares[i] * (execution_price - prices[i])
                total_slippage_cost += slippage_cost
                buy_amount = buy_shares[i] * execution_price
                commission = max(buy_amount * self.commission_rate, self.commission_min)
                total_commission += commission
                market = self.stock_markets[i]
                transfer_fee = 0.0
                if market == 'SH':
                    transfer_fee = buy_amount * self.sh_transfer_fee_rate
                    total_transfer_fee += transfer_fee
                cost = buy_amount + commission + transfer_fee
                if self.cash >= cost - 1e-8:                       # 现金够才买(double-check 不穿仓)
                    self.cash -= cost

        new_shares = old_shares - sell_shares + buy_shares
        total_cost = total_commission + total_stamp_tax + total_transfer_fee + total_slippage_cost

        return {
            'old_shares': old_shares,
            'new_shares': new_shares,
            'pre_trade_capital': pre_trade_capital,
            'post_trade_cash': self.cash,
            'total_cost': total_cost,
            'commission': total_commission,
            'stamp_tax': total_stamp_tax,
            'transfer_fee': total_transfer_fee,
            'slippage_cost': total_slippage_cost
        }

    def _compute_asr_reward(self):
        """奖励 = 年化夏普比率(ASR)。用至今的组合日收益率序列计算。
        ⚠️ 这是该魔改版的奖励，论文原版用的是市值变化 V_t - V_{t-1}。"""
        if len(self._ret_history) < 2:                            # 收益样本太少 → 奖励 0
            return 0.0
        date_index = pd.to_datetime(self._date_history)
        rets = pd.Series(self._ret_history, index=date_index)
        reward = qs_stats.sharpe(rets, rf=0.02)                   # quantstats 算夏普(无风险利率 2%)
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        reward = max(reward, 0.0)                                 # 裁剪为非负(避免负奖励干扰训练)
        return reward

    def _step_hold(self):
        """非调仓日：只持有不交易。按持仓在'今开→明开'之间的市值变化算当日收益。"""
        current_day = self.day
        open_prices = self.prices[:, current_day, 0]
        stock_value = np.sum(self.shares_held * open_prices)
        old_capital = self.cash + stock_value                     # 今日开盘总资产

        next_day = current_day + 1
        if next_day < self.num_days:
            next_open_prices = self.prices[:, next_day, 0]
            next_stock_value = np.sum(self.shares_held * next_open_prices)
            new_capital = self.cash + next_stock_value            # 次日开盘总资产
        else:
            new_capital = old_capital

        daily_return = (new_capital - old_capital) / (old_capital + 1e-8)  # 当日收益率
        self.capital_history.append(new_capital)
        self._ret_history.append(daily_return)
        current_date = self.get_current_date()
        self._date_history.append(current_date)

        reward = self._compute_asr_reward()                       # 用累计收益序列算夏普

        info = {
            'portfolio_ret': daily_return,
            'portfolio_value': new_capital,
            'date': current_date,
            'is_rebalance': False                                 # 标记：非调仓日
        }
        self.protfolio_value = new_capital
        return reward, info

    def _step_rebalance(self, action):
        """调仓日：把智能体动作(目标权重)经过 T+1/涨跌停/整手/现金 一系列约束后真正换仓。"""
        assert action is not None, "action is None on rebalance day"
        current_day = self.day
        open_prices = self.prices[:, current_day, 0]              # 以开盘价成交
        high_prices = self.prices[:, current_day, 1]
        low_prices = self.prices[:, current_day, 2]

        if current_day > 0:
            prev_close_prices = self.prices[:, current_day - 1, 3]
        else:
            prev_close_prices = open_prices

        # ① 更新涨跌停掩码
        if self.enable_price_limit:
            self._update_price_limit_masks(prev_close_prices, open_prices, high_prices, low_prices)
        else:
            self.can_buy_mask[:] = True
            self.can_sell_mask[:] = True

        # ② 动作 → 合法权重 → 目标股数
        target_weights_all = self._normalize_action_weights(action)   # [现金, 各股] 归一化
        target_weights = target_weights_all[1:]                       # 各股票权重
        cash_weight = target_weights_all[0]                           # 现金权重
        current_capital = self._get_total_capital()
        target_shares = self._weights_to_shares(target_weights, current_capital, open_prices)

        # ③ 依次施加各项约束（顺序很重要：T+1 → 涨跌停 → 整手 → 现金）
        if self.enable_t1_rule:
            target_shares = self._apply_t1_constraint(target_shares)
        if self.enable_price_limit:
            target_shares = self._apply_price_limit_constraint(target_shares)
        if self.enable_lot_size:
            target_shares = self._apply_lot_size_constraint(target_shares)
        target_shares = self._apply_cash_constraint(target_shares, open_prices, self.shares_held)

        # ④ 真正执行交易（扣费、更新现金）
        trade_result = self._execute_trades(self.shares_held, target_shares, open_prices, high_prices, low_prices)
        self.shares_held = trade_result['new_shares'].copy()
        self.cash = trade_result['post_trade_cash']

        # ⑤ 记录今日新买入的股数为"冻结"(T+1：今天买的明天才能卖)
        if self.enable_t1_rule:
            bought = np.maximum(0, target_shares - trade_result['old_shares'])
            self.shares_frozen = bought.copy()

        # ⑥ 算调仓后资产 与 当日收益(今开总资产 → 次开总资产)
        stock_value = np.sum(self.shares_held * open_prices)
        post_rebalance_capital = self.cash + stock_value
        next_day = current_day + 1
        if next_day < self.num_days:
            next_open_prices = self.prices[:, next_day, 0]
            next_stock_value = np.sum(self.shares_held * next_open_prices)
            next_capital = self.cash + next_stock_value
        else:
            next_capital = post_rebalance_capital

        old_capital = trade_result['pre_trade_capital']
        daily_return = (next_capital - old_capital) / (old_capital + 1e-8)  # 当日收益(已含交易成本拖累)

        self.capital_history.append(next_capital)
        self._ret_history.append(daily_return)
        current_date = self.get_current_date()
        self._date_history.append(current_date)

        reward = self._compute_asr_reward()
        self.last_rebalance_day = current_day                     # 更新上次调仓日

        info = {
            'portfolio_ret': daily_return,
            'portfolio_value': next_capital,
            'date': current_date,
            'is_rebalance': True,
            'cash': self.cash,
            'stock_value': stock_value,
            'total_cost': trade_result['total_cost'],             # 本次调仓总成本
            'commission': trade_result['commission'],
            'stamp_tax': trade_result['stamp_tax'],
            'transfer_fee': trade_result['transfer_fee'],
            'slippage_cost': trade_result['slippage_cost'],
            'target_weights': target_weights_all,                 # 目标权重(模型想要的)
            'actual_weights': np.concatenate([                    # 实际权重(约束后真实成交的)
                [cash_weight],
                self.shares_held * open_prices / (post_rebalance_capital + 1e-8)])
        }
        self.protfolio_value = next_capital
        print("Rebalance day={}, capital={:.2f}, return={:.6f}, cost={:.2f}, cash={:.2f}".format(
            current_day, next_capital, daily_return, trade_result['total_cost'], self.cash))
        return reward, info

    def step(self, action=None):
        """【主入口】gym 标准 step。每个时间步：判断是否调仓日 → 持有或调仓 → 推进一天 → 返回。"""
        if self.enable_t1_rule:
            self.shares_frozen[:] = 0          # 每天开盘先解冻(昨天买的今天可卖了)；若今天调仓会重新冻结今买的

        # 是否到了调仓日：距上次调仓 ≥ rebalance_period 天
        is_rebalance_day = (self.day - self.last_rebalance_day >= self.rebalance_period)

        if not is_rebalance_day:
            reward, info = self._step_hold()         # 非调仓日：只持有
        else:
            reward, info = self._step_rebalance(action)   # 调仓日：执行换仓

        self.day += 1                                # 推进到下一天
        done = self.day >= self.num_days - 1         # 到数据末尾则结束

        # 下一状态 = 最近 days 天特征窗口
        next_state = self.features[:, self.day - self.days + 1: self.day + 1, :]
        self.state = next_state
        if not done:
            self.protfolio_value = info.get('portfolio_value', self.protfolio_value)

        terminated = done
        truncated = False                            # 本环境不用 truncated(超时截断)
        # gym 新接口返回 5 元组：(下一状态, 奖励, 是否终止, 是否截断, 信息)
        return next_state, reward, terminated, truncated, info


# ==========================================================================================
# 【一步交互的完整流程图（新版环境）】
#
#   step(action)
#     ├─ 解冻昨日买入(shares_frozen=0)
#     ├─ 是否调仓日? (day - last_rebalance_day >= rebalance_period)
#     │     ├─ 否 → _step_hold():        只持有, 按持仓市值变化算日收益 → 算夏普奖励
#     │     └─ 是 → _step_rebalance(a):
#     │              动作 → 归一化权重 → 目标股数
#     │              → T+1约束 → 涨跌停约束 → 整手约束 → 现金约束
#     │              → _execute_trades(): 逐股成交(滑点/佣金/印花税/过户费), 更新现金&持仓
#     │              → 冻结今日买入(T+1) → 算日收益(含成本) → 算夏普奖励
#     ├─ day += 1, 判断 done
#     └─ 返回 (next_state, reward, terminated, truncated, info)
#
# 【与旧版环境的本质区别】
#   旧版: 直接用"权重×当日涨跌幅"算组合收益, 无股数/无费用细节/无 A 股规则, 每天都"调仓"。
#   新版: 股数级账本 + A 股全套交易摩擦 + t+N 调仓, 更贴近实盘, 但训练更慢、收益更"真实"(更低)。
# ==========================================================================================
