# -*- coding: utf-8 -*-
"""
A股投资组合环境ASR增强版
在现有EnvironmentASR基础上，移植AGDRL_calmar_v2交易执行逻辑
功能：t+N调仓、T+1、涨跌停、手续费、滑点、现金约束、整手约束
奖励：年化夏普比率（ASR）
该文件位置：pm/environment/pm_based_portfolio_ASR.py
"""

import numpy as np
from typing import List, Any
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd
import gym
from quantstats import stats as qs_stats
from pm.registry import ENVIRONMENT


@ENVIRONMENT.register_module()
class EnvironmentASR(gym.Env):
    def __init__(self,
                 mode: str = "train",
                 dataset: Any = None,
                 if_norm: bool = True,
                 if_norm_temporal: bool = True,
                 scaler: List[StandardScaler] = None,
                 days: int = 10,
                 start_date: str = None,
                 end_date: str = None,
                 initial_amount: int = 1e5,
                 transaction_cost_pct: float = 1e-3,
                 rebalance_period: int = 5,
                 enable_t1_rule: bool = True,
                 enable_price_limit: bool = True,
                 enable_lot_size: bool = True,
                 lot_size: int = 100,
                 commission_rate: float = 0.0000855,
                 commission_min: float = 5.0,
                 stamp_tax_rate: float = 0.0005,
                 sh_transfer_fee_rate: float = 0.00002,
                 sz_transfer_fee_rate: float = 0.0,
                 slip_perc: float = 0.001,
                 enable_slippage: bool = True,
                 cash_reserve_ratio: float = 0.01
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

        # 交易参数
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

        self.stocks = self.dataset.stocks
        self.n_stocks = len(self.stocks)
        self.stocks2id = self.dataset.stocks2id
        self.id2stocks = self.dataset.id2stocks
        self.aux_stocks = self.dataset.aux_stocks

        self.features_name = self.dataset.features_name
        self.prices_name = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
        self.temporals_name = self.dataset.temporals_name
        self.labels_name = self.dataset.labels_name
        self.stocks_df = []
        self._ret_history = []
        self._date_history = []

        prices = []
        if if_norm:
            print("normalize datasets")

            if self.mode == "train":
                self.scaler = []
                for df in self.dataset.stocks_df:

                    if end_date is not None:
                        df = df.loc[start_date:end_date].copy()
                    else:
                        df = df.loc[start_date:].copy()

                    df[self.prices_name] = df[[name.lower() for name in self.prices_name]]
                    price_df = df[self.prices_name]
                    prices.append(price_df.values)

                    scaler = StandardScaler()
                    if self.if_norm_temporal:
                        df[self.features_name + self.temporals_name] = scaler.fit_transform(
                            df[self.features_name + self.temporals_name])
                    else:
                        df[self.features_name] = scaler.fit_transform(df[self.features_name])

                    self.scaler.append(scaler)
                    self.stocks_df.append(df)
            else:
                assert self.scaler is not None, "val mode or test mode is not None."

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

        self.features = []
        for df in self.stocks_df:
            df = df[self.features_name + self.temporals_name]
            self.features.append(df.values)
        self.features = np.stack(self.features)

        self.prices = np.stack(prices)

        self.labels = []
        for df in self.stocks_df:
            df = df[self.labels_name]
            self.labels.append(df.values)
        self.labels = np.stack(self.labels)

        self.num_days = self.features.shape[1]
        self.dates = pd.to_datetime(self.stocks_df[0].index)
        print("features shape {}, prices shape {}, labels shape {}, num days {}".format(self.features.shape,
                                                                                        self.prices.shape,
                                                                                        self.labels.shape,
                                                                                        self.features.shape[1]))
        print("dates range:", self.dates[0], "to", self.dates[-1])

        # 交易状态初始化
        self.shares_held = np.zeros(self.n_stocks, dtype=np.float64)
        self.shares_frozen = np.zeros(self.n_stocks, dtype=np.float64)
        self.cash = self.initial_amount
        self.capital_history = []
        self.last_rebalance_day = -self.rebalance_period
        self.can_buy_mask = np.ones(self.n_stocks, dtype=bool)
        self.can_sell_mask = np.ones(self.n_stocks, dtype=bool)

        # 从股票代码推断交易所和涨跌停幅度
        self.stock_markets = []
        self.stock_limit_ratios = np.zeros(self.n_stocks, dtype=np.float64)
        for i, stock in enumerate(self.stocks):
            market, limit_ratio = self._infer_market_and_limit(stock)
            self.stock_markets.append(market)
            self.stock_limit_ratios[i] = limit_ratio

        print("stock markets inferred, n_stocks={}, rebalance_period={}".format(self.n_stocks, self.rebalance_period))

        if self.mode == "train":
            self.day = random.randint(self.days - 1, 3 * (self.num_days // 4))
        else:
            self.day = self.days - 1

    def _infer_market_and_limit(self, stock):
        """从股票代码推断交易所和涨跌停幅度"""
        if '.' in stock:
            code, suffix = stock.split('.')
        else:
            code = stock
            suffix = 'SH'

        if suffix == 'SH':
            market = 'SH'
        else:
            market = 'SZ'

        if 'ST' in stock.upper():
            limit_ratio = 0.05
        elif code.startswith('688'):
            limit_ratio = 0.20
        elif code.startswith('300') or code.startswith('301'):
            limit_ratio = 0.20
        elif code.startswith('8') or code.startswith('4'):
            limit_ratio = 0.30
        else:
            limit_ratio = 0.10

        return market, limit_ratio

    def get_current_date(self):
        return self.stocks_df[0].index[self.day]

    def reset(self, *, seed=None, options=None):
        self._ret_history = []
        self._date_history = []
        self.capital_history = []

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if self.mode == "train":
            self.day = random.randint(self.days - 1, 3 * (self.num_days // 4))
        else:
            self.day = self.days - 1

        self.shares_held = np.zeros(self.n_stocks, dtype=np.float64)
        self.shares_frozen = np.zeros(self.n_stocks, dtype=np.float64)
        self.cash = self.initial_amount
        self.last_rebalance_day = self.day - self.rebalance_period
        self.can_buy_mask = np.ones(self.n_stocks, dtype=bool)
        self.can_sell_mask = np.ones(self.n_stocks, dtype=bool)
        self.protfolio_value = self.initial_amount

        state = self.features[:, self.day - self.days + 1: self.day + 1, :]
        self.state = state

        info = {
            "current_date": self.get_current_date()
        }

        print("EnvironmentASR reset, day={}, cash={:.2f}, initial_amount={}".format(
            self.day, self.cash, self.initial_amount))

        return state, info

    def get_prices(self):
        prices = self.prices[:, self.day, :]
        o, h, l, c = prices[:, 0], prices[:, 1], prices[:, 2], prices[:, 3]
        if self.day > 0:
            prev_c = self.prices[:, self.day - 1, 3]
        else:
            prev_c = o
        return o, h, l, c, prev_c

    def _get_total_capital(self):
        current_prices = self.prices[:, self.day, 0]
        stock_value = np.sum(self.shares_held * current_prices)
        return self.cash + stock_value

    def _normalize_action_weights(self, action):
        clean_action = np.maximum(action, 0)
        total = clean_action.sum()
        if total > 1e-8:
            target_weights = clean_action / total
        else:
            target_weights = np.ones(len(clean_action)) / len(clean_action)
        return target_weights

    def _weights_to_shares(self, weights, capital, prices):
        target_shares = np.zeros(self.n_stocks, dtype=np.float64)
        safe_capital = capital * (1 - self.cash_reserve_ratio)

        for i in range(self.n_stocks):
            if weights[i] <= 1e-8 or prices[i] < 1e-8:
                target_shares[i] = 0
                continue
            target_amount = weights[i] * safe_capital
            target_shares[i] = target_amount / prices[i]

        return target_shares

    def _apply_t1_constraint(self, target_shares):
        constrained_shares = target_shares.copy()
        sellable_shares = np.maximum(self.shares_held - self.shares_frozen, 0)
        for i in range(self.n_stocks):
            if target_shares[i] < sellable_shares[i]:
                constrained_shares[i] = sellable_shares[i]
        return constrained_shares

    def _update_price_limit_masks(self, prev_close, open_price, high, low):
        for i in range(self.n_stocks):
            if prev_close[i] < 1e-8:
                self.can_buy_mask[i] = True
                self.can_sell_mask[i] = True
                continue
            limit_ratio = self.stock_limit_ratios[i]
            upper_limit = prev_close[i] * (1 + limit_ratio)
            lower_limit = prev_close[i] * (1 - limit_ratio)
            if high[i] >= upper_limit - 0.01:
                self.can_buy_mask[i] = False
                self.can_sell_mask[i] = True
            elif low[i] <= lower_limit + 0.01:
                self.can_buy_mask[i] = True
                self.can_sell_mask[i] = False
            else:
                self.can_buy_mask[i] = True
                self.can_sell_mask[i] = True

    def _apply_price_limit_constraint(self, target_shares):
        constrained_shares = target_shares.copy()
        for i in range(self.n_stocks):
            if not self.can_buy_mask[i]:
                constrained_shares[i] = min(target_shares[i], self.shares_held[i])
            if not self.can_sell_mask[i]:
                constrained_shares[i] = max(target_shares[i], self.shares_held[i])
        return constrained_shares

    def _apply_lot_size_constraint(self, target_shares):
        if not self.enable_lot_size:
            return target_shares.copy()
        constrained_shares = target_shares.copy()
        for i in range(self.n_stocks):
            constrained_shares[i] = (constrained_shares[i] // self.lot_size) * self.lot_size
        return constrained_shares

    def _apply_cash_constraint(self, target_shares, prices, old_shares):
        sell_shares = np.maximum(0, old_shares - target_shares)

        if self.enable_slippage:
            sell_prices = prices * (1 - self.slip_perc)
        else:
            sell_prices = prices

        sell_amounts = sell_shares * sell_prices
        sell_costs = np.zeros(len(sell_amounts))
        for i in range(len(sell_amounts)):
            if sell_amounts[i] > 0:
                commission = max(sell_amounts[i] * self.commission_rate, self.commission_min)
                stamp_tax = sell_amounts[i] * self.stamp_tax_rate
                market = self.stock_markets[i]
                if market == 'SH':
                    transfer_fee = sell_amounts[i] * self.sh_transfer_fee_rate
                else:
                    transfer_fee = 0
                sell_costs[i] = commission + stamp_tax + transfer_fee
        net_sell_proceeds = np.sum(sell_amounts - sell_costs)
        available_cash = self.cash + net_sell_proceeds

        buy_shares = np.maximum(0, target_shares - old_shares)

        if self.enable_slippage:
            buy_prices = prices * (1 + self.slip_perc)
        else:
            buy_prices = prices

        buy_amounts = buy_shares * buy_prices
        buy_costs = np.zeros(len(buy_amounts))
        for i in range(len(buy_amounts)):
            if buy_amounts[i] > 0:
                commission = max(buy_amounts[i] * self.commission_rate, self.commission_min)
                market = self.stock_markets[i]
                if market == 'SH':
                    transfer_fee = buy_amounts[i] * self.sh_transfer_fee_rate
                else:
                    transfer_fee = 0
                buy_costs[i] = commission + transfer_fee
        total_buy_needed = np.sum(buy_amounts + buy_costs)

        if total_buy_needed > available_cash + 1e-8:
            scale_ratio = available_cash / total_buy_needed
            buy_shares = buy_shares * scale_ratio
            if self.enable_lot_size:
                buy_shares = (buy_shares // self.lot_size) * self.lot_size
            buy_amounts = buy_shares * buy_prices
            buy_costs = np.zeros(len(buy_amounts))
            for i in range(len(buy_amounts)):
                if buy_amounts[i] > 0:
                    commission = max(buy_amounts[i] * self.commission_rate, self.commission_min)
                    market = self.stock_markets[i]
                    if market == 'SH':
                        transfer_fee = buy_amounts[i] * self.sh_transfer_fee_rate
                    else:
                        transfer_fee = 0
                    buy_costs[i] = commission + transfer_fee

        constrained_shares = old_shares - sell_shares + buy_shares
        return constrained_shares

    def _execute_trades(self, old_shares, target_shares, prices, highs, lows):
        stock_value = np.sum(old_shares * prices)
        pre_trade_capital = self.cash + stock_value

        sell_shares = np.maximum(0, old_shares - target_shares)
        buy_shares = np.maximum(0, target_shares - old_shares)

        total_commission = 0.0
        total_stamp_tax = 0.0
        total_transfer_fee = 0.0
        total_slippage_cost = 0.0

        for i in range(self.n_stocks):
            if sell_shares[i] > 1e-8:
                execution_price = prices[i]
                if self.enable_slippage:
                    execution_price = execution_price * (1 - self.slip_perc)
                execution_price = np.clip(execution_price, lows[i], highs[i])
                slippage_cost = sell_shares[i] * (prices[i] - execution_price)
                total_slippage_cost += slippage_cost
                sell_amount = sell_shares[i] * execution_price
                commission = max(sell_amount * self.commission_rate, self.commission_min)
                total_commission += commission
                stamp_tax = sell_amount * self.stamp_tax_rate
                total_stamp_tax += stamp_tax
                market = self.stock_markets[i]
                transfer_fee = 0.0
                if market == 'SH':
                    transfer_fee = sell_amount * self.sh_transfer_fee_rate
                    total_transfer_fee += transfer_fee
                self.cash += (sell_amount - commission - stamp_tax - transfer_fee)

        for i in range(self.n_stocks):
            if buy_shares[i] > 1e-8:
                execution_price = prices[i]
                if self.enable_slippage:
                    execution_price = execution_price * (1 + self.slip_perc)
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
                if self.cash >= cost - 1e-8:
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
        if len(self._ret_history) < 2:
            return 0.0
        date_index = pd.to_datetime(self._date_history)
        rets = pd.Series(self._ret_history, index=date_index)
        reward = qs_stats.sharpe(rets, rf=0.02)
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        reward = max(reward, 0.0)
        return reward

    def _step_hold(self):
        current_day = self.day
        open_prices = self.prices[:, current_day, 0]
        stock_value = np.sum(self.shares_held * open_prices)
        old_capital = self.cash + stock_value

        next_day = current_day + 1
        if next_day < self.num_days:
            next_open_prices = self.prices[:, next_day, 0]
            next_stock_value = np.sum(self.shares_held * next_open_prices)
            new_capital = self.cash + next_stock_value
        else:
            new_capital = old_capital

        daily_return = (new_capital - old_capital) / (old_capital + 1e-8)
        self.capital_history.append(new_capital)
        self._ret_history.append(daily_return)
        current_date = self.get_current_date()
        self._date_history.append(current_date)

        reward = self._compute_asr_reward()

        info = {
            'portfolio_ret': daily_return,
            'portfolio_value': new_capital,
            'date': current_date,
            'is_rebalance': False
        }

        self.protfolio_value = new_capital

        return reward, info

    def _step_rebalance(self, action):
        assert action is not None, "action is None on rebalance day"
        current_day = self.day
        open_prices = self.prices[:, current_day, 0]
        high_prices = self.prices[:, current_day, 1]
        low_prices = self.prices[:, current_day, 2]

        if current_day > 0:
            prev_close_prices = self.prices[:, current_day - 1, 3]
        else:
            prev_close_prices = open_prices

        if self.enable_price_limit:
            self._update_price_limit_masks(prev_close_prices, open_prices, high_prices, low_prices)
        else:
            self.can_buy_mask[:] = True
            self.can_sell_mask[:] = True

        target_weights_all = self._normalize_action_weights(action)
        target_weights = target_weights_all[1:]
        cash_weight = target_weights_all[0]

        current_capital = self._get_total_capital()
        target_shares = self._weights_to_shares(target_weights, current_capital, open_prices)

        if self.enable_t1_rule:
            target_shares = self._apply_t1_constraint(target_shares)

        if self.enable_price_limit:
            target_shares = self._apply_price_limit_constraint(target_shares)

        if self.enable_lot_size:
            target_shares = self._apply_lot_size_constraint(target_shares)

        target_shares = self._apply_cash_constraint(target_shares, open_prices, self.shares_held)

        trade_result = self._execute_trades(self.shares_held, target_shares, open_prices, high_prices, low_prices)

        self.shares_held = trade_result['new_shares'].copy()
        self.cash = trade_result['post_trade_cash']

        if self.enable_t1_rule:
            bought = np.maximum(0, target_shares - trade_result['old_shares'])
            self.shares_frozen = bought.copy()

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
        daily_return = (next_capital - old_capital) / (old_capital + 1e-8)

        self.capital_history.append(next_capital)
        self._ret_history.append(daily_return)
        current_date = self.get_current_date()
        self._date_history.append(current_date)

        reward = self._compute_asr_reward()
        self.last_rebalance_day = current_day

        info = {
            'portfolio_ret': daily_return,
            'portfolio_value': next_capital,
            'date': current_date,
            'is_rebalance': True,
            'cash': self.cash,
            'stock_value': stock_value,
            'total_cost': trade_result['total_cost'],
            'commission': trade_result['commission'],
            'stamp_tax': trade_result['stamp_tax'],
            'transfer_fee': trade_result['transfer_fee'],
            'slippage_cost': trade_result['slippage_cost'],
            'target_weights': target_weights_all,
            'actual_weights': np.concatenate([[cash_weight], self.shares_held * open_prices / (post_rebalance_capital + 1e-8)])
        }

        self.protfolio_value = next_capital

        print("Rebalance day={}, capital={:.2f}, return={:.6f}, cost={:.2f}, cash={:.2f}".format(
            current_day, next_capital, daily_return, trade_result['total_cost'], self.cash))

        return reward, info

    def step(self, action=None):
        if self.enable_t1_rule:
            self.shares_frozen[:] = 0

        is_rebalance_day = (self.day - self.last_rebalance_day >= self.rebalance_period)

        if not is_rebalance_day:
            reward, info = self._step_hold()
        else:
            reward, info = self._step_rebalance(action)

        self.day += 1
        done = self.day >= self.num_days - 1

        next_state = self.features[:, self.day - self.days + 1: self.day + 1, :]
        self.state = next_state
        if not done:
            self.protfolio_value = info.get('portfolio_value', self.protfolio_value)

        terminated = done
        truncated = False

        return next_state, reward, terminated, truncated, info
