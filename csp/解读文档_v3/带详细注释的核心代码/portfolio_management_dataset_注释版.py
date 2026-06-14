# -*- coding: utf-8 -*-
"""
==========================================================================================
数据集 PortfolioManagementDataset —— 逐行详细中文注释版（学习用）
对应原文件：EarnMore-main_v20260610/pm/dataset/portfolio_management_dataset.py
（新旧版逐字节相同；本注释为 v3 补充，旧版注释集未覆盖此文件）
==========================================================================================

【这份文件做什么】
它是整个项目的"数据层"，负责把磁盘上的三类原始文件读进内存、组织好，交给环境使用：
  ① 全局股票池 GSP 的股票代码列表（stocks_path 指向的 txt，一行一个代码）
  ② 每只股票的特征 CSV（data_path 目录下，每只股票一个 .csv）
  ③ 各"可定制股票池 CSP"的定义（aux_stocks_path 目录下的多个 txt，每个定义一个子池）

【与论文的关系】★关键
- 论文的"可定制股票池(CSP)"就在这里落地：每个 CSP 用一个 mask 向量表示——
  在子池内的股票 mask=0、池外的股票 mask=1。环境/智能体据此知道"哪些股票被投资者排除"。
- 论文 §3.2 的状态特征 X_t = [价格, 技术指标, 时序信息]，对应这里的 features/temporals。

【输出（供环境读取的成员）】
  self.stocks       : GSP 股票代码列表（长度 N）
  self.stocks2id    : 代码 → 索引
  self.id2stocks    : 索引 → 代码
  self.aux_stocks   : dict，每个 CSP 子池的定义（含 mask）
  self.stocks_df    : list，每只股票一个 DataFrame（按日期索引）
==========================================================================================
"""

import os.path
import pandas as pd
from typing import List
from glob import glob
import numpy as np

from pm.registry import DATASET            # 注册器：把本类登记进"电话簿"，配置里 type="PortfolioManagementDataset" 即可创建


@DATASET.register_module()                 # ← 登记进注册器
class PortfolioManagementDataset():
    def __init__(self,
                 root: str = None,          # 项目根目录（其余路径都相对它拼接）
                 data_path: str = None,     # 每只股票特征 CSV 所在目录（相对 root）
                 stocks_path: str = None,   # GSP 股票列表 txt（相对 root）
                 aux_stocks_path: str = None,  # 各 CSP 子池定义 txt 所在目录（相对 root）
                 features_name: List[str] = None,   # 要用的特征列名（技术指标等）
                 temporals_name: List[str] = None,  # 时序特征列名（星期/月份等日期信息）
                 labels_name: List[str] = None):    # 标签列名（如未来收益，本项目环境用得少）
        super(PortfolioManagementDataset, self).__init__()

        self.root = root
        self.data_path = data_path
        self.stocks_path = stocks_path
        self.features_name = features_name
        self.temporals_name = temporals_name
        self.labels_name = labels_name

        # 把相对路径拼成绝对路径
        self.data_path = os.path.join(root, self.data_path)
        self.stocks_path = os.path.join(root, self.stocks_path)
        self.aux_stocks_path = os.path.join(root, aux_stocks_path)

        # ① 读 GSP 股票列表
        self.stocks = self._init_stocks()

        # 建立 代码↔索引 的双向映射（后续 mask、权重都按这个索引顺序对齐）
        self.stocks2id = {stock: i for i, stock in enumerate(self.stocks)}
        self.id2stocks = {i: stock for i, stock in enumerate(self.stocks)}

        # ② 读各 CSP 子池定义
        self.aux_stocks = self._init_aux_stocks()

        # ③ 人为插入 id=0 的特殊"子池"——它就是完整 GSP（mask 全 0 = 一只都不屏蔽）
        #    这样验证/测试时，id=0 代表"在完整全局池上跑"，其余 id 代表各投资者子池。
        self.aux_stocks[0] = {
            "id": 0,
            "type": "all",
            "name": "All",
            "stocks": self.stocks,
            "mask": np.zeros(len(self.stocks)),     # 全 0：不屏蔽任何股票
        }

        # ④ 读每只股票的特征 DataFrame
        self.stocks_df = self._init_stocks_df()

    def _init_stocks(self):
        """读 GSP 股票列表 txt（一行一个股票代码）。"""
        print("init stocks...")
        stocks = []
        with open(self.stocks_path) as op:
            for line in op.readlines():
                line = line.strip()              # 去掉首尾空白/换行
                stocks.append(line)
        print("init stocks success...")
        return stocks

    def _init_stocks_df(self):
        """读每只股票的特征 CSV，组织成按交易日索引的 DataFrame 列表。"""
        print("init stocks dataframe...")
        stocks_df = []
        for stock in self.stocks:
            path = os.path.join(self.data_path, f"{stock}.csv")   # 每只股票一个 CSV
            df = pd.read_csv(path, index_col=0)
            df = df.set_index("trade_date")                       # 用交易日做索引（方便按日期切片）
            # 只保留需要的列：特征 + 时序 + 标签（顺序固定，后续环境按这个顺序取）
            df = df[self.features_name + self.temporals_name + self.labels_name]
            stocks_df.append(df)
        print("init stocks dataframe success...")
        return stocks_df

    def _init_aux_stocks(self) -> dict:
        """读各 CSP 子池定义 txt，为每个子池生成 mask 向量。★论文'可定制股票池'的核心落地。

        子池 txt 命名形如 "01_IT.txt" / "02_Medical.txt"：前缀数字是 id，后缀是子池名。
        文件内容是该子池包含的股票代码（一行一个）。
        """
        print("init aux stocks...")
        aux_stocks = {}
        aux_stocks_files = glob(os.path.join(self.aux_stocks_path, "*.txt"))   # 找出目录下所有子池 txt
        for path in aux_stocks_files:
            name = os.path.basename(path).split(".")[0]   # 去掉扩展名，得到 "01_IT"
            id, name = name.split("_")                     # 拆成 id="01", name="IT"
            id = int(id)

            # 读该子池包含的股票代码
            with open(path) as op:
                stocks = []
                for line in op.readlines():
                    line = line.strip()
                    stocks.append(line)

            aux_stocks[id] = {
                "name": name,
                "type": "aux",
                "stocks": stocks,
                "num_stocks": len(stocks),
                # ★★★ 生成 mask：遍历"全局池每只股票"，在本子池内的记 0、不在的记 1。
                #     mask=0 → 该股票属于这个 CSP，可以投资；
                #     mask=1 → 该股票被投资者排除（池外），后续会被掩码/重加权压成 0 权重。
                #     注意 mask 的长度 = GSP 股票数 N，顺序与 self.stocks 一致（与状态/动作对齐）。
                "mask": np.array([0.0 if stock in stocks else 1.0 for stock in self.stocks])
            }
        # 打印各子池概况，便于核对
        for k, v in aux_stocks.items():
            print(f"aux stocks id: {k}, name: {v['name']}, num stocks: {v['num_stocks']}")
        print("init aux stocks success...")
        return aux_stocks


# ==========================================================================================
# 【数据流串联】
#   stocks.txt        → self.stocks（GSP 代码列表，定义动作/mask 的索引顺序）
#   features/*.csv    → self.stocks_df（每股按日期的特征表）→ 环境堆叠成 [N, T, F] 张量
#   aux_*.txt         → self.aux_stocks（每个 CSP 一个 mask 向量，长度 N）
#
# 【训练 vs 推理时 mask 的用法（呼应论文算法 1/2）】
#   - 训练：环境/智能体并不直接用某个固定 mask，而是由 MAE 用"随机采样的掩码率"模拟各种子池
#     （见 mae.py 的 random_masking、mask_ratio_generator）。这样一次训练见过海量随机子池。
#   - 推理/验证：用 self.aux_stocks[id]["mask"] —— 即投资者真实指定的某个 CSP，
#     把池外股票当成"被掩盖的"。这正是"一次训练、面对任意子池无需重训"的落地方式。
# ==========================================================================================
