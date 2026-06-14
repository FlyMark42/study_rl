# -*- coding: utf-8 -*-
# ============================================================================
#  文件: pm/utils/helpers.py  （带详细中文注释版）
#  作用: 实现论文第 4.3 节的"重加权(Re-weighting)"机制 —— 公式(9)
#       它把 Actor 网络输出的"原始 logits"包装成"投资组合权重"。
#  关键词: 温度 softmax、掩码忽略、稀疏化
# ============================================================================
import torch
import torch.nn.functional as F
import numpy as np


def discretized_actions(action, discretized_low, discretized_high):
    """
    把 [0,1] 的连续动作离散化成整数份额（本论文主线没用到，可忽略）。
    例如把 0.37 这样的连续权重，按总份额映射成整数手数。
    """
    # 先把 [0,1] 的动作线性缩放到 [low, high] 区间
    scaled_tensor = action * (discretized_high - discretized_low) + discretized_low
    # 向下取整得到整数份额
    discretized_tensor = torch.floor(scaled_tensor)
    # 计算取整后与"应有总份额"的差额（取整会丢失一些，要补回来）
    diff = torch.tensor([discretized_high - discretized_low - torch.sum(row) for row in discretized_tensor])

    # 逐行把差额补/扣到"小数部分最大/最小"的位置，保证总份额守恒
    for row_idx, row_diff in enumerate(diff):
        if row_diff > 0:
            # 还差几份 → 给"被截断得最多"的几个位置各 +1
            indices_to_increment = torch.topk(scaled_tensor[row_idx] - discretized_tensor[row_idx], int(row_diff))[1]
            discretized_tensor[row_idx, indices_to_increment] += 1
        elif row_diff < 0:
            # 多了几份 → 扣掉
            indices_to_decrement = torch.topk(discretized_tensor[row_idx] - (scaled_tensor[row_idx] - 1), int(abs(row_diff)))[1]
            discretized_tensor[row_idx, indices_to_decrement] -= 1

    return discretized_tensor.long()


def get_optim_param(optimizer: torch.optim) -> list:
    """
    从优化器的 state_dict 里把所有"状态张量"抽出来成一个列表。
    用途: 保存/加载检查点时，配合 MethodType 给优化器加一个 .parameters() 方法。
    （工程细节，不影响算法理解，可跳过）
    """
    params_list = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list


# ============================================================================
#  下面 3 个 wrapper 是本文件的核心，结构几乎一样，区别只是输入/输出略有不同：
#    - get_action_wrapper        : 包装 Actor 的 get_action（采样动作，训练用）
#    - get_action_logprob_wrapper : 包装 get_action_logprob（采样动作 + 返回对数概率）
#    - forward_action_wrapper     : 包装 Actor 的 forward（确定性前向，验证/推理用）
#
#  它们都实现了论文公式(9):  Re(x) = e^(x_i / T) / Σ_j e^(x_j / T)
#  也就是"带温度 T 的 softmax"。同时支持"掩码忽略"(把池外股票权重压到≈0)。
# ============================================================================

def get_action_wrapper(func,
                       method="softmax",   # 可选 "softmax" 或 "reweight"
                       T=1.0,              # 温度参数 T，越小权重越集中(稀疏)，越大越均匀
                       ):
    """
    返回一个新的 get_action 函数。它先调用原始 Actor 的 func 拿到 logits，
    再做"掩码处理 + 温度 softmax"，最终输出归一化的投资组合权重(和为1)。
    """
    def get_action(x,
                   mask=None,           # 掩码: numpy 数组, 1=被掩盖(池外)股票, 0=保留
                   mask_value=1e6,      # 给被掩盖股票的 logit 减去这个巨大值
                   **kwargs):
        # ---- 第一步: 如果有掩码, 构造一个"掩码惩罚向量" ----
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).to(x.device)
            # 在最前面拼一列 0 (对应"现金"位置, 现金永远不被掩盖)
            mask_bool = torch.concat(
                [torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device), mask_tensor], dim=1).float()
            # 被掩盖的位置变成一个巨大的数 (后面要从 logit 里减掉它)
            mask_bool = mask_bool * mask_value

        # ---- 第二步: 根据 method 计算权重 ----
        if method == "softmax":
            # 普通版: 先除以温度, 再(可选)减掉掩码惩罚, 最后 softmax
            pred = func(x, **kwargs)
            pred = pred / T
            if mask is not None:
                pred = pred - mask_bool      # 被掩盖股票 logit 变得极小 → softmax 后≈0
            weight = F.softmax(pred, dim=-1)

        elif method == "reweight":
            # 论文采用的版本: 先(可选)减掩码, 再做温度 softmax
            pred = func(x, **kwargs)
            if mask is not None:
                pred = pred - mask_bool
            # ★★ 这一行就是论文公式(9): Re(x) = e^(x_i/T) / Σ e^(x_j/T) ★★
            weight = F.softmax(pred / T, dim=-1)
        else:
            raise NotImplementedError
        return weight
    return get_action


def get_action_logprob_wrapper(func,
                       method="softmax",
                       T=1.0,
                       ):
    """
    和上面几乎一样, 唯一区别: 原始 func 会多返回一个 logprob(对数概率)。
    SAC 计算"熵正则项 α·logπ"时需要这个 logprob (见 mask_sac.py 的 update_net)。
    """
    def get_action(x,
                   mask=None,
                   mask_value=1e6,
                   **kwargs):
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).to(x.device)
            mask_bool = torch.concat(
                [torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device), mask_tensor], dim=1).float()
            mask_bool = mask_bool * mask_value

        if method == "softmax":
            pred, logprob = func(x, **kwargs)    # ← 多返回了 logprob
            pred = pred / T
            if mask is not None:
                pred = pred - mask_bool
            weight = F.softmax(pred, dim=-1)

        elif method == "reweight":
            pred, logprob = func(x, **kwargs)
            if mask is not None:
                pred = pred - mask_bool
            # 论文公式(9)
            weight = F.softmax(pred / T, dim=-1)
        else:
            raise NotImplementedError
        return weight, logprob               # ← 权重 + 对数概率一起返回
    return get_action


def forward_action_wrapper(func,
                    method="softmax",
                    T=1.0,
                    ):
    """
    包装 Actor 的 forward()（确定性前向，不带随机采样）。
    用于验证/推理阶段(validate_net): 此时希望动作稳定可复现, 所以用 forward 而非 get_action。
    """
    def forward_action(x,
                       mask=None,
                       mask_value=1e6,
                       **kwargs):
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).to(x.device)
            mask_bool = torch.concat([torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device), mask_tensor], dim=1).float()
            mask_bool = mask_bool * mask_value

        if method == "softmax":
            pred = func(x, **kwargs)
            pred = pred / T
            if mask is not None:
                pred = pred - mask_bool
            weight = F.softmax(pred, dim=-1)

        elif method == "reweight":
            pred = func(x, **kwargs)
            if mask is not None:
                pred = pred - mask_bool
            # 论文公式(9)
            weight = F.softmax(pred / T, dim=-1)
        else:
            raise NotImplementedError
        return weight

    return forward_action

# ============================================================================
#  小结:
#   1) 温度 T 的直觉:
#        T → 0   : softmax 极尖锐, 几乎全部资金压到 logit 最大的那只股票(最稀疏)
#        T = 1   : 退化为普通 softmax
#        T → ∞   : 趋于均匀分配(每只股票权重接近相等)
#      配置文件里 T=0.01, 说明追求"高度集中、稀疏"的投资组合(论文§4.3 的目的)。
#   2) 掩码减大常数(1e6)的直觉:
#        被投资者排除的股票, logit 减去 1e6 后变成极小负数,
#        经过 softmax(e^(很小的负数)≈0), 权重几乎为 0 → 优雅地"忽略池外股票"。
#      这正是 EarnMore "改变掩码即可适配任意子池、无需重训" 的落地细节。
# ============================================================================
