# -*- coding: utf-8 -*-
# ============================================================================
#  文件: pm/net/sac/mask_sac_net.py  （带详细中文注释版）
#  作用: 论文模块(c) 的两个网络 —— 对应论文第 4.2 节:
#        - ActorMaskSAC : 演员/策略网络 π_φ, 负责"做动作"(输出投资组合权重)
#        - CriticMaskSAC: 评论家/价值网络 Q_θ, 负责"打分"(评估动作好坏)
#  二者输入都是"可掩码股票表征 ρ"(由 MaskTimeState.forward_state 产出)。
# ============================================================================
import torch
import torch.nn as nn
from typing import List
from functools import partial
from torch.distributions.normal import Normal      # 正态分布(策略输出动作分布)
from timm.models.layers import Mlp
from torch.nn import functional as F
from pm.registry import NET
from pm.net import MaskTimeState


@NET.register_module(force=True)   # 注册名 "ActorMaskSAC"
class ActorMaskSAC(nn.Module):
    """
    演员(策略网络) π_φ。
    输入: 可掩码股票表征 x, 形状 (B, N, C) = (批量, 股票数, 嵌入维度)
    输出: 投资组合权重, 形状 (B, N+1) (含一个 [CLS]/现金位)
    SAC 是"随机策略", 所以网络输出的是"动作分布的均值和标准差", 再从分布里采样。
    """
    def __init__(self,
                 *args,
                 embed_dim: int = 128,
                 depth: int = 2,                       # MLP 层数
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 cls_embed: bool = True,               # 是否加 [CLS] 标记(管现金分配)
                 **kwargs):
        super(ActorMaskSAC, self).__init__()
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.cls_embed = cls_embed

        # [CLS] 标记: 一个可学习向量, 拼到序列最前面, 用来代表"现金/全局"信息
        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 主干: depth 层 MLP(每层把 embed_dim → embed_dim)
        self.blocks = nn.ModuleList(
            [Mlp(in_features=embed_dim, hidden_features=embed_dim, out_features=embed_dim) for i in range(depth)]
        )
        self.norm = norm_layer(embed_dim)

        # 预测头: 把每个 token 映射成 2 个数 → (动作均值 a_avg, 对数标准差 a_std_log)
        self.decoder_pred = nn.Linear(embed_dim, 2, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 线性层用正交初始化(RL 里常见, 利于稳定); LayerNorm 标准初始化
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        """加 [CLS] → 过 MLP → LayerNorm。x: (B, N, C) → (B, N+1, C)"""
        if self.cls_embed:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # 复制到每个样本
            x = torch.cat((cls_tokens, x), dim=1)                   # [CLS] 拼到最前
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_decoder(self, x):
        """预测头: (B, N+1, C) → (B, N+1, 2)"""
        return self.decoder_pred(x)

    def forward(self, x):
        """
        确定性前向(验证/推理用): 直接用动作均值那一列当 logits, 输出权重。
        注意: 这里有个本仓库特有的"按排序加权"技巧(论文原版没有):
              soft_logits = logits * log(排序索引 + 1)。
        """
        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)
        logits = pred[:, :, 0]                       # 取第0列作为 logits
        indices = torch.sort(logits)[1]              # 排序后的索引
        soft_logits = logits * torch.log(indices + 1)  # 按排序位置加权(魔改技巧)
        weight = F.softmax(soft_logits, dim=-1).squeeze(-1)  # 归一化成权重
        return weight

    def get_action(self, x):
        """
        随机采样动作(训练用)。
        步骤: 网络输出 (均值 a_avg, 对数std a_std_log) → 构造正态分布 → 重参数化采样 → softmax。
        """
        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)
        a_avg, a_std_log = pred.chunk(2, dim=-1)     # 拆成两半: 均值 / 对数标准差
        a_std = a_std_log.clamp(-16, 2).exp()        # 限制范围后取指数 → 标准差(保证>0)
        a_avg = a_avg.squeeze(-1)
        a_std = a_std.squeeze(-1)

        dist = Normal(a_avg, a_std)                  # 构造正态分布 N(a_avg, a_std)
        logits = dist.rsample()                      # ★ 重参数化采样(论文公式6的 f_φ(ε;s)) ★
        indices = torch.sort(logits)[1]
        soft_logits = logits * torch.log(indices + 1)
        weight = F.softmax(soft_logits, dim=-1).squeeze(-1)
        return weight

    def get_action_logprob(self, x):
        """
        采样动作的同时返回"对数概率 logprob"。
        SAC 的熵正则项 α·logπ(a|s) 需要这个 logprob (见 mask_sac.py 的 update_net)。
        """
        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)
        a_avg, a_std_log = pred.chunk(2, dim=-1)
        a_std = a_std_log.clamp(-16, 2).exp()
        a_avg = a_avg.squeeze(-1)
        a_std = a_std.squeeze(-1)

        dist = Normal(a_avg, a_std)
        logits = dist.rsample()
        indices = torch.sort(logits)[1]
        soft_logits = logits * torch.log(indices + 1)
        weight = F.softmax(soft_logits, dim=-1).squeeze(-1)

        # 计算对数概率, 并做 tanh/softmax 修正(SAC 标准做法: 把动作经过非线性后要修正 logprob)
        logprob = dist.log_prob(a_avg)
        logprob -= (-weight.pow(2) + 1.000001).log()
        return weight, logprob.sum(1)                # 对各维 logprob 求和


@NET.register_module(force=True)   # 注册名 "CriticMaskSAC"
class CriticMaskSAC(nn.Module):
    """
    评论家(价值网络) Q_θ。输入"表征 ρ + 动作 a", 输出该 (状态,动作) 的 Q 值。
    ★ 双 Q 设计: 输出 2 个 Q 值(q1, q2), 用来抑制 Q 值高估(SAC/TD3 标准技巧)。
    """
    def __init__(self,
                 *args,
                 embed_dim: int = 128,
                 depth: int = 2,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 cls_embed: bool = True,
                 **kwargs):
        super(CriticMaskSAC, self).__init__()
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.cls_embed = cls_embed

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 注意输入维度是 embed_dim + 1: 因为要把"动作"(1维)拼接到每个 token 的表征后面
        self.blocks = nn.ModuleList(
            [Mlp(in_features=embed_dim + 1, hidden_features=embed_dim + 1, out_features=embed_dim + 1) for i in range(depth)]
        )
        self.norm = norm_layer(embed_dim + 1)
        # 输出 2 个值 = 两个 Q 网络(q1, q2)
        self.decoder_pred = nn.Linear(embed_dim + 1, 2, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, action):
        """加 [CLS] → 把动作拼到每个 token 后面 → 过 MLP。"""
        if self.cls_embed:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)        # (B, N+1, C)
        if len(action.shape) == 2:
            action = action.unsqueeze(-1)                 # (B, N+1) → (B, N+1, 1)
        x = torch.concat([x, action], dim=-1)             # 表征与动作拼接 → (B, N+1, C+1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_decoder(self, x):
        return self.decoder_pred(x)

    def forward(self, x, action):
        """返回两个 Q 的均值(更新 Actor 时用)。"""
        latent = self.forward_encoder(x, action)
        latent = torch.sum(latent, dim=1)         # 对所有 token 求和 → (B, C+1)
        pred = self.forward_decoder(latent)        # → (B, 2)
        value = pred.mean(dim=1)                    # 两个 Q 取均值
        return value

    def get_q_min(self, x, action):
        """返回两个 Q 的较小值 —— 算 TD 目标时用(抑制高估)。"""
        latent = self.forward_encoder(x, action)
        latent = torch.sum(latent, dim=1)
        pred = self.forward_decoder(latent)
        value = pred.min(dim=1)[0]                  # ★ 取 min ★
        return value

    def get_q1_q2(self, x, action):
        """分别返回 q1, q2 —— 训练 Critic 时两个都要算 TD 误差。"""
        latent = self.forward_encoder(x, action)
        latent = torch.sum(latent, dim=1)
        pred = self.forward_decoder(latent)
        value = pred
        return value[:, 0], value[:, 1]

# ============================================================================
#  小结(对照论文与学习清单):
#   - Actor = 学习清单第9部分的"策略网络", 用重参数化采样(rsample)对应论文公式(6)。
#   - Critic = 学习清单第2.3节的"Q 函数", 双 Q 取 min 对应 SAC 抑制高估的技巧。
#   - 两个网络都不直接吃原始行情, 而是吃 MAE 产出的"可掩码股票表征 ρ"
#     —— 这正是 EarnMore 把"表征学习"和"强化学习"结合的接口所在。
# ============================================================================
