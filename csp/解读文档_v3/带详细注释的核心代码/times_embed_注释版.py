# -*- coding: utf-8 -*-
# ============================================================================
#  文件: pm/embed/times_embed.py  （带详细中文注释版）
#  作用: 论文模块(a) —— "股票级嵌入(Stock-level Embedding)", 对应论文公式(2)
#        l_s(X_t) = ψ_e(D_t; θ_e) + ψ_c(P_t, Y_t; θ_c)
#        即: 把每只股票"近 D 天 × F 个特征"的原始数据, 压成一个 embed_dim 维向量。
#  借鉴: TimesNet 的做法(价值卷积 + 时序嵌入 + 位置嵌入 相加)
# ============================================================================
import torch
import math
import torch.nn as nn
from pm.registry import EMBED
from einops import rearrange
from typing import List
from timm.models.layers import to_2tuple


class PositionalEmbedding(nn.Module):
    """
    位置嵌入(正弦/余弦), 来自 Transformer 经典做法。
    作用: 告诉模型"序列中每个位置(这里是第几天)的先后", 因为纯注意力本身不区分顺序。
    它不含可学习参数, 用 sin/cos 公式直接算好存为 buffer。
    """
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False  # 位置编码固定, 不参与训练

        position = torch.arange(0, max_len).float().unsqueeze(1)
        # 不同维度用不同频率的 sin/cos (经典 Transformer 公式)
        div_term = (torch.arange(0, embed_dim, 2).float()
                    * -(math.log(10000.0) / embed_dim)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维用 cos
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 只取前 x.size(1) 个位置的编码
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    价值嵌入(ψ_c 的核心): 用一维卷积把"F 个原始特征"映射到"embed_dim 维"。
    一维卷积沿"天数"方向滑动, 能捕捉相邻几天的局部模式。
    """
    def __init__(self, input_dim, embed_dim):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # in_channels=特征数, out_channels=嵌入维度, kernel_size=3(看相邻3天)
        # padding_mode='circular' 循环填充, 避免边界丢信息
        self.tokenConv = nn.Conv1d(in_channels=input_dim,
                                   out_channels=embed_dim,
                                   kernel_size=3,
                                   padding=padding,
                                   padding_mode='circular',
                                   bias=False)
        # 用 Kaiming 初始化卷积权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: (样本, 天数, 特征) → Conv1d 要求 (样本, 通道=特征, 天数), 所以先 permute
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x  # 输出 (样本, 天数, embed_dim)


class FixedEmbedding(nn.Module):
    """
    固定(不可训练)的查表嵌入, 用 sin/cos 初始化。用于离散的时间类别(星期/日/月)。
    """
    def __init__(self, input_dim, embed_dim):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(input_dim, embed_dim).float()
        w.require_grad = False
        position = torch.arange(0, input_dim).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float()
                    * -(math.log(10000.0) / embed_dim)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(input_dim, embed_dim)
        self.emb.weight = nn.Parameter(w, requires_grad=False)  # 固定权重

    def forward(self, x):
        return self.emb(x).detach()


class TimeFeatureEmbedding(nn.Module):
    """另一种时序嵌入: 直接用线性层把 3 维时间特征映射到 embed_dim (当 embed_type='timeF' 时用)。"""
    def __init__(self, embed_dim, embed_type='timeF'):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(3, embed_dim, bias=False)

    def forward(self, x):
        return self.embed(x)


class TemporalEmbedding(nn.Module):
    """
    时序嵌入(ψ_e): 把"星期几、几号、几月"三个离散时间特征分别查表成向量, 再相加。
    对应论文公式(2)里的 ψ_e(D_t; θ_e) 部分。
    """
    def __init__(self, embed_dim, embed_type='fixed'):
        super(TemporalEmbedding, self).__init__()
        weekday_size = 7   # 星期: 0~6
        day_size = 32      # 日: 1~31 (留一点余量)
        month_size = 13    # 月: 1~12

        # embed_type='fixed' 用固定嵌入, 否则用可训练的 nn.Embedding
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        self.weekday_embed = Embed(weekday_size, embed_dim)
        self.day_embed = Embed(day_size, embed_dim)
        self.month_embed = Embed(month_size, embed_dim)

    def forward(self, x):
        x = x.long()
        # 分别查表: 第0列=星期, 第1列=日, 第2列=月
        weekday_x = self.weekday_embed(x[:, :, 0])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 2])
        return weekday_x + day_x + month_x  # 三者相加


@EMBED.register_module(force=True)   # 注册进 EMBED 电话簿, 名字 "TimesEmbed"
class TimesEmbed(nn.Module):
    """
    ★ 股票级嵌入的总装类 ★  对应论文公式(2):
       l_s(X_t) = 价值嵌入(卷积) + 时序嵌入(查表) + 位置嵌入(sin/cos)
    输入: (B, C, N, D, F) = (批量, 通道=1, 股票数, 天数, 特征数)
    输出: (B, N, embed_dim)  —— 每只股票一个向量
    """
    def __init__(self,
                 *args,
                 img_size=(10, 102),      # (天数, 特征数)
                 patch_size=(10, 102),
                 frames=420,              # 股票数 N
                 t_patch_size=1,
                 input_dim: int = 102,    # 总特征数(含时序)
                 temporal_dim: int = 3,   # 时序特征数(星期/日/月)
                 embed_dim: int = 128,    # 输出维度
                 embed_type='fixed',
                 **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.temporal_dim = temporal_dim
        self.embed_dim = embed_dim

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # 一些维度整除性检查
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert frames % t_patch_size == 0
        num_patches = ((img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (frames // t_patch_size))
        # input_size = (股票数, 1, 1) —— MAE 里会用到的"网格尺寸"
        self.input_size = (frames // t_patch_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.grid_size = img_size[0] // patch_size[0]
        self.t_grid_size = frames // t_patch_size

        # 真正的(非时序)特征数 = 总特征 - 时序特征
        self.feature_dim = self.input_dim - temporal_dim

        # 三个嵌入子模块
        self.value_embedding = TokenEmbedding(input_dim=self.feature_dim, embed_dim=embed_dim)  # ψ_c 卷积
        self.position_embedding = PositionalEmbedding(embed_dim=embed_dim)                       # 位置
        self.temporal_embedding = TemporalEmbedding(embed_dim=embed_dim, embed_type=embed_type) \
            if embed_type != 'timeF' else TimeFeatureEmbedding(embed_dim=embed_dim, embed_type=embed_type)  # ψ_e 时序

    def forward(self, x):
        B, C, N, D, F = x.shape  # (批量, 通道, 股票数, 天数, 特征数)

        # 把前三维合并, 方便逐"股票"独立做嵌入: (B*C*N, D, F)
        x = rearrange(x, "b c n d f -> (b c n) d f", b=B, c=C, n=N)

        # 拆分: 前面是普通特征, 最后 temporal_dim 列是时序特征
        feature = x[..., :-self.temporal_dim]      # (B*C*N, D, feature_dim)
        temporal = x[..., -self.temporal_dim:]     # (B*C*N, D, temporal_dim)

        # ★ 论文公式(2): 价值嵌入 + 时序嵌入 + 位置嵌入 三者相加 ★
        x = self.value_embedding(feature) + self.temporal_embedding(temporal) + self.position_embedding(feature)

        # 还原回 (B, C, N, D, embed_dim)
        x = rearrange(x, "(b c n) d f -> b c n d f", b=B, c=C, n=N)
        # 沿"天数"维求平均(把 D 天压成 1 个向量), 再去掉通道维 → (B, N, embed_dim)
        x = x.mean(dim=-2).squeeze(1)
        return x


@EMBED.register_module(force=True)
class TimesEmbedWoPos(nn.Module):
    """与 TimesEmbed 完全相同, 只是去掉了位置嵌入(WoPos = without Position)。备选方案, 主线没用。"""
    def __init__(self, *args, img_size=(10, 102), patch_size=(10, 102), frames=420, t_patch_size=1,
                 input_dim: int = 102, temporal_dim: int = 3, embed_dim: int = 128, embed_type='fixed', **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.temporal_dim = temporal_dim
        self.embed_dim = embed_dim
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert frames % t_patch_size == 0
        num_patches = ((img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (frames // t_patch_size))
        self.input_size = (frames // t_patch_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.grid_size = img_size[0] // patch_size[0]
        self.t_grid_size = frames // t_patch_size
        self.feature_dim = self.input_dim - temporal_dim
        self.value_embedding = TokenEmbedding(input_dim=self.feature_dim, embed_dim=embed_dim)
        self.temporal_embedding = TemporalEmbedding(embed_dim=embed_dim, embed_type=embed_type) \
            if embed_type != 'timeF' else TimeFeatureEmbedding(embed_dim=embed_dim, embed_type=embed_type)

    def forward(self, x):
        B, C, N, D, F = x.shape
        x = rearrange(x, "b c n d f -> (b c n) d f", b=B, c=C, n=N)
        feature = x[..., :-self.temporal_dim]
        temporal = x[..., -self.temporal_dim:]
        x = self.value_embedding(feature) + self.temporal_embedding(temporal)  # 没有 + position
        x = rearrange(x, "(b c n) d f -> b c n d f", b=B, c=C, n=N)
        x = x.mean(dim=-2).squeeze(1)
        return x

# ============================================================================
#  小结: 这一层做的事很简单 —— "把每只股票近 D 天的原始行情, 浓缩成 1 个向量"。
#  之所以叫"股票级(stock-level)", 是因为它只看单只股票内部的时间信息(纵向),
#  还没考虑股票之间的关系(横向)。股票之间的关系交给下一步的 MAE(池级)去捕捉。
# ============================================================================
