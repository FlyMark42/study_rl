# -*- coding: utf-8 -*-
# ============================================================================
#  文件: pm/net/mask_time_state.py  （带详细中文注释版）
#  作用: 论文§4.1 的实际表征网络 MaskTimeState（继承自 MAE）。
#        它在 MAE 基础上做了两件关键的事:
#          1) 用"交叉注意力(CrossAttention)"做解码器——重构时还能参考"完整未掩码序列",
#             重构得更准、捕捉股票间关系更强。
#          2) 提供 forward_state() 接口——专门给强化学习用, 只产出"可掩码股票表征 ρ",
#             不算重构损失(重构损失在 forward() 里单独算)。
#  ★ 一句话: forward_state() 出"状态给 RL 用"; forward() 出"重构损失给自监督训练用"。★
# ============================================================================
import torch
from typing import List
from functools import partial
from pm.registry import NET
from pm.net import MAE
from einops import rearrange, repeat
from typing import Final, Set, Optional, Union, Tuple
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, DropPath, use_fused_attn


class LayerScale(nn.Module):
    """LayerScale: 给残差分支乘一个可学习的小系数 γ, 让深层 Transformer 训练更稳。"""
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class CrossAttention(nn.Module):
    """
    交叉注意力: Q 来自一个序列, K/V 来自另一个序列。
    在这里: Q = "被 mask_token 填充的待重构序列", K/V = "完整未掩码序列"。
    这样解码器在重构被掩盖股票时, 能"查阅"所有真实存在的股票, 更好地利用股票间关系。
    """
    fused_attn: Final[bool]

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False,
                 attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, 'dim 必须能被 num_heads 整除'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        # 注意: Q/K/V 分开三个线性层(因为来源不同)
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossBlock(nn.Module):
    """用 CrossAttention 的 Transformer 块(解码器用)。结构同标准块, 只是注意力换成交叉注意力。"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False,
                 proj_drop=0., attn_drop=0., init_values=None, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_layer=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, k, v):
        # q 自己更新, 但注意力的 k/v 来自外部(完整序列)
        q = q + self.drop_path1(self.ls1(self.attn(self.norm1(q), k, v)))
        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q


@NET.register_module(force=True)   # 注册名 "MaskTimeState"(配置里 rep_net.type 用它)
class MaskTimeState(MAE):
    """论文§4.1 的可掩码股票表征网络。继承 MAE, 把解码器换成交叉注意力块, 并新增 forward_state()。"""
    def __init__(self, *args, embed_type="TimesEmbed", feature_size=(10, 99), patch_size=(10, 99),
                 t_patch_size=1, num_stocks=420, pred_num_stocks=420, in_chans=1, embed_dim=128,
                 depth=2, num_heads=4, decoder_embed_dim=64, decoder_depth=1, decoder_num_heads=8,
                 mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False,
                 cls_embed=True, sep_pos_embed=True, trunc_init=False, no_qkv_bias=False,
                 mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25, **kwargs):
        # 先调用父类 MAE 构造所有零件
        super(MaskTimeState, self).__init__(*args, embed_type=embed_type, feature_size=feature_size,
            patch_size=patch_size, t_patch_size=t_patch_size, num_stocks=num_stocks, pred_num_stocks=pred_num_stocks,
            in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads, decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss, cls_embed=cls_embed, sep_pos_embed=sep_pos_embed, trunc_init=trunc_init,
            no_qkv_bias=no_qkv_bias, mask_ratio_min=mask_ratio_min, mask_ratio_max=mask_ratio_max,
            mask_ratio_mu=mask_ratio_mu, mask_ratio_std=mask_ratio_std, **kwargs)
        # ★ 关键: 用 CrossBlock(交叉注意力)替换父类的普通解码器块 ★
        self.decoder_blocks = nn.ModuleList(
            [CrossBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=not no_qkv_bias, norm_layer=norm_layer)
             for i in range(decoder_depth)])
        self.initialize_weights()

    def initialize_weights(self):
        # (初始化细节, 与父类类似, 略读)
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)
            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        if getattr(self.patch_embed, "proj", None) is not None:
            w = self.patch_embed.proj.weight.data
            if self.trunc_init:
                torch.nn.init.trunc_normal_(w)
                torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 这里线性层用正交初始化(和父类的 xavier 略不同)
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, mask=None, ids_restore=None, if_mask=True):
        """
        编码器。相比父类多了三种工作模式:
          - mask=None 且 if_mask=True : 随机生成掩码(训练采样新数据时)
          - mask 已给定               : 复用外部传入的掩码(从回放缓冲区取数据时, 保证一致)
          - if_mask=False             : 完全不掩码(用于得到"完整序列"当交叉注意力的 K/V)
        """
        x = self.patch_embed(x)   # 股票级嵌入
        B, L, C = x.shape
        if if_mask:
            if mask is None:
                # 随机采样掩码率并掩码
                mask_ratio = self.mask_ratio_generator.rvs(1)[0]
                x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
                x = x.view(B, -1, C)
            else:
                # 复用外部掩码: 根据 ids_restore 和 mask 算出要保留哪些, 并取出
                ids_keep = torch.argsort(ids_restore, dim=1)[:, :(mask[0, :] == 0).sum().item()]
                x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))
        else:
            # 不掩码: 保留全部股票(ids_keep = 0..L-1)
            ids_keep = torch.arange(0, L).unsqueeze(0).repeat(B, 1).to(x.device)

        if self.cls_embed:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        # 位置编码(同父类逻辑)
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(1, self.input_size[0], 1) + \
                torch.repeat_interleave(self.pos_embed_temporal, self.input_size[1], dim=1)
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(pos_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]))
            if self.cls_embed:
                pos_embed = torch.cat([self.pos_embed_class.expand(pos_embed.shape[0], -1, -1), pos_embed], 1)
        else:
            cls_ind = 1 if self.cls_embed else 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(pos_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]))
            if self.cls_embed:
                pos_embed = torch.cat([self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1), pos_embed], 1)
        x = x.view([B, -1, C]) + pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.cls_embed:
            x = x[:, 1:, :]
        return x, mask, ids_restore

    def forward_decoder(self, x, kv, ids_restore):
        """解码器(交叉注意力版): x=被填充的待重构序列, kv=完整未掩码序列。重构被掩盖股票价格。"""
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size
        x = self.decoder_embed(x)
        kv = self.decoder_embed(kv)   # ← 完整序列也投影到解码器维度, 作为 K/V
        C = x.shape[-1]
        # 用 mask_token 填补被掩盖位置 → 还原顺序
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))
        x = x_.view([N, T * H * W, C])
        if self.cls_embed:
            decoder_cls_tokens = self.decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)
            kv = torch.cat((decoder_cls_tokens, kv), dim=1)
        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(1, self.input_size[0], 1) + \
                torch.repeat_interleave(self.decoder_pos_embed_temporal, self.input_size[1] * self.input_size[2], dim=1)
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [self.decoder_pos_embed_class.expand(decoder_pos_embed.shape[0], -1, -1), decoder_pos_embed], 1)
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]
        x = x + decoder_pos_embed
        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T, H * W, C])
            kv = kv.view([N, T, H * W, C])
        k = v = kv
        for blk in self.decoder_blocks:   # 交叉注意力解码: q=x, k=v=完整序列
            x = blk(x, k, v)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        if requires_t_shape:
            x = x.view([N, T * H * W, -1])
        if self.cls_embed:
            x = x[:, 1:, :]
        return x

    def forward_state(self, x, mask=None, ids_restore=None):
        """
        ★★★ 强化学习专用接口 ★★★  产出"可掩码股票表征 ρ"。
        流程: 编码 → 用 mask_token 填补 → 还原顺序 → 加位置编码 → (不过解码 Transformer, 直接返回)。
        注意: 它只产生"状态表征", 不做重构、不算损失。Agent 拿这个 ρ 喂给 Actor/Critic。
        返回: (ρ, mask, ids_restore)
        """
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        # 编码(可复用外部 mask/ids_restore, 保证训练时 state 和 next_state 用同样掩码)
        x, mask, ids_restore = self.forward_encoder(x, mask=mask, ids_restore=ids_restore)
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size
        x = self.decoder_embed(x)
        C = x.shape[-1]
        # 用 mask_token 填补 → 还原顺序(这一步得到论文的 ρ = η_mf(l_p, m))
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))
        x = x_.view([N, T * H * W, C])
        if self.cls_embed:
            decoder_cls_tokens = self.decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)
        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(1, self.input_size[0], 1) + \
                torch.repeat_interleave(self.decoder_pos_embed_temporal, self.input_size[1] * self.input_size[2], dim=1)
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [self.decoder_pos_embed_class.expand(decoder_pos_embed.shape[0], -1, -1), decoder_pos_embed], 1)
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]
        x = x + decoder_pos_embed
        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T * H * W, -1])
        if self.cls_embed:
            x = x[:, 1:, :]   # 去掉 [CLS], 返回 (B, N, C) 的表征 ρ
        return x, mask, ids_restore

    def forward_loss(self, imgs, pred, mask):
        """重构损失(论文公式8), 只在被掩盖股票上算 MSE。逻辑同父类。"""
        _imgs = torch.index_select(imgs, 2,
            torch.linspace(0, imgs.shape[2] - 1, self.num_stocks).long().to(imgs.device))
        target = self.patchify(_imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        mask = mask.view(loss.shape)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, x, mask=None, ids_restore=None):
        """
        自监督训练接口: 返回重构损失。
        关键: 它编码两次——一次"带掩码"得 latent(query), 一次"不带掩码"得 kv(完整序列),
        然后用交叉注意力解码、算重构损失。Agent 的 update_rep_net 调用的就是这个。
        """
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        latent, mask, ids_restore = self.forward_encoder(x, mask=mask, ids_restore=ids_restore)  # 带掩码
        kv = self.forward_encoder(x, if_mask=False)[0]   # 不带掩码 → 完整序列作为 K/V
        pred = self.forward_decoder(latent, kv, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, mask, ids_restore

# ============================================================================
#  小结 —— 这个类是 EarnMore 第一大创新的核心载体:
#   * forward_state(): 给 RL 用, 输出"可掩码股票表征 ρ"(论文的状态 s_t)。
#   * forward():       给自监督用, 输出重构损失(论文公式8)。
#   * 两者在 Agent.update_net() 里"同时优化"(论文强调的端到端、不预训练)。
#   * CrossAttention 解码器让模型在重构缺失股票时能参考完整股票序列, 强化"股票间关系"。
# ============================================================================
