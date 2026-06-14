# -*- coding: utf-8 -*-
# ============================================================================
#  文件: pm/net/mae.py  （带详细中文注释版）
#  作用: 掩码自编码器(Masked Autoencoder, MAE)的基类 —— 论文模块(a)(b)的骨架。
#        对应论文第 4.1 节: 通过"随机掩盖一部分股票 → 编码 → 用掩码标记填回 → 解码重构"
#        来学习"池级表征", 并产生重构损失(论文公式 3/4/8)。
#  关键概念: 随机掩码、未掩码送编码器(省算力)、可学习 mask_token、[CLS] 标记、位置编码。
#  说明: MaskTimeState(实际使用的表征网络)继承自本类, 见 mask_time_state_注释版.py。
# ============================================================================
import torch
import torch.nn as nn
from typing import List
import scipy.stats as stats
from functools import partial
from timm.models.vision_transformer import DropPath, Mlp
from pm.registry import EMBED


class Attention(nn.Module):
    """标准的多头自注意力(Self-Attention)。让序列里每个元素都"看一眼"其他所有元素, 捕捉关系。"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, input_size=(4, 14, 14)):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5   # 缩放因子 1/sqrt(d), 防止点积过大
        # Q/K/V 三个线性投影
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x):
        B, N, C = x.shape
        # 把 Q/K/V 拆成多头: (B, num_heads, N, head_dim)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # 注意力分数 = Q·Kᵀ / sqrt(d), 再 softmax
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # 用注意力权重对 V 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class Block(nn.Module):
    """一个标准 Transformer 块: (LayerNorm→注意力→残差) + (LayerNorm→MLP→残差)。"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_func=Attention):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))  # 残差连接1: 注意力
        x = x + self.drop_path(self.mlp(self.norm2(x)))    # 残差连接2: MLP
        return x


class MAE(nn.Module):
    """
    掩码自编码器基类。负责构造 编码器/解码器/位置编码/掩码标记 等所有零件,
    并实现 random_masking / forward_encoder / forward_decoder / forward_loss 等核心方法。
    """
    def __init__(self,
                 *args,
                 embed_type: str = "PatchEmbed",
                 feature_size: List[int] = (10, 99),   # (天数, 特征数)
                 patch_size: List[int] = (10, 99),
                 t_patch_size: int = 1,
                 num_stocks: int = 420,                 # 股票数 N
                 pred_num_stocks: int = 420,
                 in_chans: int = 1,
                 input_dim: int = 102,
                 temporal_dim: int = 3,
                 embed_dim: int = 128,                  # 编码器嵌入维度
                 depth: int = 2,                        # 编码器层数
                 num_heads: int = 4,
                 decoder_embed_dim: int = 64,           # 解码器嵌入维度
                 decoder_depth: int = 1,
                 decoder_num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 norm_pix_loss: bool = False,
                 cls_embed: bool = True,                # 是否用 [CLS] 标记
                 sep_pos_embed: bool = True,            # 位置编码是否分"空间(股票)+时间"
                 trunc_init: bool = False,
                 no_qkv_bias: bool = False,
                 mask_ratio_min: float = 0.5,           # 掩码率下界 a
                 mask_ratio_max: float = 1.0,           # 掩码率上界 b
                 mask_ratio_mu: float = 0.55,           # 掩码率均值 μ
                 mask_ratio_std: float = 0.25,          # 掩码率标准差 σ
                 **kwargs):
        super(MAE, self).__init__()
        self.embed_type = embed_type
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.num_stocks = num_stocks
        self.pred_num_stocks = pred_num_stocks
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.norm_pix_loss = norm_pix_loss
        self.cls_embed = cls_embed
        self.sep_pos_embed = sep_pos_embed
        self.trunc_init = trunc_init
        self.no_qkv_bias = no_qkv_bias
        self.t_pred_patch_size = t_patch_size * pred_num_stocks // num_stocks

        # 掩码率分布的参数(论文公式3: 截断高斯)
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.mask_ratio_mu = mask_ratio_mu
        self.mask_ratio_std = mask_ratio_std

        # 构造"嵌入层"(用注册器, 默认是 TimesEmbed) —— 论文模块(a)
        self.emb_config = dict(type=embed_type, img_size=feature_size, patch_size=feature_size,
                               in_chans=in_chans, input_dim=input_dim, temporal_dim=temporal_dim,
                               embed_dim=embed_dim, frames=num_stocks, t_patch_size=t_patch_size)
        self.patch_embed = EMBED.build(self.emb_config)
        self.num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        # ★ 掩码率生成器: scipy 的截断正态分布, 把采样限制在 [min, max] 区间 ★ (论文公式3)
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
            (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
            loc=mask_ratio_mu, scale=mask_ratio_std)

        # [CLS] 标记(编码器和解码器各一个)
        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # 位置编码: sep 模式下分"空间(股票位置)"和"时间"两套, 相加得到完整位置编码
        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, input_size[1] * input_size[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, input_size[0], embed_dim))
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            _num_patches = self.num_patches + 1 if self.cls_embed else self.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, _num_patches, embed_dim))

        # 编码器: depth 个 Transformer 块
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=not no_qkv_bias, qk_scale=None, norm_layer=norm_layer)
             for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # 编码器维度 → 解码器维度 的线性投影
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # ★ 可学习的掩码标记 mask_token ★ —— 论文中的 [M], 用来"填补"被掩盖的股票位置
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # 解码器的位置编码
        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(torch.zeros(1, input_size[1] * input_size[2], decoder_embed_dim))
            self.decoder_pos_embed_temporal = nn.Parameter(torch.zeros(1, input_size[0], decoder_embed_dim))
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        else:
            _num_patches = self.num_patches + 1 if self.cls_embed else self.num_patches
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, _num_patches, decoder_embed_dim))

        # 解码器: decoder_depth 个 Transformer 块
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=not no_qkv_bias, qk_scale=None, norm_layer=norm_layer)
             for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        # 预测头: 把解码器输出映射回"原始价格 patch"的维度, 用于重构
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.t_pred_patch_size * patch_size[0] * patch_size[1] * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # 各种参数的初始化(细节, 不影响理解, 略读)
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
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """把原始输入切成 patch(这里一只股票 = 一个 patch), 用于和重构结果对比算损失。"""
        N, _, T, H, W = imgs.shape
        p1 = self.patch_embed.patch_size[0]
        p2 = self.patch_embed.patch_size[1]
        u = self.t_pred_patch_size
        h = H // p1
        w = W // p2
        t = T // u
        x = imgs.reshape(shape=(N, self.in_chans, t, u, h, p1, w, p2))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p1 * p2 * self.in_chans))
        self.patch_info = (N, T, H, W, p1, p2, u, t, h, w)
        return x

    def unpatchify(self, x):
        """patchify 的逆操作(本主线没直接用)。"""
        N, T, H, W, p1, p2, u, t, h, w = self.patch_info
        x = x.reshape(shape=(N, t, h, w, u, p1, p2, self.in_chans))
        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, self.in_chans, T, H, W))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        ★★ MAE 的灵魂: 随机掩码 ★★  对应论文掩码操作 η_mo。
        给每只股票一个随机噪声 → 按噪声排序 → 保留前 len_keep 只(未掩码), 其余丢弃。
        返回:
          x_masked   : 只剩"未掩码"股票的嵌入(送进编码器, 省算力——这是 MAE 高效的关键)
          mask       : 二值掩码 (0=保留, 1=被掩盖)
          ids_restore: 还原索引(把"打乱+丢弃"后的顺序, 还原回原始 N 只的顺序, 解码时要用)
          ids_keep   : 被保留股票的原始索引
        """
        N, L, D = x.shape   # 批量, 序列长度(股票数), 维度
        len_keep = int(L * (1 - mask_ratio))   # 保留的数量

        noise = torch.rand(N, L, device=x.device)   # 每只股票一个 [0,1) 随机数
        ids_shuffle = torch.argsort(noise, dim=1)    # 按噪声排序得到打乱索引
        ids_keep = ids_shuffle[:, :len_keep].sort()[0]    # 前 len_keep 个=保留(再排序保持稳定)
        ids_nokeep = ids_shuffle[:, len_keep:].sort()[0]  # 其余=丢弃
        ids_shuffle = torch.concat([ids_keep, ids_nokeep], dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)    # 还原索引

        # 只把"保留"的股票嵌入取出来
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 生成二值掩码: 前 len_keep 个为 0(保留), 其余为 1(掩盖), 再按 ids_restore 还原顺序
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x):
        """嵌入 → 随机掩码 → 加[CLS]和位置编码 → 过编码器 Transformer。"""
        x = self.patch_embed(x)             # 股票级嵌入 (B, L, C)
        B, L, C = x.shape
        mask_ratio = self.mask_ratio_generator.rvs(1)[0]   # 采样一个掩码率(截断高斯)
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)  # 掩码
        x = x.view(B, -1, C)

        if self.cls_embed:                  # 加 [CLS]
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        # 加位置编码(只给保留下来的位置加, 用 ids_keep 取对应的位置编码)
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
        for blk in self.blocks:             # 过编码器
            x = blk(x)
        x = self.norm(x)
        if self.cls_embed:
            x = x[:, 1:, :]                  # 去掉 [CLS], 返回各股票的潜在表征
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """用 mask_token 把潜在表征补齐到 N 个 → 还原顺序 → 过解码器 → 预测被掩盖股票的价格。"""
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size
        x = self.decoder_embed(x)            # 维度投影到解码器维度
        C = x.shape[-1]
        # ★ 用 mask_token 填补被掩盖的位置 ★ (论文掩码填充 η_mf)
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)
        x_ = x_.view([N, T * H * W, C])
        # 按 ids_restore 还原成原始 N 只股票的顺序(unshuffle)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))
        x = x_.view([N, T * H * W, C])
        if self.cls_embed:
            decoder_cls_tokens = self.decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)
        # 加解码器位置编码
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
        for blk in self.decoder_blocks:      # 过解码器
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)             # 预测头 → 重构的价格
        if requires_t_shape:
            x = x.view([N, T * H * W, -1])
        if self.cls_embed:
            x = x[:, 1:, :]                  # 去掉 [CLS]
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        重构损失(论文公式8): 只在"被掩盖"的股票上算 MSE。
        逻辑: target=真实价格 patch, pred=重构价格, 算 (pred-target)², 再用 mask 只保留被掩盖位置。
        """
        _imgs = torch.index_select(imgs, 2,
            torch.linspace(0, imgs.shape[2] - 1, self.num_stocks).long().to(imgs.device))
        target = self.patchify(_imgs)
        if self.norm_pix_loss:               # 可选: 对目标做归一化
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2          # 逐元素平方误差
        loss = loss.mean(dim=-1)             # 每只股票一个损失
        mask = mask.view(loss.shape)
        loss = (loss * mask).sum() / mask.sum()   # ★ 只在被掩盖股票上求平均 ★
        return loss

    def forward(self, x):
        """完整前向(自监督训练用): 编码 → 解码 → 算重构损失。"""
        if len(x.shape) == 4:
            x = x.unsqueeze(1)               # 补一个通道维
        latent, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask

# ============================================================================
#  小结(对照论文 §4.1):
#   - random_masking      = 论文掩码操作 η_mo + 论文公式(3)的掩码率采样
#   - forward_encoder     = 论文编码器 ψ_enc(只编码"未掩码"股票, 省算力)
#   - mask_token 填补      = 论文掩码填充 η_mf, 得到"可掩码股票表征 ρ"的前身
#   - forward_decoder     = 论文解码器 ψ_dec(重构被掩盖股票)
#   - forward_loss        = 论文公式(8)的重构损失(只算被掩盖股票)
#  为什么这样设计能"训练一次、适配任意子池"?
#   —— 训练时随机掩盖各种比例/组合的股票, 模型见过海量"子池", 学会了用剩余股票
#      推断缺失股票、并保留股票间关系; 推理时把"投资者排除的股票"当作"被掩盖"即可。
# ============================================================================
