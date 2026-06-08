# 带详细中文注释的核心代码

> 这里是 EarnMore 工程**最核心的 7 个文件**的逐行中文注释版，专为学习而做。
>
> - 这些文件是从 `EarnMore-main/pm/` 下的原始文件**复制后加注释**的，**不影响原工程运行**。
> - 注释里大量标注了"← 对应论文公式 X / 学习清单第 Y 部分"，请配合 `01_论文全文翻译.md`、`02_强化学习知识点学习清单.md`、`03_EarnMore代码工程解析.md` 一起看。
> - 文件名带 `_注释版` 后缀，与原文件区分。

## 建议阅读顺序（由易到难）

| 顺序 | 文件 | 对应论文 | 一句话 |
|---|---|---|---|
| 1 | `times_embed_注释版.py` | §4.1 公式(2) | 股票级嵌入：把原始特征变成向量 |
| 2 | `mae_注释版.py` | §4.1 公式(3)(4)(8) | 掩码自编码器骨架(掩码/编码/解码/重构) |
| 3 | `mask_time_state_注释版.py` | §4.1 | 表征网络：产出"可掩码股票表征 ρ" |
| 4 | `mask_sac_net_注释版.py` | §4.2 公式(6) | Actor(策略) + Critic(价值) 网络 |
| 5 | `helpers_注释版.py` | §4.3 公式(9) | 重加权：温度 softmax + 掩码忽略 |
| 6 | `pm_based_portfolio_ASR_注释版.py` | §3.2 | RL 环境：step/reset、算收益和奖励 |
| 7 | `mask_sac_注释版.py` | 算法1, 公式(5)(7)(8) | ★智能体：把一切串成完整 SAC 训练 |

> **学懂的标志**：能对着第 7 个文件的 `update_net()`，逐行说出每步更新哪个网络、对应论文哪个公式、为什么这么做。

## 原文件 → 注释版 对照

| 注释版 | 原始文件 |
|---|---|
| `times_embed_注释版.py` | `pm/embed/times_embed.py` |
| `mae_注释版.py` | `pm/net/mae.py` |
| `mask_time_state_注释版.py` | `pm/net/mask_time_state.py` |
| `mask_sac_net_注释版.py` | `pm/net/sac/mask_sac_net.py` |
| `helpers_注释版.py` | `pm/utils/helpers.py` |
| `pm_based_portfolio_ASR_注释版.py` | `pm/environment/pm_based_portfolio_ASR.py` |
| `mask_sac_注释版.py` | `pm/agent/sac/mask_sac.py` |
