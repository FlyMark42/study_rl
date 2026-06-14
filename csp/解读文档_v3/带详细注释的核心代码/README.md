# 带详细中文注释的核心代码 · v3（新版 EarnMore-main_v20260610）

> 本目录是新版代码核心文件的**逐行中文注释版**，自成一套、可独立阅读。注释里大量标注"← 对应论文公式 X / 学习清单第 Y 部分"，请配合上层的 `EarnMore_论文中英对照.md`、`02_强化学习知识点学习清单.md`、`03_EarnMore代码工程解析_新版.md` 一起看。

## 一、注释来源与新旧版关系

经逐字节比对，**新版相对旧版，核心算法文件几乎没变**，唯一重大改动是环境被重写。因此本目录的注释分两类：

- **复用既有注释**（代码与旧版逐字节相同，注释 100% 适用）：来自 `../../解读文档/带详细注释的核心代码/`，已复制进本目录使其自包含。
- **新增注释**（旧版注释集未覆盖，或新版重写）：本次新写。

## 二、完整文件清单（9 份，覆盖从数据到训练的整条流水线）

| 阅读顺序 | 注释文件 | 对应原文件 | 对应论文 | 来源 | 一句话 |
|---|---|---|---|---|---|
| 1 | `times_embed_注释版.py` | `pm/embed/times_embed.py` | §4.1 公式(2) | 复用(代码同旧版) | 股票级嵌入：把原始特征变成向量 |
| 2 | `mae_注释版.py` | `pm/net/mae.py` | §4.1 公式(3)(4)(8) | 复用 | 掩码自编码器骨架(掩码/编码/解码/重构) |
| 3 | `mask_time_state_注释版.py` | `pm/net/mask_time_state.py` | §4.1 | 复用 | 表征网络：产出"可掩码股票表征 ρ" |
| 4 | `mask_sac_net_注释版.py` | `pm/net/sac/mask_sac_net.py` | §4.2 公式(6) | 复用 | Actor(策略) + Critic(价值) 网络 |
| 5 | `helpers_注释版.py` | `pm/utils/helpers.py` | §4.3 公式(9) | 复用 | 重加权：温度 softmax + 掩码忽略 |
| 6 | `portfolio_management_dataset_注释版.py` | `pm/dataset/portfolio_management_dataset.py` | §3.2 | **新增** | 数据集：读股票/特征/CSP子池mask |
| 7 | `pm_based_portfolio_ASR_新版_注释版.py` | `pm/environment/pm_based_portfolio_ASR.py` | §3.2 | **★新增(新版重写)** | RL 环境：★A股真实交易模拟器 |
| 8 | `mask_sac_注释版.py` | `pm/agent/sac/mask_sac.py` | 算法1, 公式(5)(7)(8) | 复用* | ★智能体：把一切串成完整 SAC 训练 |
| 9 | `train_新版_注释版.py` | `tools/train.py` | 算法1, 算法2 | **新增** | 训练主循环：四大件 + episode 循环 |

> *`mask_sac.py`：新版相对旧版**仅多 4 行**无关紧要的代码（一行注释、一行 debug print、一行重复的 `os.makedirs(logs_dir, exist_ok=True)`），都在导出投资组合数据的辅助逻辑里，**不影响任何训练算法逻辑**。因此既有注释完全适用，本目录直接复用。

## 三、建议阅读顺序

**按上表 1→9 顺序读**，正好是数据在系统里的流动方向：

```
数据(6) → 股票级嵌入(1) → MAE掩码重构表征(2,3) → Actor/Critic决策(4) → 重加权(5)
        → A股交易环境(7) → 智能体把一切串起来训练(8) → 训练主循环(9)
```

- **想快速抓住论文核心**：先看 2(mae) + 3(mask_time_state) 理解"可掩码表征"，再看 8(mask_sac) 的 `update_net` 五件套。
- **想理解新版改了什么**：重点看 7(★A股环境)，并对照 `03_EarnMore代码工程解析_新版.md` 第 7 节。
- **想跑通/改实验**：看 9(train) 理解主循环，再回 `03` 第 4.2 节看配置如何驱动一切。

## 四、学懂的标志

1. **算法层**：能对着 `mask_sac_注释版.py` 的 `update_net()` 逐行说出——这步更新哪个网络(Critic/Alpha/Actor/Beta/Rep)、对应论文哪个公式(5/7/6/§4.2/8)、为什么这么做。
2. **工程层**：能讲清 `pm_based_portfolio_ASR_新版_注释版.py` 一次 `step` 里——t+N 调仓判断 / T+1 / 涨跌停 / 整手 / 费用 / 滑点 是怎么依次作用的。
3. **串联**：能从 `train_新版_注释版.py` 的主循环，一路追到 `mask_sac` 的 `update_net` 和 `mask_time_state` 的 `forward_state`。

做到这三点，新版 EarnMore 的代码就真正掌握了。
