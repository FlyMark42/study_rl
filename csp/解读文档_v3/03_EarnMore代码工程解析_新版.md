# EarnMore 代码工程解析 v3（针对新版 EarnMore-main_v20260610）

> **写给谁**：想把论文和代码对应起来、亲手跑通并改进 EarnMore 的初学者。
>
> **本版定位**：在 `解读文档_v2/03` 基础上，**针对新版代码 `EarnMore-main_v20260610` 更新**。配套阅读：`EarnMore_论文中英对照.md`（理解算法，中英对照）、`02_强化学习知识点学习清单.md`（补 RL 理论）、`带详细注释的核心代码/`（逐行注释，重点是新版环境）。
>
> **一句话结论**：新版相对旧版，**算法主干（SAC + MAE 表征 + 重加权）几乎没动**，**唯一的重大改动是把交易环境重写成贴近中国 A 股的真实交易模拟器**。学习时先用 v2 的认知把算法过一遍，再把精力放在"新环境"和"工程重组"上。

---

## 目录
1. [新旧版差异总览（最重要）](#1-新旧版差异总览最重要)
2. [整体架构鸟瞰](#2-整体架构鸟瞰)
3. [目录结构（新版）](#3-目录结构新版)
4. [核心设计模式：注册器 + 配置驱动](#4-核心设计模式注册器--配置驱动)
5. [训练主循环与数据流](#5-训练主循环与数据流)
6. [算法主干：MAE 表征 + SAC + 重加权（新旧一致）](#6-算法主干mae-表征--sac--重加权新旧一致)
7. [★新版重点：A 股真实交易环境 EnvironmentASR](#7-新版重点a-股真实交易环境-environmentasr)
8. [代码与论文的对应关系](#8-代码与论文的对应关系)
9. [新版相对论文/旧版的所有改动清单](#9-新版相对论文旧版的所有改动清单)
10. [工程外围与如何运行](#10-工程外围与如何运行)
11. [推荐阅读顺序](#11-推荐阅读顺序)

---

## 1. 新旧版差异总览（最重要）

我们逐文件比对了 `EarnMore-main`（旧）与 `EarnMore-main_v20260610`（新）。**结论：算法核心未变，改动集中在环境、配置和工程组织。**

| 文件 | 改动量 | 性质 |
|---|---|---|
| `pm/environment/pm_based_portfolio_ASR.py` | ★**213 → 620 行（重写）** | **核心逻辑改动**：理想化环境 → A 股真实交易模拟器 |
| `configs/mask_sac_portfolio_management.py` | 中等（约 31 行） | 数据期、股票数、本金等参数调整 |
| `tools/train.py` | 小（约 19 行） | 去掉发邮件包装、根路径解析、`MonitoredTrainer` 改从 `experiments/` 导入 |
| `pm/agent/sac/mask_sac.py` | 极小（4 行） | 仅实验目录 glob 与建日志目录的健壮性 |
| `pm/net/mae.py`、`mask_time_state.py`、`mask_sac_net.py`、`utils/helpers.py`、`embed/times_embed.py`、`dataset/...` | **0 行（完全一致）** | 算法网络/表征/重加权/数据集**未改动** |
| 工程工具 | 重组 | 旧版散在根目录（`hyperopt.py` 等）→ 新版收进 `experiments/`，并新增 `alpha_monitor/alpha_visualizer/best_model_finder/monitored_trainer` |

> **学习启示**：`mask_sac.py` 的 `update_net()` 五件套、MAE 掩码-重构、Actor/Critic、重加权公式——**新旧完全一致**，`解读文档_v2/03` 与旧版逐行注释直接通用。新版要新学的，只有"**环境**"这一块。

---

## 2. 整体架构鸟瞰

记住那句话：**EarnMore = SAC（强化学习）+ MAE（表征学习）+ 投资组合管理（任务）**。

```
            配置文件 (configs/*.py)  —— 定义所有超参 + 各组件"配方(dict)"
                        │ Config.fromfile()
                        ▼
   tools/train.py (训练总指挥)
     1) DATASET.build()    → PortfolioManagementDataset (读 CSV，组织成 股票×天×特征 + 子池mask)
     2) ENVIRONMENT.build()→ EnvironmentASR  ★新版=A股真实交易模拟器
                            └ gym.vector.SyncVectorEnv 包成"向量环境"(并行多子池)
     3) AGENT.build()      → AgentMaskSAC (内含 3 网络 + 优化器 + 训练逻辑)
            ├ rep_net: MaskTimeState  (MAE 表征网络  ←论文模块 a+b)
            ├ act_net: ActorMaskSAC   (演员/策略网络  ←论文模块 c)
            └ cri_net: CriticMaskSAC  (评论家/价值网络 ←论文模块 c)
     4) ReplayBuffer
     5) for episode: explore_env() → buffer.update() → update_net()(五件套) → validate()
```

**一句话**：配置文件像"配方表"，`train.py` 照配方造出数据集/环境/智能体，再反复"交互采数据 → 存缓冲区 → 抽样更新网络 → 验证"，直到收敛。新版把第 2 步的"环境"换成了真实交易模拟器，其余不变。

---

## 3. 目录结构（新版）

```text
EarnMore-main_v20260610/
├── configs/                  实验配置(每个算法一个)
│   └── mask_sac_portfolio_management.py   ← EarnMore 主配置
├── datasets/                 股票池文件(txt) + 部分数据
├── pm/                       ★★★ 核心算法包，学习重点
│   ├── registry.py           【枢纽】注册器：字符串名字 → 类
│   ├── dataset/portfolio_management_dataset.py   读CSV、组织股票数据、生成子池 mask
│   ├── environment/          ★ RL环境(gym)
│   │   ├── pm_based_portfolio_ASR.py     ← ★新版重写：A股真实交易模拟器(奖励=夏普)
│   │   ├── pm_based_portfolio_value.py   ← 原版(奖励=市值变化，贴近论文)
│   │   └── pm_based_portfolio_return.py
│   ├── embed/times_embed.py  ← 论文模块(a)股票级嵌入
│   ├── net/
│   │   ├── mae.py            ← 掩码自编码器基类(模块 a+b 骨架)
│   │   ├── mask_time_state.py← 可掩码表征网络(§4.1)
│   │   └── sac/mask_sac_net.py← ActorMaskSAC + CriticMaskSAC (§4.2)
│   ├── agent/sac/
│   │   ├── mask_sac.py       ← ★AgentMaskSAC，本论文主角
│   │   └── mask_sync_sac.py  ← 同步版掩码 SAC
│   ├── utils/{replay_buffer.py, helpers.py(重加权), misc.py, plot.py}
│   ├── criterion/ optimizer/ scheduler/ metrics/
├── tools/
│   ├── train.py              ← 训练主循环(学习重点)
│   ├── test.py               ← 测试/回测
│   ├── tools_utils.py        ← 工具函数(旧版叫 utils.py)
│   └── preprocess*.py / make_*.py
│   ─────── 新版：工程外围统一收进 experiments/(了解即可) ───────
└── experiments/
    ├── experiment_manager.py / hyperopt.py / scheduler.py / monitor.py / visualizer.py
    └── alpha_monitor.py / alpha_visualizer.py / best_model_finder.py / monitored_trainer.py  ← 新增
```

> **与旧版的目录差异**：① 工程工具从根目录搬进 `experiments/`；② 新增 4 个工具（alpha 监控/可视化、最优模型查找、受监控训练器）；③ 去掉了旧版若干 `_v2025xxxx` 版本化文件与 `train_arr.py/test_arr.py`；④ `tools/utils.py` 改名 `tools/tools_utils.py`。
>
> **学习优先级**：`net/mae.py` → `net/mask_time_state.py` → `net/sac/mask_sac_net.py` → `agent/sac/mask_sac.py` → **`environment/pm_based_portfolio_ASR.py`（新版重点）** → `tools/train.py`。

---

## 4. 核心设计模式：注册器 + 配置驱动

（与旧版一致，新版未改。初学者读这份代码**最大的拦路虎**，先讲清它。）

### 4.1 注册器（Registry）🔧
`pm/registry.py` 把 `DATASET / NET / AGENT / ENVIRONMENT / OPTIMIZER / SCHEDULER / CRITERION / EMBED` 都做成"电话簿"。类定义上方写 `@NET.register_module()` 登记；之后 `NET.build(dict(type="ActorMaskSAC", embed_dim=64))` 自动查簿造类。这套 `Registry + Config` 是商汤 mmengine（OpenMMLab）标准玩法。

### 4.2 配置驱动（Config-driven）🔧
项目"长什么样"全写在 `configs/mask_sac_portfolio_management.py`（嵌套 `dict`）。`train.py` 用 `Config.fromfile()` 读入、`.build()` 造出来。**记住"`type` 字段的值 = 真正的类名，去对应目录搜它"。**

> ✅ 配置顶部用 `check_gpu_available()` 自动切分支——GPU 用完整 `AgentMaskSAC`（含掩码表征），CPU 退化为普通 `AgentSAC` 并关掉 `if_use_rep`/`if_use_beta`。**学习以 GPU 分支为准**（那才是 EarnMore）。

**新版配置关键变化**（vs 旧版）：

| 配置项 | 旧版 | 新版 | 含义 |
|---|---|---|---|
| `num_stocks`（GSP） | 790 | **1000** | 全局股票池规模 |
| `num_stocks`（test 段） | 790 | **5** | 测试子池规模 |
| 训练期 | 2023-01 ~ 2025-03 | **2021-01 ~ 2024-06** | 训练数据拉长 |
| 验证期 | 2025-04 ~ 2025-09 | 2024-07 ~ 2024-12 | |
| 测试期 | 2025-09 | 2025-01 ~ 2025-03 | |
| `num_episodes` | 200 | **5** | 新版当前是"快速冒烟"配置，跑通用；正式训练要调大 |
| `data_path` | `csp_817_data_cut` | `csp_data_cut` | 换了数据目录 |
| `initial_amount`（env） | 1e3 | **1e5** | 初始本金 10 万 |

> ⚠️ **重要**：新版配置的 `environment` 段**只传了 `type/dataset/mode/if_norm/days/initial_amount/transaction_cost_pct` 等少数参数，并没有传新增的 A 股交易参数**（`rebalance_period`、`enable_t1_rule`、`enable_price_limit`、`lot_size`、各项费率等）。因此这些交易规则**全部走 `EnvironmentASR.__init__` 的默认值**：每 5 天调仓、T+1 开、涨跌停开、整手 100、滑点 0.1%、留 1% 现金、佣金万 0.855(最低5元)、印花税万5、沪市过户费万 0.2。**想改交易规则，要么在 config 的 environment dict 里显式加这些参数，要么改 `__init__` 默认值。**

---

## 5. 训练主循环与数据流

### 5.1 一条数据的"形状变化"之旅 🎯
记号：`B`=批量, `N`=股票数, `D`=天数, `F`=特征数, `C`=嵌入维度。
```
① 环境给原始状态                state: (B, N, D, F)
      │ rep.forward_state() ← MaskTimeState
② 股票级嵌入(TimesEmbed: 卷积+时序+位置, 沿天数求均值)  (B, N, C)
      │ 随机掩码→只留未掩码送编码器→Transformer编码
③ 池级潜在嵌入                    (B, N_keep, C)
      │ 用 mask_token 填回 N 个 → 按 ids_restore 还原顺序
④ 可掩码股票表征 ρ(论文核心状态)  (B, N, C)
      │ act.get_action(ρ) ← ActorMaskSAC
⑤ 每只股票/现金 logits → 重加权(温度softmax)
⑥ 动作=组合权重                  (B, N+1)  加起来=1, 第0位现金
      │ env.step(action)   ← ★新版在这里执行 A 股真实交易
⑦ 环境算收益、扣费、更新持仓&现金, 算夏普奖励, 进入下一步
```

### 5.2 训练主循环（`tools/train.py: main()`）🎯
```python
dataset = DATASET.build(cfg.dataset)
train_envs = gym.vector.SyncVectorEnv([...])   # 向量环境(并行多子池)
val_envs   = gym.vector.SyncVectorEnv([...])   # 每个 aux 子池一个验证环境
agent  = AGENT.build(cfg.agent)
buffer = ReplayBuffer(...)
buffer.update(agent.explore_env(train_envs, horizon_len))  # 先采一批填充

for episode in range(start, num_episodes+1):
    train_one_episode(train_envs, buffer, agent, horizon_len)  # 采数据→更新网络
    val_stats = validate(val_envs, agent)                      # 算 ARR/SR/MDD
    if metric > max_metrics:
        save_checkpoint(episode, agent, if_best=True)          # 验证创新高则存最优
```
> **新版 train.py 小改**：① 去掉了旧版末尾 `send_email(...)` 的 try/except 包装，`main(args)` 直接调用更干净；② 增加根路径解析（相对路径转绝对）；③ `MonitoredTrainer` 从 `experiments.monitored_trainer` 导入（配合目录重组）。**主循环逻辑本身没变**，仍对应论文算法 1。

---

## 6. 算法主干：MAE 表征 + SAC + 重加权（新旧一致）

这部分新旧版**完全相同**，逐行注释见本目录 `带详细注释的核心代码/`（`times_embed/mae/mask_time_state/mask_sac_net/helpers/mask_sac_注释版.py`），更概览的拆解见 `解读文档_v2/03` 第 5 节。这里只给速记：

- **`mae.py` / `mask_time_state.py`（论文 §4.1）**：`random_masking` 随机遮股票 → `forward_encoder` 编码未掩码 → `mask_token` 填回 + `ids_restore` 还原 → `forward_decoder` 重构 → `forward_loss` 只在被掩码股票上算 MSE（公式 8）。`forward_state()` 是给 RL 用的接口（只出表征 ρ、不算损失）。截断高斯掩码率（公式 3）来自 `mask_ratio_generator`。
- **`mask_sac_net.py`（论文 §4.2）**：`ActorMaskSAC` 用 `rsample()` 重参数化采样（公式 6）；`CriticMaskSAC` 双 Q、`get_q_min` 取小抑制高估（公式 5）。
- **`mask_sac.py: update_net()`（论文算法 1）** 五件套，每 batch 顺序更新：① Critic（TD 误差，公式 5）→ ② Alpha（自动熵，公式 7）→ ③ Actor（公式 6）→ ④ Beta（掩码监督惩罚，§4.2）→ ⑤ Rep（重构损失，公式 8）。其中 `q_labels = rewards + (1-dones)*gamma*(next_qs - next_logprobs*alpha)` 就是贝尔曼目标的代码实现。
- **`helpers.py`（论文 §4.3，公式 9）**：`get_action_wrapper` 用温度 softmax `F.softmax(pred/T)` 做重加权；池外股票 `pred - mask_bool*1e6` 后 softmax≈0。

---

## 7. ★新版重点：A 股真实交易环境 EnvironmentASR

**这是新版唯一的重大逻辑改动**，文件 `pm/environment/pm_based_portfolio_ASR.py`（213→620 行）。逐行注释见 `带详细注释的核心代码/pm_based_portfolio_ASR_新版_注释版.py`。

### 7.1 它解决什么问题
旧版环境是**理想化、无摩擦**的：直接"权重 × 当日涨跌幅 = 组合收益"，每天都假定能瞬间、无成本地调到目标权重。这在真实 A 股里不成立。新版把作者另一套交易系统 `AGDRL_calmar_v2` 的**真实 A 股交易执行逻辑**移植进来，变成一个**高保真交易模拟器**。

### 7.2 七大新增机制

| 机制 | 参数(默认) | 作用 |
|---|---|---|
| **t+N 调仓** | `rebalance_period=5` | 每 5 个交易日才真正换仓，其余日子只"持有"。`step` 里判断 `day - last_rebalance_day >= 5` |
| **T+1 规则** | `enable_t1_rule=True` | 当天买入的股票当天不能卖（记入 `shares_frozen`，次日开盘解冻） |
| **涨跌停限制** | `enable_price_limit=True` | 按代码推断幅度：ST±5%、科创/创业(688/300/301)±20%、北交所(8x/4x)±30%、主板±10%。封涨停禁买、封跌停禁卖 |
| **整手约束** | `enable_lot_size=True, lot_size=100` | 买卖股数向下取整到 100 股的整数倍 |
| **完整交易成本** | 见下 | 佣金(万0.855,最低5元,买卖都收) + 印花税(万5,仅卖) + 过户费(万0.2,仅沪市) |
| **滑点** | `slip_perc=0.001` | 买入成交价 ×(1+0.1%)、卖出 ×(1−0.1%)，并限制在当日[低,高]价内 |
| **现金留存** | `cash_reserve_ratio=0.01` | 永远保留 1% 现金缓冲，避免满仓穿仓 |

### 7.3 调仓日的处理流水线（`_step_rebalance`）
```
动作 a (含现金权重)
  → _normalize_action_weights:  截负、归一化(和=1)
  → _weights_to_shares:         权重 × (资本×99%) ÷ 开盘价 = 目标股数
  → _apply_t1_constraint:       不允许卖出被 T+1 冻结的股
  → _apply_price_limit_constraint: 封涨停不能加仓、封跌停不能减仓
  → _apply_lot_size_constraint: 向下取整到整手
  → _apply_cash_constraint:     预估卖出回款&买入花费(含费用滑点), 不够则按比例缩减买入
  → _execute_trades:            逐股成交, 算佣金/印花税/过户费/滑点, 更新 cash 和 shares_held
  → 冻结今日买入(shares_frozen), 算当日收益(今开→次开, 已含成本), 算夏普奖励
```
非调仓日走 `_step_hold`：只按持仓市值变化算收益、不交易。

### 7.4 状态/动作/奖励（与论文 MDP 的关系）
- **状态**：仍是 `[N股票, days天, F特征]` 的窗口（论文定义不变）。
- **动作**：仍是 `[现金, 各股票权重]`（论文定义不变）；区别在于新版会把这个"理想权重"经过一整套 A 股约束**打折**成"实际成交"。
- **奖励**：`_compute_asr_reward()` = 年化夏普比率（`qs_stats.sharpe(rets, rf=0.02)`，裁剪为非负）。⚠️ **论文原版奖励是市值变化 $V_t-V_{t-1}$**（见 `pm_based_portfolio_value.py`，代码确为 `new_portfolio_value - old_portfolio_value`），这套魔改版改用夏普。

### 7.5 学习提示
- 新版环境引入大量摩擦，**训练更慢、回测收益更"真实"（通常更低）**，但更接近实盘可执行性。
- **状态里没有暴露"持仓/现金/冻结"**——智能体看到的状态仍只是市场特征窗口，它并不"知道"自己现在持有多少、哪些被冻结。这是一个值得注意的设计点（实盘里这些通常会作为状态的一部分）。
- 想做"纯算法学习"或复现论文，可在 config 里把 `environment.type` 换成 `pm_based_portfolio_value`（贴近论文的简单环境），或显式关掉新机制（传 `enable_t1_rule=False` 等）。

---

## 8. 代码与论文的对应关系

| 论文概念 | 公式/章节 | 代码位置（新版，路径同旧版） |
|---|---|---|
| 股票级嵌入 $l_s$ | §4.1 公式(2) | `embed/times_embed.py: TimesEmbed.forward()` |
| 随机掩码率 $g(r)$ | 公式(3) | `net/mae.py: mask_ratio_generator (truncnorm)` |
| 掩码操作 $\eta_{mo}$ | 公式(4) | `net/mae.py: random_masking()` |
| 编码/填充/解码/表征 $\rho$ | 公式(4) | `net/mask_time_state.py: forward_encoder/forward_state/forward_decoder` |
| 重构损失(MSE) | 公式(8) | `net/mask_time_state.py: forward_loss()` |
| Q 网络优化(TD误差) | 公式(5) | `agent/sac/mask_sac.py: get_obj_critic_raw()` 的 `q_labels` |
| 目标网络软更新 | 公式(5) 的 $\bar\theta$ | `agent/sac/mask_sac.py: soft_update(), cri_target` |
| 策略优化(重参数化) | 公式(6) | `net/sac/mask_sac_net.py: get_action_logprob() + rsample()` |
| Alpha 自动调节 | 公式(7) | `agent/sac/mask_sac.py: obj_alpha, alpha_log` |
| 掩码监督惩罚(Beta) | §4.2 | `agent/sac/mask_sac.py: get_object_beta()` |
| 重加权 $Re(x)$ | §4.3 公式(9) | `utils/helpers.py: get_action_wrapper()` |
| 训练/推理算法 | 算法1/2 | `tools/train.py` + `agent.update_net()` / `validate_net()` |
| MDP 状态/动作/奖励 | §3.2 | **`environment/pm_based_portfolio_ASR.py`（★新版重写）** |
| 6个金融指标 | 附录A | `metrics/` 下 ARR/SR/CR/SOR/MDD/VOL |

---

## 9. 新版相对论文/旧版的所有改动清单 ✅

学习时务必注意，这份代码与论文、与旧版的出入（不是 bug，是工程落地/二次开发）：

1. **奖励函数**（与论文不同）✅：`EnvironmentASR` 奖励 = **夏普比率**（裁剪为非负）；论文原版 = **市值变化 $V_t-V_{t-1}$**（`pm_based_portfolio_value.py`）。
2. **交易环境**（新版 vs 旧版的核心改动）✅：旧版无摩擦理想环境 → 新版 **A 股真实交易模拟器**（t+N 调仓、T+1、涨跌停、整手、完整费用、滑点、现金留存、股数级账本）。
3. **数据/参数**：论文用美股 SP500/DJ30(2007~2022)；新版用 A 股自处理数据(`csp_data_cut`)、GSP 1000 只、期 2021~2025、本金 10 万、当前 `num_episodes=5`(冒烟配置)。
4. **特征维度**：论文 F=102；本仓库 `num_features` 取自定义值（旧版注释里见过 67）。
5. **GPU/CPU 双分支** ✅：GPU 用 `AgentMaskSAC`(完整掩码版)，CPU 退化普通 `AgentSAC`(关 rep/beta)。**学习以 GPU 分支为准**。
6. **Actor 排序加权** ✅：`soft_logits = logits * log(indices+1)`，论文没有（此项在未改动的 `mask_sac_net.py` 中）。
7. **工程外围重组**：旧版根目录工具 → 新版 `experiments/`，并新增 alpha 监控/可视化、最优模型查找、受监控训练器；`train.py` 去掉发邮件包装。
8. **交易参数未进配置**：A 股交易规则当前走 `__init__` 默认值，config 未显式暴露（见第 4.2 节末尾）。

---

## 10. 工程外围与如何运行

**工程外围（`experiments/`，了解即可）**：`experiment_manager.py`(实验管理)、`hyperopt.py`(超参搜索)、`scheduler.py`(并发调度)、`monitor.py`/`alpha_monitor.py`(训练/alpha 监控)、`visualizer.py`/`alpha_visualizer.py`(可视化)、`best_model_finder.py`(找最优模型)、`monitored_trainer.py`(受监控训练器)。`quickhelp.txt` 有作者整理的命令流程。

**如何跑**（研究代码，依赖较重：PyTorch + timm + einops + mmengine + qlib + gym + quantstats 等）：
```bash
pip install -r requirements.txt        # qlib 需单独装
# 准备数据到 config 的 data_path 指定目录: features/*.csv, stocks_cut.txt, aux_stocks_files_cleaned/*.txt
python tools/train.py --config configs/mask_sac_portfolio_management.py --root .
# 或用 pipeline： sh tools/pipeline_mask_sac_dj30_example.sh
```
注意：数据路径写死在 config；CPU 上会自动降规模、切普通 SAC；新版 `train.py` 已去掉发邮件，无需改。**建议先读懂核心文件、再跑通**（环境配置往往最耗时）。

---

## 11. 推荐阅读顺序

1. **建立全局观**：本文第 1~5 节 + `02_学习清单`，搞清"三大件 + 主循环 + 新旧差异"。
2. **过算法主干**（新旧一致）：`times_embed → mae → mask_time_state → mask_sac_net → helpers → mask_sac`（逐行注释在本目录 `带详细注释的核心代码/`），每读一个回对 `EarnMore_论文中英对照.md` 的公式。
3. **攻新版重点**：读 `带详细注释的核心代码/pm_based_portfolio_ASR_新版_注释版.py` + 本文第 7 节，理解 A 股交易模拟器。
4. **看主循环**：`tools/train.py`，理解一个 episode 怎么跑。
5. **动手改**：在 config 的 `environment` dict 里显式加 `rebalance_period`、`enable_price_limit=False` 等，观察对回测的影响；或把 `environment.type` 换成 `pm_based_portfolio_value` 对比"理想 vs 真实"环境的差异。

> **学懂标志**：① 能对着 `update_net()` 逐行说出"更新哪个网络、对应论文哪个公式、为什么"（算法层）；② 能讲清新版环境一次 `step` 里 t+N 调仓 / T+1 / 涨跌停 / 整手 / 费用 / 滑点 是怎么依次作用的（工程层）。
