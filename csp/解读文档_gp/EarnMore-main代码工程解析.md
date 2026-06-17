# EarnMore-main 代码工程解析

> 面向零基础读者。本文件解释 `csp/EarnMore-main` 的工程结构、核心功能、主要逻辑，以及论文 EarnMore 方法如何落到代码里。

## 1. 工程整体作用

`EarnMore-main` 是论文 **Reinforcement Learning with Maskable Stock Representation for Portfolio Management in Customizable Stock Pools** 的代码工程。

它做的事情可以概括为：

1. 读取股票行情、技术指标和自定义股票池；
2. 构建投资组合管理环境；
3. 用强化学习智能体与环境交互；
4. 通过 ReplayBuffer 存储经验；
5. 使用 SAC 或 MaskSAC 更新 Actor、Critic 和表示网络；
6. 在验证集上回测，并计算 ARR、SR、MDD 等金融指标。

## 2. 重要目录

```text
csp/EarnMore-main
├── configs/                  # 实验配置文件
├── datasets/                 # 股票池文件和部分数据
├── pm/
│   ├── agent/                # 强化学习智能体：DQN/DDPG/TD3/PPO/SAC/MaskSAC
│   ├── criterion/            # 损失函数注册
│   ├── dataset/              # 数据集读取
│   ├── embed/                # 股票时间序列 embedding
│   ├── environment/          # 投资组合管理 Gym 环境
│   ├── metrics/              # 金融评估指标
│   ├── net/                  # Actor/Critic/MAE/MaskTimeState 等网络
│   ├── optimizer/            # 优化器注册
│   ├── scheduler/            # 学习率调度器
│   └── utils/                # ReplayBuffer、动作包装、保存加载等工具
├── tools/
│   ├── train.py              # 训练入口
│   ├── test.py               # 测试入口
│   └── preprocess*.py        # 数据预处理脚本
├── experiment_manager.py     # 实验管理
├── hyperopt.py               # 超参数搜索
├── monitor.py                # 训练监控
└── visualizer.py             # 可视化
```

## 3. 最推荐的阅读顺序

1. `configs/mask_sac_portfolio_management.py`：看一个实验由哪些参数组成。
2. `tools/train.py`：看训练主流程。
3. `pm/dataset/portfolio_management_dataset.py`：看数据如何读入。
4. `pm/environment/pm_based_portfolio_ASR.py`：看状态、动作、奖励如何定义。
5. `pm/agent/sac/mask_sac.py`：看 MaskSAC 如何训练。
6. `pm/net/mask_time_state.py`：看可掩码股票表示如何实现。
7. `pm/net/sac/mask_sac_net.py`：看 Actor/Critic 网络。
8. `pm/utils/replay_buffer.py` 和 `pm/utils/helpers.py`：看经验回放和重加权。

## 4. 配置文件：实验的说明书

核心配置：`configs/mask_sac_portfolio_management.py`

配置里包含：

- 股票数量：`num_stocks`
- 历史窗口天数：`days`
- 特征数量：`num_features`
- 训练/验证/测试日期区间
- ReplayBuffer 大小：`buffer_size`
- batch size：`batch_size`
- horizon_len：每次和环境交互多少步
- 网络维度：`embed_dim`、`decoder_embed_dim`
- mask 比例：`mask_ratio_min/max/mu/std`
- 强化学习超参数：`gamma`、`repeat_times`、`soft_update_tau`
- 重加权温度：`T`

论文对应关系：

- `rep_net = MaskTimeState` 对应可掩码股票表示。
- `act_net = ActorMaskSAC` 对应 Actor。
- `cri_net = CriticMaskSAC` 对应 Critic。
- `action_wrapper_method = "reweight"` 对应论文 4.3 节重加权。

注意：当前配置中 GPU 和 CPU 会走不同分支。GPU 时使用 `AgentMaskSAC`，CPU 时会退化为普通 `AgentSAC`，以降低计算量。

## 5. 训练主流程

入口：`tools/train.py`

主流程如下：

```text
读取配置
  -> 更新 root/workdir/tag
  -> 固定随机种子
  -> 构建 Dataset
  -> 构建 train/val 环境
  -> 构建 Agent
  -> 初始化 ReplayBuffer
  -> 先探索一小段填充 buffer
  -> for episode:
       训练一个 episode
       验证一个 episode
       写 TensorBoard 和日志
       如果验证指标更好，保存 best.pth
```

关键函数：

- `main(args)`：总入口。
- `train_one_episode(environment, buffer, agent, horizon_len)`：训练一个 episode。
- `validate(environment, agent)`：验证并整理指标。

训练中真正发生学习的是：

```text
agent.explore_env(...)
buffer.update(...)
agent.update_net(buffer)
```

## 6. 数据集读取

文件：`pm/dataset/portfolio_management_dataset.py`

这个类负责把三类数据组织起来：

1. `stocks_path`：全局股票池 GSP 的股票列表；
2. `data_path`：每只股票一个 csv；
3. `aux_stocks_path`：自定义股票池 CSP 的 txt 文件。

核心字段：

- `self.stocks`：全局股票池中的股票代码。
- `self.stocks2id` / `self.id2stocks`：股票代码和索引互转。
- `self.aux_stocks`：自定义股票池。
- `self.stocks_df`：每只股票的 DataFrame。

CSP mask 的含义：

- `0.0`：这只股票在目标 CSP 内，可以交易。
- `1.0`：这只股票不在目标 CSP 内，应被 mask。

特别注意：

`aux_stocks[0]` 被设成 `All`，表示完整 GSP，它的 mask 全是 0。

## 7. 环境：状态、动作、奖励

主要文件：`pm/environment/pm_based_portfolio_ASR.py`

### 7.1 状态

环境返回的状态是：

```text
[num_stocks, days, num_features]
```

含义是：所有股票最近 `days` 天的特征窗口。

例如配置中：

- `num_stocks = 790`
- `days = 10`
- `num_features = 67`

那么单个环境状态近似是：

```text
[790, 10, 67]
```

### 7.2 动作

动作是组合权重：

```text
[cash, stock_1, stock_2, ..., stock_N]
```

第 0 维是现金权重，后面是每只股票的资金占比。

### 7.3 奖励

论文原始定义：

```text
reward = V_t - V_{t-1}
```

当前 `EnvironmentASR` 代码实现：

```text
reward = 当前历史收益序列的年化 Sharpe Ratio
若 reward 为 NaN/Inf 或小于 0，则置为 0
```

这意味着当前代码更偏向“风险调整后收益”，而不是单纯收益差。

### 7.4 step 过程

```text
读取当前收盘价
  -> day += 1
  -> 读取下一天收盘价
  -> 根据股票权重计算 portfolio_ret
  -> 更新 portfolio_value
  -> 用历史收益计算 Sharpe reward
  -> 返回 next_state, reward, terminated, truncated, info
```

## 8. MaskTimeState：论文核心表示网络

文件：`pm/net/mask_time_state.py`

它对应论文中的 **Maskable Stock Representation**。

输入可以是：

```text
[batch, stock, days, features]
```

也可以是：

```text
[batch, channel, stock, days, features]
```

核心流程：

```text
原始股票时间序列
  -> TimesEmbed 得到每只股票的 stock-level embedding
  -> 随机 mask 一部分股票
  -> encoder 编码未 mask 股票
  -> 用 mask_token 补回被 mask 股票位置
  -> decoder 重建被 mask 股票
  -> 只在被 mask 股票上计算 MSE
```

重要函数：

- `forward_encoder`：生成股票 embedding 并执行 mask。
- `forward_state`：补回 mask token，生成给 Actor/Critic 使用的状态。
- `forward_loss`：只在 mask 位置计算重建损失。
- `forward`：训练表示网络完整流程。

为什么要这样做：

- 不同 CSP 股票数量不同，但 Actor/Critic 需要固定维度输入。
- masked token 告诉模型“这些股票被投资者排除”。
- 重建任务迫使模型学习股票之间关系。

## 9. TimesEmbed：股票时间序列嵌入

文件：`pm/embed/times_embed.py`

TimesEmbed 做三件事：

1. `TokenEmbedding`：用 1D 卷积编码价格和技术指标。
2. `TemporalEmbedding`：编码 weekday、day、month 等日期信息。
3. `PositionalEmbedding`：加入时间位置编码。

然后把它们相加，并对历史天数维度求平均，得到每只股票一个向量。

简单说：它把“某只股票最近 10 天的一堆特征”压缩成一个 embedding。

## 10. ActorMaskSAC 与 CriticMaskSAC

文件：`pm/net/sac/mask_sac_net.py`

### 10.1 ActorMaskSAC

Actor 的输入：

```text
MaskTimeState.forward_state 得到的股票池表示
```

Actor 的输出：

```text
[cash_weight, stock_1_weight, ..., stock_N_weight]
```

如果启用 `cls_embed`，最前面会加一个 `[CLS]` token，用来对应现金仓位。

训练时：

- `get_action` 从高斯分布采样动作；
- `get_action_logprob` 返回动作和 log probability；
- log probability 用于 SAC 熵正则。

### 10.2 CriticMaskSAC

Critic 的输入：

```text
状态表示 + 动作权重
```

Critic 输出两个 Q 值：

```text
Q1(s, a), Q2(s, a)
```

SAC 取 `min(Q1, Q2)` 作为目标的一部分，用来减少 Q 值过估计。

## 11. AgentMaskSAC：核心强化学习智能体

文件：`pm/agent/sac/mask_sac.py`

它把表示网络、Actor、Critic、优化器、调度器和 replay buffer 串起来。

### 11.1 初始化

核心成员：

- `self.rep`：MaskTimeState 表示网络。
- `self.act`：ActorMaskSAC。
- `self.cri`：CriticMaskSAC。
- `self.cri_target`：目标 Critic。
- `self.alpha_log`：SAC 熵温度参数。
- `self.get_action`：包装后的动作采样函数，会应用重加权。

### 11.2 explore_env

逻辑：

```text
for t in horizon_len:
    state -> rep.forward_state -> rep_state
    rep_state -> actor -> action
    env.step(action)
    保存 state/action/mask/ids_restore/reward/done/next_state
```

这里保存 `mask` 和 `ids_restore` 很关键，因为训练时要复现当时的 mask 状态。

### 11.3 update_net

每次更新顺序：

```text
1. 更新 Critic
2. 软更新 target Critic
3. 更新 alpha
4. 更新 Actor
5. 如果启用 beta，惩罚投到 mask 股票上的权重
6. 如果启用 rep，更新表示网络重建损失
```

### 11.4 Critic 损失

核心公式在代码中是：

```python
q_labels = rewards + (1.0 - dones) * gamma * (next_qs - next_logprobs * alpha)
```

小白理解：

```text
当前动作应该得到的分数
= 当前奖励
+ 未来价值
+ 探索带来的价值
```

然后让 Q1 和 Q2 都去拟合这个目标。

### 11.5 beta loss

```python
beta_loss = (weight * mask_bool).sum(dim=1).mean()
```

含义：

如果某只股票被投资者 mask 掉，而 Actor 仍给它分配权重，就产生损失。这样模型会学着不投这些股票。

## 12. ReplayBuffer

文件：`pm/utils/replay_buffer.py`

作用：存储智能体和环境交互得到的 transition。

普通 transition 包括：

```text
state, action, mask, ids_restore, reward, done, next_state
```

为什么要 ReplayBuffer：

- 强化学习数据前后相关性很强；
- 随机抽样可以打散相关性；
- 旧经验可以重复利用，提高样本效率。

代码也支持 PER（Prioritized Experience Replay），也就是优先采样 TD error 更大的经验。

## 13. 重加权工具

文件：`pm/utils/helpers.py`

重加权公式：

```text
softmax(logits / T)
```

`T` 是温度：

- `T < 1`：权重更集中，投资组合更稀疏；
- `T = 1`：普通 softmax；
- `T > 1`：权重更平均。

相关函数：

- `get_action_wrapper`
- `get_action_logprob_wrapper`
- `forward_action_wrapper`

这些函数还支持传入 mask。如果某只股票被 mask，就把它的 logits 减去很大数，使它几乎拿不到权重。

## 14. 当前代码与论文的几个差异

### 14.1 奖励函数差异

论文：

```text
r_t = V_t - V_{t-1}
```

当前 `EnvironmentASR`：

```text
reward = annualized Sharpe Ratio
```

这会让训练更关注风险调整收益。

### 14.2 CPU 分支差异

配置中如果没有 GPU，会构建普通 `AgentSAC`，并关闭 `if_use_rep` 和 `if_use_beta`。这意味着 CPU 模式不一定完整复现论文的 MaskSAC。

### 14.3 注释编码问题

部分原始中文注释在文件中显示为乱码，可能是历史编码问题。本次已在核心代码处补充了新的 UTF-8 中文注释，建议后续统一清理旧乱码注释。

## 15. 一次训练的数据流

```text
CSV 股票数据
  -> PortfolioManagementDataset
  -> EnvironmentASR.reset()
  -> state: [N, days, F]
  -> AgentMaskSAC.explore_env
  -> MaskTimeState.forward_state
  -> ActorMaskSAC.get_action
  -> Re-weighting
  -> env.step(action)
  -> ReplayBuffer.update
  -> AgentMaskSAC.update_net
  -> Critic loss / Actor loss / alpha loss / beta loss / rep loss
  -> validate_net
  -> ARR/SR/MDD 等指标
```

## 16. 如何运行

README 中给出的基本流程是：

```bash
sh tools/pipeline_mask_sac_dj30_example.sh
```

也可以直接运行训练脚本：

```bash
python tools/train.py --config configs/mask_sac_portfolio_management.py --root .
```

注意：

- 当前配置中的数据路径指向 `datasets/csp_817_data_cut/...`，仓库内未必包含完整数据。
- 如果缺数据，需要先准备或预处理数据。
- 如果在 CPU 上运行，配置会自动降低规模并可能切换到普通 SAC。

## 17. 读代码时抓住这条主线

不要一开始陷入所有算法文件。先抓住这条线：

```text
配置文件
  -> train.py
  -> Dataset
  -> Environment
  -> AgentMaskSAC
  -> MaskTimeState
  -> Actor/Critic
  -> ReplayBuffer
  -> metrics
```

能把这条线讲清楚，就已经理解了工程的核心。

## 18. 核心文件注释已补充

已补充中文注释的核心文件：

- `csp/EarnMore-main/tools/train.py`
- `csp/EarnMore-main/pm/dataset/portfolio_management_dataset.py`
- `csp/EarnMore-main/pm/environment/pm_based_portfolio_ASR.py`
- `csp/EarnMore-main/pm/net/mask_time_state.py`
- `csp/EarnMore-main/pm/net/sac/mask_sac_net.py`
- `csp/EarnMore-main/pm/agent/sac/mask_sac.py`
- `csp/EarnMore-main/pm/utils/helpers.py`

这些注释集中解释论文方法和代码变量之间的对应关系。
