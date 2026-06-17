# EarnMore 代码工程解析 v2

> **写给谁**：想把论文和代码对应起来、亲手跑通并改进 EarnMore 的初学者。
>
> **本版来历**：融合"详注版"`解读文档/03`（架构设计/注册器模式/数据流/五件套↔公式/对照表）与"精简版"`解读文档_gp/代码工程解析`（逐文件走查/差异清单）。文中标 ✅ 的事实已实地核对代码。
>
> **重要提示**：你拿到的 `EarnMore-main` 是论文官方仓库（DVampire/EarnMore）的**二次开发/魔改版**，与论文有出入（见第 7 节），并加了大量"实验管理/调参/监控"工程外围。**学习请抓主干：`pm/` 目录 + `tools/train.py`，外围了解即可。**

---

## 目录
1. [整体架构鸟瞰](#1-整体架构鸟瞰)
2. [目录结构](#2-目录结构)
3. [核心设计模式：注册器 + 配置驱动](#3-核心设计模式注册器--配置驱动)
4. [训练主循环与数据流](#4-训练主循环与数据流)
5. [三大核心模块逐一拆解](#5-三大核心模块逐一拆解)
6. [代码与论文的对应关系](#6-代码与论文的对应关系)
7. [本仓库相对论文的改动](#7-本仓库相对论文的改动)
8. [外围工程化工具](#8-外围工程化工具)
9. [如何跑起来](#9-如何跑起来)
10. [推荐阅读顺序](#10-推荐阅读顺序)

---

## 1. 整体架构鸟瞰

记住那句话：**EarnMore = SAC（强化学习）+ MAE（表征学习）+ 投资组合管理（任务）**。代码就围绕这三块组织。

```
            配置文件 (configs/*.py)  —— 定义所有超参 + 各组件"配方(dict)"
                        │ Config.fromfile()
                        ▼
   tools/train.py (训练总指挥)
     1) DATASET.build()    → PortfolioManagementDataset (读 CSV，组织成 股票×天×特征)
     2) ENVIRONMENT.build()→ EnvironmentASR (gym 环境，step/reset，算收益和奖励)
                            └ gym.vector.SyncVectorEnv 包成"向量环境"(可并行多子池)
     3) AGENT.build()      → AgentMaskSAC (智能体，内含 3 网络 + 优化器 + 训练逻辑)
            ├ rep_net: MaskTimeState  (MAE 表征网络  ←论文模块 a+b)
            ├ act_net: ActorMaskSAC   (演员/策略网络  ←论文模块 c)
            └ cri_net: CriticMaskSAC  (评论家/价值网络 ←论文模块 c)
     4) ReplayBuffer       → 经验回放(存交互数据，供异策略采样)
     5) for episode:
            explore_env()  ← 与环境交互、采数据、存 buffer
            update_net()   ← 抽样更新 Critic→Alpha→Actor→(Beta)→Rep 五件套
            validate()     ← 验证集跑一遍、算 ARR/SR/MDD、存最优模型
```

**一句话**：配置文件像"配方表"，`train.py` 照配方造出数据集/环境/智能体，再反复执行"交互采数据 → 存缓冲区 → 抽样更新网络 → 验证"，直到收敛。

---

## 2. 目录结构

```text
EarnMore-main/
├── configs/                  实验配置(每个算法一个)
│   └── mask_sac_portfolio_management.py   ← EarnMore 主配置
├── datasets/                 股票池文件(txt) + 部分数据
├── pm/                       ★★★ 核心算法包，学习重点
│   ├── registry.py           【枢纽】注册器，把"字符串名字"映射到"类"
│   ├── dataset/              读CSV、组织股票数据、生成子池 mask
│   ├── environment/          ★ RL环境(gym)：reset/step、算收益和奖励
│   │   ├── pm_based_portfolio_ASR.py     ← 当前配置用(奖励=夏普比率)
│   │   ├── pm_based_portfolio_value.py   ← 原版(奖励=市值变化，贴近论文)
│   │   └── pm_based_portfolio_return.py
│   ├── embed/times_embed.py  ← 论文模块(a)股票级嵌入(卷积+时序+位置)
│   ├── net/                  ★★★ 神经网络
│   │   ├── mae.py            ← 掩码自编码器基类(模块 a+b 骨架)
│   │   ├── mask_time_state.py← MAE子类，实现"掩码-编码-填充-重构"(§4.1)
│   │   └── sac/mask_sac_net.py← ActorMaskSAC + CriticMaskSAC (§4.2)
│   ├── agent/sac/mask_sac.py ← ★AgentMaskSAC，本论文主角(最核心)
│   ├── utils/
│   │   ├── replay_buffer.py  ← 经验回放(算法1的 D)
│   │   └── helpers.py        ← ★重加权 get_action_wrapper (§4.3 公式9)
│   ├── criterion/ optimizer/ scheduler/  ← 损失/优化器/调度(注册器包装)
│   └── metrics/              ← ARR/SR/MDD/SoR/CR/VOL 计算
├── tools/
│   ├── train.py              ← 训练主循环(学习重点)
│   ├── test.py               ← 测试/回测
│   └── preprocess.py / make_*.py
│   ─────── 以下是"魔改版"新增工程外围(了解即可) ───────
├── experiment_manager.py / hyperopt.py / scheduler.py / monitor.py / visualizer.py
└── README.md
```

> **学习优先级**：`net/mae.py` → `net/mask_time_state.py` → `net/sac/mask_sac_net.py` → `agent/sac/mask_sac.py` → `environment/pm_based_portfolio_ASR.py` → `tools/train.py`。吃透这 6 个，项目就懂 80%。

---

## 3. 核心设计模式：注册器 + 配置驱动

初学者读这份代码**最大的拦路虎**，先讲清它。

### 3.1 注册器（Registry）🔧
`pm/registry.py` 只有十行：
```python
from mmengine.registry import Registry
DATASET     = Registry('dataset',     locations=['pm.dataset'])
NET         = Registry('net',         locations=['pm.net'])
AGENT       = Registry('agent',       locations=['pm.agent'])
ENVIRONMENT = Registry('environment', locations=['pm.environment'])
# ... optimizer / scheduler / criterion / embed
```
**它解决什么？** 想"用一个字符串名字就创建对应的类"，而不用到处写 `if name=='xxx'`。注册器就是一本"电话簿"：类定义上方写 `@NET.register_module()` 登记进簿；之后 `NET.build(dict(type="ActorMaskSAC", embed_dim=64))` 自动查簿、找到类、用参数造出来。

### 3.2 配置驱动（Config-driven）🔧
整个项目"长什么样"全写在 `configs/mask_sac_portfolio_management.py`——本质是一堆嵌套 `dict`，描述"用哪些组件、各组件什么参数"。`train.py` 用 `Config.fromfile()` 读入、再 `.build()` 造出来。

**好处**：改实验只改配置、不动核心代码。**坏处**：初学者觉得"到处是 dict、找不到干活的类"——记住"`type` 字段的值 = 真正的类名，去对应目录搜它"即可。

> 💡 这套 `Registry + Config` 是商汤 mmengine（OpenMMLab）的标准玩法，学会后看 mmdetection 等项目都通用。

**主配置里有什么**（`mask_sac_portfolio_management.py`）：`num_stocks` / `days` / `num_features` / 日期区间 / `buffer_size` / `batch_size` / `horizon_len` / `embed_dim` / `mask_ratio_min,max,mu,std` / `gamma` / `repeat_times` / `soft_update_tau` / 重加权温度 `T`。论文对应：`rep_net=MaskTimeState`（可掩码表征）、`act_net=ActorMaskSAC`、`cri_net=CriticMaskSAC`、`action_wrapper_method="reweight"`（§4.3）。

> ✅ 注意：配置顶部用 `check_gpu_available()` 自动切分支——GPU 用完整 `AgentMaskSAC`（含掩码表征），CPU 退化为普通 `AgentSAC` 并关掉 `if_use_rep`/`if_use_beta`。**学习以 GPU 分支为准**（那才是 EarnMore）。

---

## 4. 训练主循环与数据流

### 4.1 一条数据的"形状变化"之旅 🎯
记号：`B`=批量, `N`=股票数, `D`=天数, `F`=特征数, `C`=嵌入维度。
```
① 环境给原始状态                state: (B, N, D, F)   例 (32, 790, 10, 67)
      │ rep.forward_state() ← MaskTimeState
② 股票级嵌入 (TimesEmbed: 卷积+时序+位置, 沿天数求均值)
      │                            (B, N, C)   每只股票→一个C维向量
      │ 随机掩码→只留未掩码送编码器→Transformer编码
③ 池级潜在嵌入                    (B, N_keep, C)
      │ 用可学习 mask_token 填回 N 个 → 按 ids_restore 还原顺序
④ 可掩码股票表征 ρ (论文核心状态)  (B, N, C)
      │ act.get_action(ρ) ← ActorMaskSAC
⑤ 每只股票/现金 logits → 重加权(温度softmax)
⑥ 动作=组合权重                  (B, N+1)  加起来=1, 第0位现金
      │ env.step(action)
⑦ 环境算收益、新市值、奖励, 进入下一步
```

### 4.2 训练主循环（`tools/train.py: main()`）🎯
去掉日志枝节，主干：
```python
dataset = DATASET.build(cfg.dataset)
train_envs = gym.vector.SyncVectorEnv([...])   # 向量环境(可并行多子池)
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
`train_one_episode` 内核：
```python
while True:
    buffer_items = agent.explore_env(environment, horizon_len)  # 交互采数据
    buffer.update(buffer_items)                                 # 存缓冲区
    logging_tuple = agent.update_net(buffer)                    # 抽样更新网络 ★
    if done: break
```
**这就是 RL 的"采样—学习"循环**，对应论文算法 1 的两个 for。

---

## 5. 三大核心模块逐一拆解

### 5.1 表征模块：MAE（`mae.py` + `mask_time_state.py`）🎯🎯
对应**论文模块 (a)(b) 和 §4.1**，EarnMore 第一大创新的载体。

**`mae.py` — MAE 基类**：
- `random_masking(x, mask_ratio)`：核心。给每只股票随机噪声、按噪声排序、保留前 `len_keep=L*(1-mask_ratio)` 只（未掩码）。返回 `x_masked`（未掩码嵌入）、`mask`（0=保留/1=掩码）、`ids_restore`（把打乱顺序还原的索引，关键！）。
- `forward_encoder`：嵌入→掩码→加 [CLS] 和位置编码→过 Transformer→潜在嵌入。
- `forward_decoder`：用 `mask_token` 补齐到 N 个→按 `ids_restore` 还原顺序→过解码器→预测被掩码股票的"价格 patch"。
- `forward_loss`：只在**被掩码**股票上算 MSE（论文公式 8）。
- `mask_ratio_generator`：用 `scipy.stats.truncnorm` 实现**截断高斯**（公式 3），从 [0.6, 0.8] 采样掩码率。

**`mask_time_state.py` — MaskTimeState（实际使用的表征网络）**：
- 继承 MAE，**关键升级**是用 `CrossAttention` 解码器：解码时不仅看被填充的 query，还把"完整未掩码序列"当 key/value 参考，重构更准。
- 最重要的是 **`forward_state()`**——**RL 用的接口**：只做"编码→填充 mask_token→还原顺序"，输出**可掩码股票表征 $\rho$**，但**不做重构、不算损失**（重构损失在 `forward()` 单独算）。
- 它支持 `[batch, stock, days, features]` 与 `[batch, channel, stock, days, features]` 两种输入。

> **最易迷糊点**：同一个 `MaskTimeState` 有两套前向——`forward_state()` 出"状态表征"给 RL 用，`forward()` 出"重构损失"给自监督训练用。两者**同时进行**（论文强调的"端到端、不预训练"），这就是 `update_net` 里既更新 RL 又更新 rep 的原因。

**`times_embed.py` — TimesEmbed（股票级嵌入）**：做三件事——`TokenEmbedding`（1D 卷积编码价格/技术指标）、`TemporalEmbedding`（编码 weekday/day/month）、`PositionalEmbedding`（时间位置编码），相加后对天数维求平均，把"某股近 10 天一堆特征"压成一个向量。

### 5.2 决策模块：Actor & Critic（`mask_sac_net.py`）🎯🎯
对应**论文模块 (c) 和 §4.2**。
- **`ActorMaskSAC`**：输入 $\rho$ `(B,N,C)`；加 [CLS]（对应现金仓位）→几层 MLP→输出每个 token 的动作均值与对数标准差。`get_action` 用 `Normal(...).rsample()` **重参数化采样**（公式 6 的 $f_\phi(\epsilon;s)$）得 logits、再 softmax 成权重；`get_action_logprob` 还返回**对数概率**（SAC 算熵正则用）。✅ 该版有个魔改技巧 `soft_logits = logits * torch.log(indices+1)`（按排序加权，论文原版没有）。
- **`CriticMaskSAC`**：输入 $\rho$ + 动作，输出两个 Q（**双 Q 网络**，公式 5 用 target 时取 min 抑制高估）。`get_q1_q2` 返回两个 Q（训练用）；`get_q_min` 返回较小 Q（算 TD 目标用）。

### 5.3 智能体：AgentMaskSAC（`mask_sac.py`，最核心）🎯🎯🎯
把所有零件组装成完整 SAC。理解两个主方法就理解整个训练：

**`explore_env(env, horizon_len)` — 采样**：
```
for t in horizon_len:
    rep_state = rep.forward_state(state)   # 原始状态→可掩码表征
    action = get_action(rep_state)          # 表征→动作(经重加权)
    next_state, reward, done = env.step(action)
    存储 (state, action, mask, ids_restore, reward, done, next_state)
```
> ✅ 存 `mask` 和 `ids_restore` 是该版相对论文的工程细节：把每步掩码也存进缓冲区，更新时复用，保证 next_state 用同样掩码编码。

**`update_net(buffer)` — 学习（论文算法 1 第二个 for）** 🎯🎯🎯：每个 batch 严格按论文顺序更新**五件套**：
```python
for _ in range(repeat_times):
    # ① Critic(Q网络): 最小化 TD 误差
    obj_critic, ..., rep_states = self.get_obj_critic(buffer, batch_size)  # 公式(5)
    self.optimizer_update(self.cri_optimizer, obj_critic)
    self.soft_update(self.cri_target, self.cri, tau)        # 目标网络软更新
    # ② Alpha(温度): 自动熵调节
    action_pg, log_prob = self.get_action_logprob(rep_states)
    obj_alpha = (alpha_log * (target_entropy - log_prob).detach()).mean()  # 公式(7)
    # ③ Actor(策略): 最大化 Q - α·logπ
    q_value_pg = self.cri_target(rep_states, action_pg).mean()
    obj_actor = (q_value_pg - log_prob * alpha).mean()        # 公式(6)
    self.optimizer_update(self.act_optimizer, -obj_actor)     # 注意负号(梯度上升)
    # ④ Beta(掩码惩罚): 惩罚给被掩码股票分配权重 ← §4.2"监督损失"
    if if_use_beta: ...
    # ⑤ Rep(MAE表征): 最小化重构损失
    if if_use_rep: rep_loss = self.update_rep_net(...)        # 公式(8)
```

**`get_obj_critic_raw` — TD 误差（贝尔曼方程的代码实现）** 🎯：
```python
with torch.no_grad():
    rep_next_states = rep.forward_state(next_states, mask, ids_restore)
    next_as, next_logprobs = get_action_logprob(rep_next_states)
    next_qs = cri_target.get_q_min(rep_next_states, next_as)    # 双Q取min
    q_labels = rewards + (1-dones)*gamma*(next_qs - next_logprobs*alpha)  # ← 公式(5)!
    rep_states = rep.forward_state(states, mask, ids_restore)
q1, q2 = self.cri.get_q1_q2(rep_states, actions)
obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)  # 两个Q都算
```
**这段 = 论文公式 (5) = 贝尔曼方程 = 学习清单第 3、6 部分全部理论。** 反复看，理论与实践就打通。

**`get_object_beta` — 掩码惩罚（§4.2"监督损失"）** ✅：
```python
weight = self.forward_action(rep_states)              # Actor 给出的权重
beta_loss = (weight * mask_bool).sum(dim=1).mean()    # 给被掩码股票的权重之和→趋近0
```
这正是论文说的"第一种方法：给 actor 输出加监督损失"，论文实验证明它优于"在 TD 误差上加惩罚"。

### 5.4 重加权：动作包装器（`helpers.py`）🎯
对应**论文 §4.3、公式 (9)**。`get_action_wrapper` 给 Actor 输出再包一层：
```python
def get_action(x, mask=None, ...):
    pred = func(x)                          # Actor 原始 logits
    if mask is not None:
        pred = pred - mask_bool * 1e6       # 被掩码股票 logit 减巨大值→softmax后≈0
    weight = F.softmax(pred / T, dim=-1)    # ★ 温度 softmax = 公式(9) Re(x)
    return weight
```
- **温度 $T$**：越小 softmax 越"尖锐"、权重越集中（稀疏化）；$T=1$ 是普通 softmax。
- **掩码减大常数**：推理时投资者指定子池，池外股票被减 `1e6`、softmax 后≈0——优雅实现"忽略池外股票"。
- 相关函数：`get_action_wrapper`、`get_action_logprob_wrapper`、`forward_action_wrapper`。

### 5.5 环境：EnvironmentASR（`pm_based_portfolio_ASR.py`）🔧
实现 gym 的 `reset()`（训练模式随机选起始日做数据增强）和 `step(action)`：取动作权重（第 0 位现金）→推进一天→算组合收益 `portfolio_ret = Σ(收盘价涨跌幅 × 各股权重)`→更新市值（公式 1）→**算奖励**。
> ✅ 实地核对：`step` 里 `reward = qs_stats.sharpe(rets, rf=0.02)`，再 `if NaN/Inf: reward=0`、`reward = max(reward, 0.0)`——**这版用"年化夏普比率(裁剪为非负)"当奖励**；而论文原版用"市值变化 $V_t-V_{t-1}$"（`pm_based_portfolio_value.py` 里确为 `reward = new_portfolio_value - old_portfolio_value`）。这是重要区别。

### 5.6 数据集 & 子池掩码（`portfolio_management_dataset.py`）🔧
- 读 GSP 股票列表（`stocks_path`）、每股特征 CSV（`data_path`）、自定义子池 txt（`aux_stocks_path`）。
- 核心字段：`self.stocks`（GSP 代码）、`self.stocks2id`/`id2stocks`、`self.aux_stocks`（子池）、`self.stocks_df`。
- 每个 aux txt 定义一个 **CSP** 并生成 `mask`：`"mask": np.array([0.0 if s in stocks else 1.0 for s in self.stocks])`——池内 mask=0、池外 mask=1。✅ `aux_stocks[0]` 被设为 `All`（完整 GSP，mask 全 0）。**这就是"可定制股票池"在数据层的落地**——验证时每个 aux 子池建一个环境、测模型对不同子池的适应力。

---

## 6. 代码与论文的对应关系

| 论文概念 | 公式/章节 | 代码位置 |
|---|---|---|
| 股票级嵌入 $l_s$ | §4.1 公式(2) | `embed/times_embed.py: TimesEmbed.forward()` |
| 随机掩码率 $g(r)$ | 公式(3) | `net/mae.py: mask_ratio_generator (truncnorm)` |
| 掩码操作 $\eta_{mo}$ | 公式(4) | `net/mae.py: random_masking()` |
| 编码器 $\psi_{enc}$ | 公式(4) | `net/mask_time_state.py: forward_encoder()` |
| 掩码填充 $\eta_{mf}$ + 表征 $\rho$ | 公式(4) | `net/mask_time_state.py: forward_state()` |
| 解码器 $\psi_{dec}$ + 重构 | 公式(4) | `net/mask_time_state.py: forward_decoder()` |
| 重构损失(MSE) | 公式(8) | `net/mask_time_state.py: forward_loss()` |
| Q 网络优化(TD误差) | 公式(5) | `agent/sac/mask_sac.py: get_obj_critic_raw()` |
| 目标网络 $\bar\theta$ | 公式(5) | `agent/sac/mask_sac.py: soft_update(), cri_target` |
| 策略优化(重参数化) | 公式(6) | `net/sac/mask_sac_net.py: get_action_logprob() + rsample()` |
| Alpha 自动调节 | 公式(7) | `agent/sac/mask_sac.py: obj_alpha, alpha_log` |
| 掩码监督惩罚 | §4.2 | `agent/sac/mask_sac.py: get_object_beta()` |
| 重加权 $Re(x)$ | §4.3 公式(9) | `utils/helpers.py: get_action_wrapper()` |
| 训练算法 | 算法1 | `tools/train.py: main()` + `agent.update_net()` |
| 推理算法 | 算法2 | `agent/sac/mask_sac.py: validate_net()` |
| 6个金融指标 | 附录A | `metrics/` 下 ARR/SR/CR/SOR/MDD/VOL |

---

## 7. 本仓库相对论文的改动 ✅

学习时务必注意，这份"魔改版"与论文有以下出入（不是 bug，是作者二次开发）：

1. **奖励函数不同** ✅：当前 `EnvironmentASR` 奖励 = **夏普比率**（裁剪为非负）；论文原版 = **市值变化 $V_t-V_{t-1}$**（`pm_based_portfolio_value.py`）。
2. **数据集换了**：论文用 SP500/DJ30（美股 2007~2022）；这份配置用自处理数据（`csp_817_data_cut`，约 790 只股票），路径写死在配置里。
3. **特征维度变了**：论文 F=102（OHLC+技术指标+时序）；这份 `num_features=67`。
4. **GPU/CPU 双分支** ✅：`check_gpu_available()` 自动切——GPU 用 `AgentMaskSAC`（完整掩码版），CPU 退化普通 `AgentSAC`（关掉 rep/beta，省算力，**不完整复现 MaskSAC**）。
5. **Actor 加排序加权** ✅：`soft_logits = logits * log(indices+1)`，论文没有。
6. **新增工程外围**：`hyperopt/scheduler/monitor/visualizer/experiment_manager.py`，与算法无关。
7. **注释编码**：部分旧中文注释为乱码（历史编码问题）；`解读文档/带详细注释的核心代码/` 已补 UTF-8 注释。

---

## 8. 外围工程化工具（了解即可）

| 文件 | 作用 | 典型命令 |
|---|---|---|
| `experiment_manager.py` | 创建/列出/管理实验 | `python experiment_manager.py list-experiments` |
| `hyperopt.py` | 超参搜索(随机/贝叶斯) | `python hyperopt.py random-search --template mask_sac_base --num-samples 2` |
| `scheduler.py` | 并发调度多个实验 | `python scheduler.py run --max-concurrent 4` |
| `monitor.py` | 训练监控/早停/记录 | `python monitor.py status --experiment exp_xxx` |
| `visualizer.py` | 批量画图/对比 | `python visualizer.py compare-experiments` |

`quickhelp.txt` 有作者整理的完整命令流程。

---

## 9. 如何跑起来

> ⚠️ 研究代码，依赖较重（PyTorch + timm + einops + mmengine + qlib + gym + quantstats 等），数据路径写死在配置里。下面是大致流程。

```bash
# 1. 装依赖
pip install -r requirements.txt        # qlib 需单独装
# 2. 准备数据(放到配置 data_path 指定目录): features/*.csv, stocks_cut.txt, aux_stocks_files_cleaned/*.txt
# 3. 训练
python tools/train.py --config configs/mask_sac_portfolio_management.py --root .
# 4. 或用作者 pipeline
sh tools/pipeline_mask_sac_dj30_example.sh
```
注意：当前配置数据路径指向 `datasets/csp_817_data_cut/...`，仓库内未必含完整数据；CPU 上会自动降规模、可能切普通 SAC；`train.py` 末尾可能调 `send_email(...)`，跑前可注释掉。**建议先读懂核心 6 个文件、再跑通**（环境配置往往最耗时）。

---

## 10. 推荐阅读顺序

1. **建立全局观**：本文第 1~4 节 + `02_学习清单`，搞清"三大件 + 主循环"。
2. **配合理论攻核心**（每读一个就回对论文公式）：`times_embed.py`（股票级嵌入）→ `mae.py`（MAE 骨架）→ `mask_time_state.py`（表征，§4.1）→ `mask_sac_net.py`（Actor/Critic，§4.2）→ `helpers.py`（重加权，§4.3）→ `pm_based_portfolio_ASR.py`（环境）→ `mask_sac.py`（智能体，算法1）★最后啃。
3. **看主循环**：`tools/train.py`，理解一个 episode 怎么跑。
4. **动手改**：改 `configs` 超参（如温度 `T`、掩码率范围），观察对结果的影响——最好的学习方式。

> **学懂标志**：能对着 `update_net()` 逐行说出"这步更新哪个网络（Critic/Alpha/Actor/Beta/Rep）、对应论文哪个公式（5/7/6/§4.2/8）、为什么这么做"。配合 `../解读文档/带详细注释的核心代码/` 逐行注释印证，效果最佳。
