# EarnMore 代码工程深度解析

> **写给谁看**：想把论文和代码对应起来、亲手跑通并改进 EarnMore 的初学者。
>
> **配套阅读**：`01_论文全文翻译.md`（理解算法）、`02_强化学习知识点学习清单.md`（补理论）、`带详细注释的核心代码/`（逐行中文注释）。
>
> **重要提示**：你拿到的这份代码是论文官方仓库（DVampire/EarnMore）的一个**二次开发/魔改版本**，与论文有一些出入（下面会专门指出），它增加了大量"实验管理、超参搜索、监控、可视化"的工程化外围工具。**学习时请抓主干（`pm/` 目录 + `tools/train.py`），外围工具了解即可。**

---

## 目录
1. [整体架构鸟瞰](#1-整体架构鸟瞰)
2. [目录结构详解](#2-目录结构详解)
3. [核心设计模式：注册器 + 配置驱动](#3-核心设计模式注册器--配置驱动)
4. [数据流与训练主循环](#4-数据流与训练主循环)
5. [三大核心模块逐一拆解](#5-三大核心模块逐一拆解)
6. [代码与论文的对应关系](#6-代码与论文的对应关系)
7. [本仓库相对论文的改动](#7-本仓库相对论文的改动)
8. [外围工程化工具](#8-外围工程化工具)
9. [如何跑起来](#9-如何跑起来)
10. [学习路径建议](#10-学习路径建议)

---

## 1. 整体架构鸟瞰

EarnMore 的核心可以用一张"数据流动图"来理解。记住论文那句话：**EarnMore = SAC（强化学习） + MAE（表征学习） + 投资组合管理（任务）**。代码就是围绕这三块组织的。

```
                         ┌─────────────────────────────────────────┐
                         │           配置文件 (configs/*.py)          │
                         │   定义所有超参 + 各组件的"配方(dict)"       │
                         └────────────────────┬────────────────────┘
                                              │ Config.fromfile()
                                              ▼
   ┌──────────────────────────────  tools/train.py (训练总指挥)  ──────────────────────────────┐
   │                                                                                            │
   │   1) DATASET.build()   →  PortfolioManagementDataset  (读 CSV，组织成 股票×天×特征)          │
   │                                                                                            │
   │   2) ENVIRONMENT.build()→  EnvironmentASR (gym 环境，负责 step/reset，算收益和奖励)           │
   │                            └→ 用 gym.vector.SyncVectorEnv 包成"向量环境"(可并行多个子池)      │
   │                                                                                            │
   │   3) AGENT.build()     →  AgentMaskSAC  (智能体，内含 3 个网络 + 优化器 + 训练逻辑)           │
   │         ├── rep_net :  MaskTimeState   (MAE 表征网络  ←论文模块 a+b)                          │
   │         ├── act_net :  ActorMaskSAC    (演员/策略网络  ←论文模块 c)                           │
   │         └── cri_net :  CriticMaskSAC   (评论家/价值网络 ←论文模块 c)                          │
   │                                                                                            │
   │   4) ReplayBuffer       →  经验回放缓冲区 (存交互数据，供异策略采样)                            │
   │                                                                                            │
   │   5) for episode in range(...):                                                            │
   │          explore_env()   ← 智能体与环境交互、采数据、存进 buffer                               │
   │          update_net()    ← 从 buffer 采样、更新 Critic→Alpha→Actor→(Beta)→Rep 五件套           │
   │          validate()      ← 在验证集上跑一遍、算 ARR/SR/MDD 等指标、存最优模型                    │
   └────────────────────────────────────────────────────────────────────────────────────────┘
```

**一句话概括运行流程**：配置文件像一张"配方表"，`train.py` 照着配方把数据集、环境、智能体三大件造出来，然后反复执行"交互采数据 → 存缓冲区 → 抽样更新网络 → 验证"这个循环，直到收敛。

---

## 2. 目录结构详解

```
EarnMore-main/
├── pm/                          ★★★ 核心算法包 (Portfolio Management)，学习重点
│   ├── registry.py              【枢纽】注册器，把"字符串名字"映射到"类"
│   ├── dataset/                 数据集：读CSV、组织股票数据、生成子池掩码
│   │   └── portfolio_management_dataset.py
│   ├── environment/             ★ RL环境(gym接口)：reset/step、算收益和奖励
│   │   ├── pm_based_portfolio_ASR.py      ← 当前配置用的环境(奖励=夏普比率)
│   │   ├── pm_based_portfolio_value.py    ← 原版环境(奖励=市值变化，贴近论文)
│   │   └── pm_based_portfolio_return.py
│   ├── embed/                   嵌入层：把原始特征变成向量
│   │   ├── times_embed.py       ← 论文模块(a)股票级嵌入(卷积+时序+位置)
│   │   └── patch_embed.py
│   ├── net/                     ★★★ 神经网络定义
│   │   ├── mae.py               ← 掩码自编码器基类(论文模块 a+b 的骨架)
│   │   ├── mask_time_state.py   ← MAE子类,实现"掩码-编码-填充-重构"(论文§4.1)
│   │   ├── sac/mask_sac_net.py  ← ActorMaskSAC + CriticMaskSAC (论文§4.2)
│   │   ├── qnet/, ppo/, ddpg/, TD3/  ← 其他算法的网络(对照学习用)
│   ├── agent/                   ★★★ 智能体(把网络+优化器+训练逻辑打包)
│   │   ├── sac/mask_sac.py      ← AgentMaskSAC，本论文的主角(632行,最核心)
│   │   ├── sac/sac.py           ← 标准SAC(无掩码)
│   │   └── dqn/, ppo/, ddpg/, TD3/   ← 其他算法智能体
│   ├── utils/                   工具：回放缓冲区、动作包装器(重加权)、检查点等
│   │   ├── replay_buffer.py     ← 经验回放(论文算法1的 D)
│   │   ├── helpers.py           ← ★重加权 get_action_wrapper (论文§4.3公式9)
│   │   └── misc.py
│   ├── criterion/  optimizer/  scheduler/   ← 损失函数/优化器/学习率调度(都用注册器包装)
│   └── metrics/                 金融指标 ARR/SR/MDD/SoR/CR/VOL 的计算
│       └── metric_new.py
│
├── configs/                     ★★ 配置文件(每个算法一个)，定义所有超参
│   ├── mask_sac_portfolio_management.py   ← EarnMore 主配置
│   └── sac_/ppo_/ddpg_/dqn_/td3_...py
│
├── tools/                       ★ 入口脚本
│   ├── train.py                 ← 训练主循环(学习重点)
│   ├── test.py / test_arr.py    ← 测试/回测
│   ├── preprocess.py            ← 数据预处理
│   └── make_pipeline.py / make_scripts.py  ← 批量生成实验脚本
│
├── datasets/                    数据(CSV格式的股票特征 + 子池定义txt)
│
│   ─────────── 以下是"魔改版"新增的工程化外围(了解即可) ───────────
├── experiment_manager.py        实验管理(创建/列出/对比实验)
├── hyperopt.py                  超参数优化(随机搜索/贝叶斯搜索)
├── scheduler.py                 实验调度器(并发跑多个实验)
├── monitor.py                   训练监控(早停、指标记录)
├── visualizer.py                可视化(画对比图)
├── quickhelp.txt                作者写的快速上手命令清单
└── README.md
```

> **学习优先级**：`pm/net/mae.py` → `pm/net/mask_time_state.py` → `pm/net/sac/mask_sac_net.py` → `pm/agent/sac/mask_sac.py` → `pm/environment/pm_based_portfolio_ASR.py` → `tools/train.py`。把这 6 个文件吃透，整个项目就懂了 80%。

---

## 3. 核心设计模式：注册器 + 配置驱动

这是初学者读这份代码**最大的拦路虎**，先讲清楚它，后面看代码就顺了。

### 3.1 注册器（Registry）模式 🔧

打开 `pm/registry.py`，只有 10 行：

```python
from mmengine.registry import Registry
DATASET     = Registry('dataset',     locations=['pm.dataset'])
NET         = Registry('net',         locations=['pm.net'])
AGENT       = Registry('agent',       locations=['pm.agent'])
OPTIMIZER   = Registry('optimizer',   locations=['pm.optimizer'])
SCHEDULER   = Registry('scheduler',   locations=['pm.scheduler'])
CRITERION   = Registry('criterion',   locations=['pm.criterion'])
ENVIRONMENT = Registry('environment', locations=['pm.environment'])
EMBED       = Registry('embed',       locations=['pm.embed'])
```

**它解决什么问题？** 想象你有几十个类（各种网络、环境、算法），想"用一个字符串名字就能创建对应的类"，而不用到处写 `if name == 'xxx': return Xxx()`。注册器就是一本"**电话簿**"：
- 在类定义上方写 `@NET.register_module()`，就把这个类"登记"进电话簿，名字默认是类名。
- 之后 `NET.build(dict(type="ActorMaskSAC", embed_dim=64))` 就会自动查电话簿、找到 `ActorMaskSAC` 类、用后面的参数把它造出来。

**实际例子**（来自 `mask_sac_net.py`）：
```python
@NET.register_module(force=True)      # ← 登记进电话簿
class ActorMaskSAC(nn.Module):
    def __init__(self, embed_dim=128, depth=2, cls_embed=True, ...):
        ...
```
配置里写：
```python
act_net = dict(type="ActorMaskSAC", embed_dim=64, depth=1, cls_embed=True)
```
代码里造：
```python
self.act = NET.build(act_net)   # 等价于 ActorMaskSAC(embed_dim=64, depth=1, cls_embed=True)
```

### 3.2 配置驱动（Config-driven）🔧

整个项目"**长什么样**"全写在 `configs/mask_sac_portfolio_management.py` 这个配置文件里——它本质是一堆嵌套的 Python 字典（`dict`），描述了"要用哪些组件、各组件什么参数"。`train.py` 用 `Config.fromfile()` 把它读进来，再用 `.build()` 逐个造出来。

**好处**：改实验只要改配置文件，不用动核心代码。**坏处**：初学者会觉得"代码里到处是 dict、找不到真正干活的类在哪"——记住"`type` 字段的值 = 真正的类名，去对应目录搜这个类名"即可。

> 💡 这套 `Registry + Config` 是商汤 mmengine（OpenMMLab）框架的标准玩法，在 CV 领域很常见。学会它，以后看 mmdetection 等项目都通用。

---

## 4. 数据流与训练主循环

### 4.1 一条数据的"形状变化"之旅 🎯

跟着一个 batch 的数据走一遍，你就理解了整个前向过程。维度记号：`B`=批量, `N`=股票数, `D`=天数, `F`=特征数, `C`=嵌入维度。

```
① 环境给出原始状态                    state:  (B, N, D, F)      例如 (32, 790, 10, 67)
        │  rep.forward_state()  ← MaskTimeState (MAE)
        ▼
② 股票级嵌入 (TimesEmbed: 卷积+时序+位置, 沿天数求均值)
        │                                      (B, N, C)        每只股票变成一个C维向量
        │  随机掩码 → 只留未掩码的送编码器 → Transformer编码
        ▼
③ 池级潜在嵌入                                 (B, N_keep, C)
        │  用可学习的 mask_token 填充回 N 个 → unshuffle 还原顺序
        ▼
④ 可掩码股票表征 ρ  (论文的核心状态)            (B, N, C)
        │  act.get_action(ρ)  ← ActorMaskSAC
        ▼
⑤ 每只股票/现金的 logits → 重加权(温度softmax)
        ▼
⑥ 动作=投资组合权重                            (B, N+1)         加起来=1, 第0位是现金
        │  env.step(action)
        ▼
⑦ 环境算出 收益、新市值、奖励, 进入下一时间步
```

### 4.2 训练主循环（`tools/train.py: main()`）🎯

去掉监控、日志等枝节，主干就是这样：

```python
# === 准备阶段 ===
dataset = DATASET.build(cfg.dataset)                  # 造数据集
train_environment = ENVIRONMENT.build(cfg.environment) # 造训练环境
train_envs = gym.vector.SyncVectorEnv([...])          # 包成向量环境(可并行多子池)
val_envs   = gym.vector.SyncVectorEnv([...])          # 验证环境(每个aux子池一个)
agent = AGENT.build(cfg.agent)                        # 造智能体(内含3网络)
buffer = ReplayBuffer(...)                            # 造回放缓冲区

agent.last_state = train_envs.reset()                 # 初始化
buffer.update(agent.explore_env(train_envs, horizon_len))  # 先采一批数据填充缓冲区

# === 训练循环 ===
for episode in range(start, num_episodes+1):
    # 1) 训练一个回合: 反复"采数据→更新网络"直到环境done
    train_one_episode(train_envs, buffer, agent, horizon_len)
    # 2) 在验证集跑一遍, 算 ARR/SR/MDD 等
    val_stats = validate(val_envs, agent)
    # 3) 如果验证指标(ASR)创新高, 保存为最优模型
    if metric > max_metrics:
        save_checkpoint(episode, agent, exp_path, if_best=True)
```

其中 `train_one_episode` 的内核（`train.py` 第 410 行）：
```python
while True:
    buffer_items = agent.explore_env(environment, horizon_len)  # 交互采数据
    buffer.update(buffer_items)                                 # 存入缓冲区
    logging_tuple = agent.update_net(buffer)                    # 抽样更新网络 ★
    if 环境结束(done): break
```

**这就是 RL 的"采样—学习"循环**，对应论文算法 1 的两个 for 循环。

---

## 5. 三大核心模块逐一拆解

### 5.1 表征模块：MAE（`mae.py` + `mask_time_state.py`）🎯🎯

对应**论文模块 (a)(b) 和第 4.1 节**。这是 EarnMore 第一大创新的载体。

**`mae.py` — MAE 基类**：定义了掩码自编码器的通用骨架。
- `random_masking(x, mask_ratio)`：核心中的核心。它给每只股票一个随机噪声、按噪声排序、保留前 `len_keep=L*(1-mask_ratio)` 只（=未掩码），其余视为被掩码。返回：
  - `x_masked`：只剩未掩码股票的嵌入；
  - `mask`：二值掩码（0=保留, 1=掩码）；
  - `ids_restore`：用于把打乱的顺序还原回去的索引（关键！后面"填充+还原"靠它）。
- `forward_encoder`：嵌入 → 掩码 → 加 [CLS] 和位置编码 → 过 Transformer → 输出潜在嵌入。
- `forward_decoder`：用 `mask_token` 把潜在嵌入补齐到 N 个 → 按 `ids_restore` 还原顺序 → 过解码器 → 预测被掩码股票的"价格 patch"。
- `forward_loss`：只在**被掩码**的股票上算 MSE 重构损失（对应论文公式 8）。
- `mask_ratio_generator`：用 `scipy.stats.truncnorm` 实现的**截断高斯分布**，对应论文公式 (3)，从 [0.6, 0.8] 采样掩码率。

**`mask_time_state.py` — MaskTimeState（MAE 子类，实际使用的表征网络）**：
- 它继承 MAE，但**关键升级**是用了 `CrossAttention`（交叉注意力）的解码器：解码时不仅看被填充的 query，还把"完整未掩码序列"当作 key/value 来参考，重构得更准。
- 最重要的方法是 **`forward_state()`**：这是**RL 用的接口**——它只做"编码→填充 mask_token→还原顺序"，输出**可掩码股票表征 $\rho$**（论文的状态），但**不做重构、不算损失**（重构损失在 `forward()` 里单独算）。Agent 拿到 $\rho$ 后送给 Actor/Critic。
- `forward()`：训练表征用，返回重构损失（供 `update_rep_net` 更新）。

> **初学者最容易迷糊的点**：同一个 `MaskTimeState` 有两套前向——`forward_state()` 出"状态表征"给 RL 用，`forward()` 出"重构损失"给自监督训练用。两者**同时进行**（论文强调的"端到端、不预训练"），这就是 `update_net` 里既更新 RL 又更新 rep 的原因。

### 5.2 决策模块：Actor & Critic（`mask_sac_net.py`）🎯🎯

对应**论文模块 (c) 和第 4.2 节**。

**`ActorMaskSAC`（演员/策略网络）**：
- 输入：可掩码股票表征 $\rho$，形状 `(B, N, C)`。
- 结构：加一个 [CLS] 标记 → 几层 MLP → 输出每个 token 的 2 个值（动作均值 `a_avg` 和对数标准差 `a_std_log`）。
- `get_action(x)`：用 `Normal(a_avg, a_std).rsample()` **重参数化采样**（对应论文公式 6 的 $f_\phi(\epsilon;s)$）得到 logits，再 softmax 成权重。
- `get_action_logprob(x)`：除了动作，还返回**对数概率 `logprob`**——SAC 算熵正则要用它。
- 注意这里有个特别处理 `soft_logits = logits * torch.log(indices+1)`，是该魔改版加的一个按排序加权的技巧（论文原版没有）。

**`CriticMaskSAC`（评论家/价值网络）**：
- 输入：表征 $\rho$ **和** 动作 `action`，拼在一起。
- 输出两个 Q 值（**双 Q 网络**，对应论文公式 5 用 target 时取 min 来抑制高估）。
- `get_q1_q2`：返回两个 Q（训练 Critic 用）；`get_q_min`：返回较小的 Q（算 TD 目标用）。

### 5.3 智能体：AgentMaskSAC（`mask_sac.py`，最核心 632 行）🎯🎯🎯

这是把上面所有零件**组装成完整 SAC 算法**的地方。理解它的两个主方法就理解了整个训练：

**`explore_env(env, horizon_len)` — 采样（与环境交互）**：
```
for t in range(horizon_len):
    rep_state = rep.forward_state(state)   # 原始状态 → 可掩码表征
    action = get_action(rep_state)          # 表征 → 动作(经过重加权)
    next_state, reward, done = env.step(action)   # 环境给反馈
    存储 (state, action, mask, ids_restore, reward, done, next_state)
```
注意它存储了 `mask` 和 `ids_restore`——这是该版本相对论文的一个工程细节：把每步的掩码也存进缓冲区，更新时复用，保证 next_state 用同样的掩码编码。

**`update_net(buffer)` — 学习（更新网络）** 🎯🎯🎯：
这是论文算法 1 第二个 for 循环的实现。每个 batch 严格按论文说的顺序更新**五件套**：
```python
for _ in range(repeat_times):
    # ① 更新 Critic (Q网络): 最小化 TD 误差
    obj_critic, ..., rep_states = self.get_obj_critic(buffer, batch_size)
    self.optimizer_update(self.cri_optimizer, obj_critic, ...)   # 论文公式(5)
    self.soft_update(self.cri_target, self.cri, tau)             # 目标网络软更新

    # ② 更新 Alpha (温度系数): 自动熵调节
    action_pg, log_prob = self.get_action_logprob(rep_states)
    obj_alpha = (alpha_log * (target_entropy - log_prob).detach()).mean()  # 论文公式(7)
    self.optimizer_update(self.alpha_optimizer, obj_alpha, ...)

    # ③ 更新 Actor (策略网络): 最大化 Q - α·logπ
    q_value_pg = self.cri_target(rep_states, action_pg).mean()
    obj_actor = (q_value_pg - log_prob * alpha).mean()           # 论文公式(6)
    self.optimizer_update(self.act_optimizer, -obj_actor, ...)   # 注意负号(梯度上升)

    # ④ 更新 Beta (掩码惩罚): 惩罚给被掩码股票分配权重 ← 论文§4.2"监督损失"
    if if_use_beta:
        beta_loss = self.get_object_beta(rep_states, mask, ids_restore)
        self.optimizer_update(self.beta_optimizer, beta_loss, ...)

    # ⑤ 更新 Rep (MAE表征网络): 最小化重构损失
    if if_use_rep:
        rep_loss = self.update_rep_net(state, mask, ids_restore, ...)  # 论文公式(8)
```

**`get_obj_critic_raw` — 计算 TD 误差（贝尔曼方程的代码实现）** 🎯：
```python
with torch.no_grad():
    # 算 TD 目标: r + γ(1-done)(Q_target(s',a') - α·logπ(a'|s'))
    rep_next_states = rep.forward_state(next_states, mask, ids_restore)
    next_as, next_logprobs = get_action_logprob(rep_next_states)
    next_qs = cri_target.get_q_min(rep_next_states, next_as)    # 双Q取min
    q_labels = rewards + (1-dones)*gamma*(next_qs - next_logprobs*alpha)  # ← 论文公式(5)!
    rep_states = rep.forward_state(states, mask, ids_restore)
# 当前 Q
q1, q2 = self.cri.get_q1_q2(rep_states, actions)
# TD 误差 = (Q - 目标)²,  两个 Q 都要算
obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
```
**这段代码 = 论文公式 (5) = 贝尔曼方程 = 学习清单第 3、6 部分的全部理论。** 反复看这一段，理论和实践就打通了。

**`get_object_beta` — 掩码惩罚（论文第 4.2 节的"监督损失"）**：
```python
# mask_bool: 标记哪些是被掩码的股票(第0位现金不算)
weight = self.forward_action(rep_states)        # Actor 给出的权重
beta_loss = (weight * mask_bool).sum(dim=1).mean()  # 给被掩码股票的权重之和→希望它趋近0
```
这正是论文说的"第一种方法：给 actor 输出额外加监督损失"，论文实验证明它比"在 TD 误差上加惩罚"效果更好。

### 5.4 重加权：动作包装器（`helpers.py`）🎯

对应**论文第 4.3 节、公式 (9)**。`get_action_wrapper` 把 Actor 的原始输出再包一层：
```python
def get_action(x, mask=None, ...):
    pred = func(x)                          # Actor 原始 logits
    if mask is not None:
        pred = pred - mask_bool * 1e6       # 把被掩码股票的 logit 减去巨大值→softmax后≈0
    weight = F.softmax(pred / T, dim=-1)    # ★ 温度 softmax = 论文公式(9) Re(x)
    return weight
```
- **温度 $T$**（配置里 `T=0.01`）：$T$ 越小，softmax 越"尖锐"，权重越集中在少数股票（稀疏化）；$T=1$ 就是普通 softmax。
- **掩码减大常数**：推理时投资者指定子池，池外股票被减去 `1e6`，softmax 后权重几乎为 0——优雅地实现了"忽略池外股票"。

### 5.5 环境：EnvironmentASR（`pm_based_portfolio_ASR.py`）🔧

实现 gym 的 `reset()` 和 `step()` 接口：
- `reset()`：训练模式随机选起始日（数据增强），返回初始状态。
- `step(action)`：
  1. 取出动作权重 `weights`（第 0 位现金）；
  2. 推进一天，算组合收益 `portfolio_ret = Σ (收盘价涨跌幅 × 各股权重)`；
  3. 更新市值（对应论文公式 1）；
  4. **算奖励**：`reward = qs_stats.sharpe(收益序列)` ← **注意：这版用"夏普比率"当奖励**（ASR=Adjusted Sharpe Reward），而**论文原版用"市值变化 $V_t - V_{t-1}$"当奖励**（见 `pm_based_portfolio_value.py`）。这是个重要区别。

### 5.6 数据集 & 子池掩码（`portfolio_management_dataset.py`）🔧

- 读取 `stocks_cut.txt`（全局股票池 GSP 列表）和每只股票的特征 CSV。
- 读取 `aux_stocks_files_cleaned/` 下的 txt 文件，每个文件定义一个**可定制子池 CSP**，并生成对应的 `mask`：
  ```python
  "mask": np.array([0.0 if stock in stocks else 1.0 for stock in self.stocks])
  ```
  在子池内的股票 mask=0，池外的 mask=1。**这就是论文"可定制股票池"在数据层的落地**——验证时每个 aux 子池建一个环境，测试模型对不同子池的适应力。

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
| 训练算法 | 算法1 | `tools/train.py: main(), train_one_episode()` + `agent.update_net()` |
| 推理算法 | 算法2 | `agent/sac/mask_sac.py: validate_net()` |
| 6个金融指标 | 附录A | `metrics/metric_new.py: ARR/SR/CR/SOR/MDD/VOL` |

---

## 7. 本仓库相对论文的改动

学习时务必注意，这份"魔改版"和论文有以下出入（不是 bug，是作者的二次开发）：

1. **奖励函数不同**：当前配置用 `EnvironmentASR`，奖励 = **夏普比率**（且裁剪为非负）；论文原版奖励 = **市值变化 $V_t-V_{t-1}$**（在 `pm_based_portfolio_value.py`）。
2. **数据集换了**：论文用 SP500/DJ30（美股，2007~2022）；这份配置用的是自己处理的数据（`csp_817_data_cut`，日期 2023~2025，790 只股票，67 个特征），看起来像 A 股/自定义数据。
3. **特征维度变了**：论文 F=102（含 OHLC+技术指标+时序）；这份 `num_features=67`，且作者注释里在试 72/64 个特征的不同组合。
4. **GPU/CPU 双配置**：配置文件顶部用 `check_gpu_available()` 自动切换——GPU 用 `AgentMaskSAC`（完整掩码版），CPU 退化成普通 `AgentSAC`（无掩码，省算力）。学习请以 GPU 分支为准（那才是 EarnMore）。
5. **Actor 加了排序加权**：`soft_logits = logits * log(indices+1)`，论文没有，是该版本的小改动。
6. **新增大量工程外围**：`hyperopt.py / scheduler.py / monitor.py / visualizer.py / experiment_manager.py` 都是论文外的工程化工具，用于批量调参和监控，与算法本身无关。
7. **训练里发邮件**：`train.py` 末尾会调 `send_email(...)`（训练完成/出错时通知），是作者自用功能，跑之前可能要改掉或注释。

---

## 8. 外围工程化工具（了解即可）

这些不是论文内容，是为了"高效做大量实验"加的：

| 文件 | 作用 | 典型命令 |
|---|---|---|
| `experiment_manager.py` | 创建/列出/管理实验 | `python experiment_manager.py list-experiments` |
| `hyperopt.py` | 超参搜索(随机/贝叶斯) | `python hyperopt.py random-search --template mask_sac_base --num-samples 2` |
| `scheduler.py` | 并发调度多个实验 | `python scheduler.py run --max-concurrent 4` |
| `monitor.py` | 训练监控、早停、记录指标 | `python monitor.py status --experiment exp_xxx` |
| `visualizer.py` | 批量画图、对比分析 | `python visualizer.py compare-experiments` |

`quickhelp.txt` 里有作者整理的完整命令流程，想跑批量实验时再看。

---

## 9. 如何跑起来

> ⚠️ 这是研究代码，依赖较重（需要 PyTorch + timm + einops + mmengine + qlib + gym + quantstats 等），且数据路径写死在配置里。下面是大致流程，实际跑通可能需要调依赖和路径。

```bash
# 1. 装依赖(参考 requirements.txt / requirement_linux.txt)
pip install -r requirements.txt
# qlib 需单独装: git clone microsoft/qlib && python setup.py install

# 2. 准备数据(把股票CSV放到 configs 里 data_path 指定的目录)
#    需要: features/*.csv, stocks_cut.txt, aux_stocks_files_cleaned/*.txt

# 3. 训练(最直接的方式)
python tools/train.py --config configs/mask_sac_portfolio_management.py

# 4. 或用作者的pipeline(见README和quickhelp.txt)
sh tools/pipeline_mask_sac_dj30_example.sh
```

**先读懂、再跑通**：建议先把核心 6 个文件读明白，跑通可以放到后面（环境配置往往最耗时）。

---

## 10. 学习路径建议

把这份代码当"教材"来啃，推荐顺序：

1. **先建立全局观**：读本文档第 1~4 节 + `02_学习清单`，搞清楚"三大件 + 主循环"。
2. **配合理论攻核心**：按下面顺序读"带详细注释的核心代码"，每读一个文件就回头对一遍论文公式：
   - `times_embed.py`（股票级嵌入，最简单）
   - `mae.py`（掩码自编码器骨架）
   - `mask_time_state.py`（表征网络，论文§4.1）
   - `mask_sac_net.py`（Actor/Critic，论文§4.2）
   - `helpers.py`（重加权，论文§4.3）
   - `pm_based_portfolio_ASR.py`（环境）
   - `mask_sac.py`（智能体，把一切串起来，论文算法1）★最后啃这个
3. **看主循环**：`tools/train.py`，理解一个 episode 是怎么跑的。
4. **动手改**：试着改 `configs` 里的超参（比如温度 `T`、掩码率范围），观察对结果的影响——这是最好的学习方式。

> **检验自己学懂的标志**：你能对着 `update_net()` 函数，逐行说出"这一步在更新哪个网络、对应论文哪个公式、为什么要这么做"。做到这一点，你就真正掌握了 EarnMore。

---

> **下一步**：打开 `带详细注释的核心代码/` 目录，那里有上述核心文件的**逐行中文注释版**，建议和本文档对照着看。
