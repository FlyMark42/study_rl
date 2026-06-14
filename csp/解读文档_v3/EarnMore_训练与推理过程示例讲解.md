# EarnMore 训练与推理过程示例讲解

> 论文：`Reinforcement Learning with Maskable Stock Representation for Portfolio Management in Customizable Stock Pools`  
> 本文承接 `EarnMore_核心概念原理实验示例讲解.md`，重点解释 EarnMore 如何训练，以及测试/推理时如何输出投资组合。

## 1. 一句话理解

**训练阶段**：模型在全局股票池 GSP 上反复模拟各种用户自定义股票池 CSP，学习“哪些股票能买、哪些股票被 mask、资金该怎么分配”。

**推理/测试阶段**：用户给定一个真实 CSP，模型不再更新参数，直接根据当前市场状态输出投资组合权重。

也就是说：

```text
训练：学本领
推理/测试：用本领
```

## 2. 具体例子

假设全局股票池 GSP 有 5 只股票：

```text
AAPL、MSFT、JPM、GE、GS
```

再加上现金，模型最终要分配的是：

```text
现金、AAPL、MSFT、JPM、GE、GS
```

某个用户今天只想投：

```text
AAPL、MSFT、GS
```

那么：

```text
可交易股票：AAPL、MSFT、GS
被 mask 股票：JPM、GE
```

输入给 EarnMore 时，不是删除 `JPM` 和 `GE`，而是变成：

```text
AAPL、MSFT、[MASK]、[MASK]、GS
```

`[MASK]` 的意思是：这里原本有股票，但当前用户不允许买。

## 3. 训练阶段：EarnMore 怎么学习

训练的核心是：**随机制造很多 CSP，让模型提前见过各种用户偏好。**

### 3.1 读取市场状态

在第 `t` 天，模型读取每只股票最近若干天的数据。

例如论文中使用最近 10 天：

```text
AAPL 最近 10 天价格、技术指标、时间信息
MSFT 最近 10 天价格、技术指标、时间信息
JPM  最近 10 天价格、技术指标、时间信息
GE   最近 10 天价格、技术指标、时间信息
GS   最近 10 天价格、技术指标、时间信息
```

这些信息构成强化学习里的 **状态 state**。

在投资场景中，状态可以理解为：

```text
当前模型能看到的市场信息
```

### 3.2 随机 mask 一部分股票

训练时，EarnMore 会随机遮住一部分股票，模拟用户自定义股票池。

例如本轮训练随机 mask：

```text
JPM、GE
```

得到：

```text
AAPL、MSFT、[MASK]、[MASK]、GS
```

下一轮训练可能 mask：

```text
AAPL、GS
```

得到：

```text
[MASK]、MSFT、JPM、GE、[MASK]
```

模型训练很多轮后，就会见过大量不同 CSP。

这一步解决的问题是：

```text
未来用户怎么改股票池，模型都不至于完全陌生。
```

### 3.3 生成股票级表示

模型先把每只股票最近 10 天的数据压缩成一个向量，也就是 stock-level embedding。

可以理解为：

```text
AAPL 最近 10 天的一堆数据 -> AAPL 的浓缩特征向量
MSFT 最近 10 天的一堆数据 -> MSFT 的浓缩特征向量
```

这个向量不是人直接看的，而是神经网络用来判断的。

### 3.4 遮住再重建

EarnMore 使用自监督学习思想：先遮住一部分股票，再让模型尝试重建被遮住股票的信息。

例子：

```text
模型看到：
AAPL、MSFT、[MASK]、[MASK]、GS

模型尝试重建：
JPM、GE 的价格特征
```

这一步的目的不是为了真的预测 `JPM` 和 `GE`，而是让模型学习股票之间的关系。

例如模型可能学到：

```text
AAPL 和 MSFT 都是科技股，走势可能有关联
JPM 和 GS 都是金融股，可能受利率影响较大
某些股票虽然行业不同，但在市场恐慌时会一起下跌
```

通俗理解：

```text
遮住再重建 = 让模型学会股票池内部的结构关系
```

### 3.5 Actor 输出投资组合

EarnMore 使用 SAC，即 Soft Actor-Critic。

里面有两个关键角色：

| 组件 | 作用 | 类比 |
|---|---|---|
| Actor | 输出资金分配 | 基金经理 |
| Critic | 评价这次分配好不好 | 投资评审员 |

假设 Actor 初始输出：

```text
现金 10%
AAPL 30%
MSFT 25%
JPM 5%
GE 5%
GS 25%
```

但当前 `JPM` 和 `GE` 被 mask，用户不允许买。

因此这个动作是不理想的，训练时会被惩罚。

模型希望学成：

```text
现金 10%
AAPL 35%
MSFT 30%
JPM 0%
GE 0%
GS 25%
```

### 3.6 Re-weighting 重新加权

Actor 有时会输出一些很小但无意义的仓位。

例如：

```text
现金 10%
AAPL 31%
MSFT 29%
JPM 1%
GE 1%
GS 28%
```

这里 `JPM` 和 `GE` 的 1% 很可能是噪声。

EarnMore 使用带温度参数 `T` 的 softmax 做 re-weighting：

```text
T 小：权重更集中
T 大：权重更平均
```

论文中最佳温度参数为：

```text
T = 0.1
```

调整后可能变成：

```text
现金 10%
AAPL 34%
MSFT 31%
JPM 0%
GE 0%
GS 25%
```

这一步的作用是减少无意义小仓位，让组合更集中。

### 3.7 与市场环境交互，得到奖励

模型按当前权重投资一天。

假设今天组合价值是：

```text
10000 元
```

第二天股票价格变化后，组合价值变成：

```text
10150 元
```

则奖励可以直观理解为：

```text
reward = 10150 - 10000 = 150
```

如果第二天组合价值变成：

```text
9900 元
```

则奖励是：

```text
reward = 9900 - 10000 = -100
```

模型会通过大量这样的交易步骤学习：

```text
什么市场状态下，应该买哪些股票，买多少比例，长期更容易赚钱。
```

### 3.8 更新模型参数

训练时会更新几类参数：

| 参数模块 | 学什么 |
|---|---|
| 表示网络 | 如何理解带 mask 的股票池 |
| Actor | 如何输出更好的投资权重 |
| Critic | 如何评价某个投资动作的长期价值 |
| alpha | SAC 中探索程度的调节参数 |

训练主流程可以简化为：

```text
读取市场数据
-> 随机生成 CSP mask
-> 得到 maskable stock representation
-> Actor 输出组合权重
-> re-weighting 调整权重
-> 环境计算收益 reward
-> Critic 评价动作好坏
-> 更新 Actor、Critic、表示网络
-> 重复很多轮
```

## 4. 推理/测试阶段：EarnMore 怎么使用

推理阶段和训练阶段最大的区别是：

```text
推理阶段不再更新模型参数。
```

模型只是根据当前输入，直接输出投资组合。

### 4.1 用户给定真实 CSP

假设用户明确说：

```text
我要投 AAPL、MSFT、GS
不要 JPM、GE
```

那么当前 CSP 是：

```text
CSP = {AAPL, MSFT, GS}
```

mask 情况是：

```text
AAPL：可交易
MSFT：可交易
JPM ：mask
GE  ：mask
GS  ：可交易
```

### 4.2 输入最近市场数据

模型读取当前时点最近 10 天的市场数据：

```text
AAPL、MSFT、JPM、GE、GS 最近 10 天数据
```

然后根据用户 CSP 做 mask：

```text
AAPL、MSFT、[MASK]、[MASK]、GS
```

### 4.3 生成当前股票池表示

训练好的表示网络生成：

```text
当前 CSP 的 maskable stock representation
```

这可以理解为模型内部的一份“当前股票池说明书”：

```text
哪些股票可买
哪些股票不可买
当前市场状态如何
股票之间大致有什么关系
```

### 4.4 Actor 输出投资权重

训练好的 Actor 直接输出资金分配。

例如：

```text
现金 12%
AAPL 38%
MSFT 27%
JPM 0%
GE 0%
GS 23%
```

这就是 EarnMore 在当前状态下给出的投资组合。

注意，推理时输出的不是：

```text
AAPL 明天涨 2%
```

而是：

```text
当前应该给 AAPL 分配 38% 资金。
```

所以 EarnMore 更像组合配置模型，而不是单纯股价预测模型。

## 5. 测试/回测阶段怎么跑

测试时通常会用历史数据模拟真实交易。

例如从 2021-01-07 到 2022-06-26 回测。

每天重复：

```text
第 t 天：
读取最近 10 天数据
-> 根据用户 CSP mask 股票
-> Actor 输出今日投资权重
-> 用第 t+1 天真实价格计算组合收益
-> 更新组合价值
-> 进入下一天
```

假设测试开始组合价值为：

```text
10000 元
```

第一天后：

```text
10150 元
```

第二天后：

```text
10080 元
```

第三天后：

```text
10300 元
```

最后得到一条组合价值曲线。

论文再根据这条曲线计算：

| 指标 | 说明 |
|---|---|
| ARR | 年化收益率 |
| SR | 夏普比率 |
| CR | Calmar Ratio |
| SoR | Sortino Ratio |
| MDD | 最大回撤 |
| VOL | 波动率 |

## 6. 训练和推理的区别

| 对比项 | 训练阶段 | 推理/测试阶段 |
|---|---|---|
| 是否随机 mask | 是，用来模拟各种 CSP | 否，使用用户指定 CSP |
| 是否更新参数 | 是 | 否 |
| 是否计算损失 | 是 | 通常不计算训练损失 |
| 是否与环境交互 | 是 | 测试时会用历史价格回测 |
| 主要目标 | 学会处理各种股票池 | 输出当前投资组合 |
| 输出结果 | 更新后的模型参数 | 现金和股票权重 |

## 7. 训练和推理的伪代码

### 7.1 训练伪代码

```text
输入：全局股票池 GSP、历史市场数据
初始化：表示网络、Actor、Critic、Replay Buffer

for 每一轮训练:
    读取当前市场状态 state
    随机生成 mask，模拟一个 CSP
    用 [MASK] 替换被排除股票
    生成 maskable stock representation
    Actor 输出投资权重
    re-weighting 调整权重
    环境根据下一天价格计算 reward
    把 state、action、reward、next_state 存入 Replay Buffer
    从 Replay Buffer 抽样
    更新 Critic
    更新 Actor
    更新表示网络

输出：训练好的 EarnMore
```

### 7.2 推理/测试伪代码

```text
输入：训练好的 EarnMore、用户指定 CSP、当前市场数据

for 每个交易日:
    读取最近 10 天市场状态 state
    根据用户 CSP 生成 mask
    用 [MASK] 替换不可交易股票
    生成 maskable stock representation
    Actor 输出投资权重
    re-weighting 调整权重
    得到今日投资组合

如果是回测:
    用下一天真实价格计算组合收益
    记录组合价值

输出：每日投资组合权重、累计收益曲线、金融评价指标
```

## 8. 最容易混淆的点

### 8.1 EarnMore 的“预测”不是预测股价

它不是主要回答：

```text
AAPL 明天涨还是跌？
```

它回答的是：

```text
在当前市场状态和用户股票池限制下，资金应该怎么分配？
```

即：

```text
输入：市场状态 + 用户 CSP
输出：现金和各股票的权重
```

### 8.2 mask 不是删除股票

删除股票会让输入维度变化，模型结构难以统一。

mask 是保留位置，但告诉模型：

```text
这只股票当前不可买。
```

所以：

```text
删除 = 股票从输入中消失
mask = 股票位置还在，但被标记为不可交易
```

### 8.3 训练时随机 mask，测试时按用户真实需求 mask

训练时随机 mask 是为了让模型提前练习各种情况。

测试时不随机，而是用户真实指定：

```text
今天哪些股票可以买，哪些股票不能买。
```

## 9. 最简总结

EarnMore 的训练过程可以理解为：

```text
让模型在历史市场中反复练习：
如果用户临时排除一些股票，我该如何重新分配资金？
```

EarnMore 的推理过程可以理解为：

```text
用户给出当前可交易股票池，模型直接输出现金和各股票的投资比例。
```

用前面的例子：

```text
GSP：
AAPL、MSFT、JPM、GE、GS

用户 CSP：
AAPL、MSFT、GS

模型输入：
AAPL、MSFT、[MASK]、[MASK]、GS

模型输出：
现金 12%、AAPL 38%、MSFT 27%、JPM 0%、GE 0%、GS 23%
```

这就是 EarnMore 从训练到推理的完整逻辑。
