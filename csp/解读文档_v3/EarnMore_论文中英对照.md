# 基于可掩码股票表征的强化学习：面向可定制股票池的投资组合管理（中英对照译文）

> **Reinforcement Learning with Maskable Stock Representation for Portfolio Management in Customizable Stock Pools**
>
> - **发表会议**：ACM Web Conference 2024（WWW'24），2024 年 5 月 13–17 日，新加坡
> - **作者**：Wentao Zhang, Yilei Zhao, Shuo Sun*, Jie Ying, Yonggang Xie, Zitao Song, Xinrun Wang, Bo An（南洋理工大学、浙江大学、Skywork AI）
> - **arXiv**: 2311.10801v4 [q-fin.PM]　**DOI**: 10.1145/3589334.3645615
> - **代码**：https://github.com/DVampire/EarnMore （即本目录下 `EarnMore-main`）
> - **许可**：本论文采用 CC-BY 4.0 国际许可证发布，允许翻译与再分发（需署名）。
>
> 体例说明：每段先列英文原文（引用块 🇬🇧），后接中文译文（💬）。公式按原文编号用 LaTeX 重排（PDF 提取的公式有乱序，已人工复原）。译者注以「📌 译注」标出，供初学者参考。

---

## 摘要（Abstract）

> 🇬🇧 Portfolio management (PM) is a fundamental financial trading task, which explores the optimal periodical reallocation of capitals into different stocks to pursue long-term profits. Reinforcement learning (RL) has recently shown its potential to train profitable agents for PM through interacting with financial markets. However, existing work mostly focuses on fixed stock pools, which is inconsistent with investors' practical demand. Specifically, the target stock pool of different investors varies dramatically due to their discrepancy on market states and individual investors may temporally adjust stocks they desire to trade (e.g., adding one popular stocks), which lead to customizable stock pools (CSPs). Existing RL methods require to retrain RL agents even with a tiny change of the stock pool, which leads to high computational cost and unstable performance.

💬 投资组合管理（Portfolio Management, PM）是一项基础性的金融交易任务，它研究如何周期性地把资金以最优方式重新分配到不同股票上，以追求长期收益。强化学习（Reinforcement Learning, RL）近来已展现出潜力：通过与金融市场交互，可以训练出能盈利的 PM 智能体。然而，现有工作大多聚焦于**固定股票池**，这与投资者的实际需求并不一致。具体来说，不同投资者由于对市场状态的判断不同，其目标股票池差异极大；而且单个投资者也可能临时调整想交易的股票（例如加入一只热门股票），这就形成了**可定制股票池**（Customizable Stock Pools, CSPs）。现有 RL 方法即便股票池只发生微小变化也必须重新训练智能体，导致计算开销高昂、性能不稳定。

> 🇬🇧 To tackle this challenge, we propose EarnMore, a rEinforcement leARNing framework with Maskable stOck REpresentation to handle PM with CSPs through one-shot training in a global stock pool (GSP). Specifically, we first introduce a mechanism to mask out the representation of the stocks outside the target pool. Second, we learn meaningful stock representations through a self-supervised masking and reconstruction process. Third, a re-weighting mechanism is designed to make the portfolio concentrate on favorable stocks and neglect the stocks outside the target pool. Through extensive experiments on 8 subset stock pools of the US stock market, we demonstrate that EarnMore significantly outperforms 14 state-of-the-art baselines in terms of 6 popular financial metrics with over 40% improvement on profit. Code is available in PyTorch.

💬 为了应对这一挑战，我们提出 **EarnMore**——一个带有**可掩码股票表征**（Maskable Stock Representation）的强化学习框架，只需在**全局股票池**（Global Stock Pool, GSP）上进行**一次性训练**，即可处理各种 CSP 下的投资组合管理。具体而言：第一，我们引入一种机制，把目标池之外股票的表征**掩码**（mask）掉；第二，我们通过自监督的「掩码—重建」过程学习有意义的股票表征；第三，我们设计了一种**重加权**（re-weighting）机制，使投资组合集中于看好的股票、忽略目标池之外的股票。在美股市场 8 个子股票池上的大量实验表明，EarnMore 在 6 个常用金融指标上显著优于 14 个最先进的基线方法，收益提升超过 40%。代码以 PyTorch 实现并已开源。

📌 译注：「掩码」（mask）借鉴自 BERT/MAE 等自监督模型——把一部分输入遮住，用一个可学习的占位向量代替。本文用它表示「投资者不想要的股票」，从而让同一个智能体适配任意子股票池。

**CCS 概念**：信息系统→数据挖掘；计算方法→机器学习；应用计算→电子商务。
**关键词**：投资组合管理（Portfolio Management）、强化学习（Reinforcement Learning）、表征学习（Representation Learning）

---

## 1 引言（INTRODUCTION）

> 🇬🇧 The stock market, which involves over $90 trillion market capitalization, has attracted the attention of innumerable investors around the world. Portfolio management, which dynamically allocates the proportion of capitals among different stocks, plays a key role to make profits for investors. Reinforcement learning (RL) has recently become a promising methodology for financial trading tasks due to its stellar performance on solving complex sequential decision-making problems such as Go [25] and matrix multiplication [9]. In fact, RL has achieved significant success in various quantitative trading tasks such as algorithmic trading [4], portfolio management [31], order execution [8] and market making [26]. To apply RL methods for PM, existing work [31, 32, 35] always train RL agents to make investment decisions based on a fixed stock pool, which requires retraining with high computational cost when investors need to change their target stock pools. Furthermore, investors are unable to engage in or influence the agent's decision-making process, as it is uncontrollable for them.

💬 股票市场总市值超过 90 万亿美元，吸引了全球无数投资者的关注。投资组合管理通过在不同股票之间动态分配资金比例，是投资者获利的关键。强化学习因其在围棋 [25]、矩阵乘法 [9] 等复杂序贯决策问题上的卓越表现，近来已成为金融交易任务中一种很有前景的方法论。事实上，RL 已在多种量化交易任务中取得显著成功，例如算法交易 [4]、投资组合管理 [31]、订单执行 [8] 和做市 [26]。在将 RL 应用于 PM 时，现有工作 [31, 32, 35] 总是基于一个**固定的股票池**训练智能体做投资决策；当投资者需要更换目标股票池时，就必须以高昂的计算代价重新训练。此外，投资者无法参与或影响智能体的决策过程——这个过程对他们而言是不可控的。

📌 译注：原文脚注 2 指出——现有工作中的「固定股票池」就是本文设定下的「全局股票池（GSP）」。

> 🇬🇧 Investors and agents may work better with each other in collaboration. On one hand, investors have a limited understanding of stock trends, hidden connections among stocks, and the overall market dynamics. On the other hand, the RL agent may make occasional decision errors and lacks the human capacity to acquire information in diverse ways. All of these factors can result in diminished or suboptimal returns. For instance, if a certain stock is delisted in the trading process, the agent may still consider it as a candidate for investment allocation, potentially leading to a loss of returns. In a different scenario, when a stock exhibits significant future profit potential, investors may want to add it to the target pools to maximize their returns. Therefore, we introduce the task of portfolio management with customizable stock pools (CSPs) to meet the mentioned demands and address the above issues.

💬 投资者与智能体若能协作，效果可能更好。一方面，投资者对股票趋势、股票间的隐藏关联以及整体市场动态的理解有限；另一方面，RL 智能体偶尔会出现决策失误，而且不具备人类通过多种渠道获取信息的能力。这些因素都会导致收益减少或次优。举例来说，如果某只股票在交易过程中退市，智能体可能仍把它当作备选的投资标的，从而造成收益损失；在另一种场景中，当某只股票展现出可观的未来盈利潜力时，投资者会希望把它加入目标池以最大化收益。因此，我们提出了**带可定制股票池（CSPs）的投资组合管理**这一任务，以满足上述需求并解决上述问题。

> 🇬🇧 As shown in Fig. 1, the demand of PM with customizable stock pools (CSPs) is ubiquitous for different financial practitioners in real-world trading scenarios. For instance, stock brokerages need to offer real-time portfolio suggestions for millions of investors with diversified preferences on stock pools. The investors also desire to adapt their stock pools based on different market conditions from time to time. Therefore, an RL algorithm with the ability to handle PM with CSPs is urgently needed.

💬 如图 1 所示，在真实交易场景中，各类金融从业者对「可定制股票池下的 PM」的需求无处不在。例如，券商需要为数以百万计、对股票池偏好各异的投资者提供实时的组合建议；投资者自己也希望随市场环境变化不时调整股票池。因此，业界亟需一种能处理 CSP 下 PM 问题的 RL 算法。

> 🇬🇧 There are 3 straightforward methods for implementing PM in CSPs: i) training an agent from scratch on each individual CSP. However, randomly picking 5 stocks from 30 for a CSP leads to 140k+ combinations, which is unfeasible in practice; ii) adjusting the output dimensions of the policy network and then fine-tune the agents by mapping the action space of the GSP to CSP. Fine-tuning agent on PM with contiguous action space might be equally time-consuming as starting from scratch, making it impractical in reality; iii) using action-masking method by subtracting a large constant from policy network logits that represent unfavorable stocks to reduce their investment allocation. Human-induced decision changes in this method do not truly express the agent's real decision-making, as it lacks awareness of CSP stocks, significantly impacting effectiveness. Although the existing work considers the dynamics of the stock pool, such as the study by Betancourt et al. [2] focuses on the changing number of assets, training an actor and critic for each stock is resource-intensive and time-consuming. Additionally, this approach disallows for changes in the stock pool in terms of number of stocks and composition during the investment process, which would suffer the same limitations if implemented by action-masking.

💬 要在 CSP 上实现 PM，有 3 种直接的做法：
i) **对每个 CSP 从零训练一个智能体**。然而仅从 30 只股票中随机挑 5 只组成 CSP，就有 14 万种以上的组合，实践中根本不可行；
ii) **调整策略网络的输出维度，再把 GSP 的动作空间映射到 CSP 上微调**（fine-tune）。在连续动作空间的 PM 中微调智能体可能与从零训练同样耗时，因而在现实中也不切实际；
iii) **动作掩码**（action-masking）：从代表「不看好股票」的策略网络 logits 中减去一个大常数，以降低对它们的投资分配。这种由人为干预造成的决策改变并不能真正表达智能体的实际决策——因为智能体本身并不「知道」CSP 里有哪些股票，效果会大打折扣。
虽然已有工作考虑了股票池的动态性，例如 Betancourt 等 [2] 研究了资产数量的变化，但为每只股票单独训练一套 actor 和 critic 既耗资源又耗时，而且这种方法不允许在投资过程中改变股票池的数量与构成——若用动作掩码实现，也会遭遇同样的局限。

> 🇬🇧 To handle PM with CSPs, we face the following two major challenges: i) how to learn unified representations that are aligned for stock pools with different sizes and stocks; ii) how to guide RL agents to construct portfolios that concentrate on most favorable stocks and neglect the stocks outside the target pool. To tackle these two challenges, we have designed an RL framework that called EarnMore with one-time training that takes into account the distinct investment preferences of each investor and invests in various CSPs. This framework allows for dynamic adjustments to the pool in the investment process, contributing to a more tailored and effective portfolio. Our contributions are four-fold:

💬 处理 CSP 下的 PM，我们面临两大挑战：i) 如何为大小不同、成分不同的股票池学到**对齐的统一表征**；ii) 如何引导 RL 智能体构建的组合**集中于最看好的股票、忽略目标池之外的股票**。为了解决这两个挑战，我们设计了名为 EarnMore 的 RL 框架：只训练一次，就能兼顾每位投资者各自的投资偏好并投资于各种 CSP。该框架允许在投资过程中动态调整股票池，从而得到更贴合个人需求、更有效的组合。我们的贡献有四点：

> 🇬🇧 • We introduce a learnable masked token to represent unfavorable stocks, which enables the unified representation of stock pools with different sizes and stocks.
> • We derive meaningful embeddings using a self-supervised masking and reconstruction process that captures stock relationships.
> • We propose a re-weighting mechanism to rescale the distribution of portfolios to make it concentrate on favorable stocks and neglect stocks outside the target pool.
> • Experiments on 8 subset stock pools of the US stock market demonstrate the superiority of EarnMore over 14 baselines in terms of 6 popular financial metrics with one-time training.

💬
- 我们引入一个**可学习的掩码 token**（masked token）来表示不看好的股票，使大小、成分各异的股票池能够获得统一表征。
- 我们通过自监督的**掩码—重建**过程获得有意义的嵌入（embedding），以捕捉股票之间的关系。
- 我们提出一种**重加权机制**，对组合的分布进行重新缩放，使其集中于看好的股票、忽略目标池之外的股票。
- 在美股市场 8 个子股票池上的实验表明：仅一次训练，EarnMore 在 6 个常用金融指标上全面超越 14 个基线方法。

**图 1**：EarnMore 在可定制股票池（CSPs）中进行投资组合管理的总览。（左：全局股票池；中：投资者按偏好定制出 CSP1/CSP2/CSP3；右：EarnMore 对每个 CSP 输出含现金在内的投资比例向量。）

---

## 2 相关工作（RELATED WORK）

### 2.1 投资组合管理（Portfolio Management）

> 🇬🇧 Portfolio management is an essential aspect of investment and involves a strategic allocation of resources to achieve optimal returns and avoid risks simultaneously. There are two commonly used traditional rule-based methods, i.e., mean reversion [22] and momentum [13]. The former buys low-priced stocks and sells high-priced ones, whereas the latter relies on recent performance with the expectation that trends will continue. Cross-Sectional Momentum [14] and Time-Series Momentum [20] are two classical momentum trading methods. However, traditional rule-based methods are difficult to capture fleeting patterns in changing market conditions and perform well only in specific scenarios [4].

💬 投资组合管理是投资中的核心环节，其目标是通过对资源的战略性配置，在追求最优收益的同时规避风险。传统的基于规则的方法主要有两类：**均值回归**（mean reversion, [22]）与**动量**（momentum, [13]）。前者买入低价股、卖出高价股；后者依赖近期表现，期望趋势会延续。**横截面动量**（Cross-Sectional Momentum, [14]）与**时间序列动量**（Time-Series Momentum, [20]）是两种经典的动量交易方法。然而，传统规则方法难以捕捉多变市场中转瞬即逝的模式，只在特定场景下表现良好 [4]。

📌 译注：均值回归 = 押注价格会「回到平均」（超跌买入、超涨卖出）；动量 = 押注「强者恒强」（追涨）。二者哲学相反，是量化交易的两大经典流派。

> 🇬🇧 In the past few years, advanced prediction-based methods have significantly surpassed traditional rule-based methods in performance. These methods treat PM as a supervised learning task and predict future returns (regression) or price movements (classification). Then, the heuristic strategy generator allocates asset investments based on the prediction results [36]. Specifically, prediction-based methods can be categorized into two kinds, machine learning models like XGBoost [3] and LightGBM [16], along with deep learning models such as ALSTM [23] and TCN [1]. However, the volatility and noisy nature of the financial market makes it extremely difficult to accurately predict future prices [7]. Furthermore, the gap between prediction signals and profitable trading actions [10] is difficult to bridge. Therefore, prediction-based methods do not perform satisfactorily in general.

💬 过去几年，先进的**基于预测**（prediction-based）的方法在性能上大幅超越了传统规则方法。这类方法把 PM 当作监督学习任务，预测未来收益（回归）或价格涨跌（分类），然后由启发式的策略生成器根据预测结果分配资产投资 [36]。基于预测的方法又可分两类：机器学习模型（如 XGBoost [3]、LightGBM [16]）和深度学习模型（如 ALSTM [23]、TCN [1]）。然而，金融市场的高波动与高噪声本质使得准确预测未来价格极为困难 [7]；而且**预测信号与可盈利交易动作之间的鸿沟** [10] 也难以弥合。因此，基于预测的方法总体表现并不令人满意。

📌 译注：脚注 3——例如 top-k [36] 策略会挑出预测收益最高的 k 只资产，并按预测收益的排名大小决定投资比例。

> 🇬🇧 Recent years have witnessed the successful marriage of reinforcement learning and portfolio management [28] due to its ability to handle sequential decision-making problems. EIIE [15] utilizes the convolutional neural network for feature extraction and RL for decision-making. The Investor-Imitator [5] framework demonstrates its utility in financial investment by emulating investor actions. SARL [35] leverages the price movement prediction as additional states. DeepTrader [32] dynamically balances risk-return with market indicators and utilizes a unique graph structure to generate portfolios. HRPM [31] presents a hierarchical framework addressing long-term profits and considers price slippage as part of the trading cost. DeepScalper [29] combines intraday trading and risk-aware tasks to capture the investment opportunities.

💬 近年来，得益于 RL 处理序贯决策问题的能力，强化学习与投资组合管理的「联姻」取得了成功 [28]。**EIIE** [15] 用卷积神经网络提取特征、用 RL 做决策；**Investor-Imitator** [5] 框架通过模仿投资者动作展现其在金融投资中的效用；**SARL** [35] 把价格涨跌预测作为额外的状态加以利用；**DeepTrader** [32] 结合市场指标动态平衡风险与收益，并用一种独特的图结构生成组合；**HRPM** [31] 提出一个分层框架，关注长期收益并把价格滑点（slippage）计入交易成本；**DeepScalper** [29] 则结合日内交易与风险感知任务来捕捉投资机会。

📌 译注：脚注 4——HRPM 和 DeepScalper 因引入了「限价订单簿（order book）」这类 K 线价格之外的额外数据，未纳入本文的实验对比。

### 2.2 掩码自编码器（Masked Autoencoders）

> 🇬🇧 Masked Autoencoders (MAEs) [12] are neural networks employed in self-supervised learning to obtain effective embeddings. An autoencoder is designed to learn a representation for a set of data, which is initially used for self-supervised learning in image, video and audio. Recently, they have been extensively studied in the field of time series prediction. PatchTST [21] enhances multivariate time series forecasting through self-self-supervised learning. It achieves this by partitioning time series data into patches and utilizing separate channels for univariate time series. This approach boosts memory efficiency and improves the model's ability to capture historical patterns. Several notable works have used MAEs to learn time series representations, such as SimMTM [6] and Ti-MAE [18].

💬 **掩码自编码器**（Masked Autoencoders, MAEs, [12]）是用于自监督学习、以获得有效嵌入的神经网络。自编码器旨在为一组数据学习一种表征，最初被用于图像、视频和音频的自监督学习。近来，它在时间序列预测领域被广泛研究。**PatchTST** [21] 通过自监督学习增强多变量时间序列预测：它把时间序列切分成「补丁」（patch），并为每条单变量序列使用独立通道，从而提升内存效率、改善模型捕捉历史模式的能力。也有若干代表性工作用 MAE 来学习时间序列表征，例如 SimMTM [6] 和 Ti-MAE [18]。

📌 译注：MAE 的核心思想——随机遮住一部分输入（如图像块），让模型仅凭可见部分去「重建」被遮住的部分。重建任务迫使模型学到数据的内在结构，无需人工标注。本文把这套思想搬到股票特征上。

> 🇬🇧 In financial markets, which tend to have lower signal-to-noise ratios than typical time series data. Leveraging MAEs for self-supervised learning, we efficiently reduce data dimensionality, filter noise, and highlight essential information. MAEs uncover hidden stock relationships and represent investor-unfavorable stocks with masked tokens, improving investor-agent interactions in PM.

💬 金融市场的信噪比往往比一般时间序列数据更低。借助 MAE 做自监督学习，我们能高效地降低数据维度、过滤噪声、突出关键信息。MAE 可以揭示股票之间的隐藏关系，并用掩码 token 表示投资者不看好的股票，从而改善 PM 中投资者与智能体之间的交互。

---

## 3 预备知识（PRELIMINARIES）

> 🇬🇧 In this section, we present definitions and formulas for necessary terms in PM. Next, we provide a Markov Decision Process (MDP) model for portfolio management with CSPs.

💬 本节先给出 PM 中若干必要术语的定义与公式，再给出可定制股票池（CSPs）下投资组合管理的**马尔可夫决策过程**（MDP）模型。

### 3.1 定义与公式（Definitions and Formulas）

> 🇬🇧 **Definition 1. (OHLCV).** Open-High-Low-Close-Volume is a type of bar chart obtained from the financial market. The OHLCV of stock $i$ at time $t$ is denoted as $P_{i,t}=[p^o_{i,t}, p^h_{i,t}, p^l_{i,t}, p^c_{i,t}, v_{i,t}]$, where $p^o_{i,t}, p^h_{i,t}, p^l_{i,t}, p^c_{i,t}$ and $v_{i,t}$ are open, high, low, close prices and volume.

💬 **定义 1（OHLCV）**。开盘—最高—最低—收盘—成交量（Open-High-Low-Close-Volume）是从金融市场获得的一种 K 线（bar chart）数据。股票 $i$ 在时刻 $t$ 的 OHLCV 记为 $P_{i,t}=[p^o_{i,t}, p^h_{i,t}, p^l_{i,t}, p^c_{i,t}, v_{i,t}]$，其中 $p^o, p^h, p^l, p^c$ 分别是开盘价、最高价、最低价、收盘价，$v_{i,t}$ 是成交量。

> 🇬🇧 **Definition 2. (Technical indicators).** A technical indicator indicates a feature calculated by a formulaic combination of the historical OHLCV. We denote the technical indicator vector at time $t$ as $Y_t=[y^1_t, y^2_t, \dots, y^K_t]^T \in \mathbb{R}^K$. For $k\in[1,K]$, $y^k_t$ is represented as $y^k_t = f^k_t(x_{t-D+1}, \dots, x_{t-1}, x_t \mid \theta^k)$, where $D$ denotes past time steps up to $t$, and $\theta^k$ are hyperparameters for indicator $k$.

💬 **定义 2（技术指标）**。技术指标（technical indicator）是由历史 OHLCV 经过某种公式化组合计算出的特征。记时刻 $t$ 的技术指标向量为 $Y_t=[y^1_t, y^2_t, \dots, y^K_t]^T \in \mathbb{R}^K$。对 $k\in[1,K]$，第 $k$ 个指标 $y^k_t = f^k_t(x_{t-D+1}, \dots, x_{t-1}, x_t \mid \theta^k)$，其中 $D$ 表示截至 $t$ 的过去时间步数，$\theta^k$ 是指标 $k$ 的超参数。

📌 译注：技术指标如 MA（移动平均）、MACD、RSI 等，都是把历史价格/成交量按固定公式加工出的衍生特征。

> 🇬🇧 **Definition 3. (Portfolio).** A portfolio is a combination of financial assets, denoted as $W_t=[w^0_t, w^1_t, \dots, w^N_t]\in\mathbb{R}^{N+1}$, with $N+1$ assets, including 1 risk-free cash and $N$ risky stocks. Each asset $i$ is assigned a weight $w^i_t$ representing its portfolio proportion, subject to the constraint that $\sum_{i=0}^{N} w^i_t = 1$ for full investment.

💬 **定义 3（投资组合）**。投资组合是金融资产的组合，记为 $W_t=[w^0_t, w^1_t, \dots, w^N_t]\in\mathbb{R}^{N+1}$，共 $N+1$ 个资产，包括 1 份无风险**现金**和 $N$ 只风险股票。每个资产 $i$ 被赋予权重 $w^i_t$ 表示其在组合中的占比，并满足**满仓投资**约束 $\sum_{i=0}^{N} w^i_t = 1$。

> 🇬🇧 **Definition 4. (Portfolio Value).** Portfolio value at time step $t$, denoted as $V_t$, represents the sum of individual asset values in the portfolio, $V_0$ represents the initial cash and $V_t$ calculated using stock closing prices through the following formula:

💬 **定义 4（组合价值）**。时刻 $t$ 的组合价值记为 $V_t$，表示组合中各资产价值之和；$V_0$ 为初始现金。$V_t$ 用股票收盘价按下式计算：

$$V_t = w^0_t V_{t-1} + (1-w^0_t)V_{t-1}\left(1 + \sum_{i=1}^{N} w^i_t \frac{p^c_{i,t}-p^c_{i,t-1}}{p^c_{i,t-1}}\right) \tag{1}$$

📌 译注：式 (1) 含义——现金部分 $w^0_t V_{t-1}$ 价值不变；其余资金按各股票当期收益率 $\frac{p^c_{i,t}-p^c_{i,t-1}}{p^c_{i,t-1}}$ 加权增值。

### 3.2 问题形式化（Problem Formulation）

> 🇬🇧 We model portfolio management as a Markov Decision Process (MDP) and provide a detailed description of the MDP modeling process for portfolio management with CSPs in this section.
>
> **MDP Formulation for PM.** We formulate PM as an MDP following a standard RL scenario, where an agent (investor) interacts with an environment (the financial markets) in discrete time to make actions (investment decisions) and get rewards (profits). In this work, the objective is to maximize the final portfolio value within a long-term investment time horizon. We formulate PM as an MDP, which is constructed by a 5-tuple $(\mathcal{S}, \mathcal{A}, T, R, \gamma)$. Specifically, $\mathcal{S}$ is a finite set of states. $\mathcal{A}$ is a finite set of actions. The state transition function $T:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\to[0,1]$ encapsulates transition probabilities between states based on chosen actions. The reward function $R:\mathcal{S}\times\mathcal{A}\to\mathbb{R}$ quantifies the immediate reward of taking an action in a state. The discount factor is $\gamma\in[0,1)$. A policy $\pi:\mathcal{S}\times\mathcal{A}\to[0,1]$ assigns each state $s\in\mathcal{S}$ a distribution over actions, where $a\in\mathcal{A}$ has probability $\pi(a|s)$.

💬 我们把投资组合管理建模为**马尔可夫决策过程**（MDP），本节详细描述 CSP 下 PM 的 MDP 建模过程。

**PM 的 MDP 形式化**。我们按标准 RL 场景把 PM 表示为 MDP：一个**智能体**（投资者）在离散时间上与**环境**（金融市场）交互，做出**动作**（投资决策）并获得**奖励**（收益）。本文目标是在长期投资时段内**最大化最终组合价值**。MDP 由五元组 $(\mathcal{S}, \mathcal{A}, T, R, \gamma)$ 构成：$\mathcal{S}$ 是状态的有限集合；$\mathcal{A}$ 是动作的有限集合；状态转移函数 $T:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\to[0,1]$ 刻画在选定动作下状态间的转移概率；奖励函数 $R:\mathcal{S}\times\mathcal{A}\to\mathbb{R}$ 量化在某状态下采取某动作的即时奖励；折扣因子 $\gamma\in[0,1)$；策略 $\pi:\mathcal{S}\times\mathcal{A}\to[0,1]$ 为每个状态 $s$ 分配一个动作分布，动作 $a$ 的概率为 $\pi(a|s)$。

> 🇬🇧 **MDP Formulation for PM with CSPs.** Existing work [32, 35] focus on the GSP and lacks formal modeling in the broader context of CSPs. We denote a GSP as $U$, consisting of $N$ individual stocks. By randomly masking some stocks that investors are unfavorable to, it can generate a diverse set of CSPs, which are subsets of $U$. We define a sub-pool of GSP $U$ that masks $N^*$ stocks at time step $t$ to create CSP $C_t$, where $|C_t|=N-N^*$. We model an MDP for PM with CSP $C_t$ using maskable stock representation. The widely used MDP formulation in existing work is a special case in our PM with CSPs formulation when $C_t=U$. The details of the PM with CSPs as an MDP are set as follows:

💬 **CSP 下 PM 的 MDP 形式化**。现有工作 [32, 35] 仅聚焦于 GSP，在更广义的 CSP 语境下缺乏正式建模。我们记 GSP 为 $U$，含 $N$ 只个股。通过随机掩码掉投资者不看好的部分股票，可生成一组多样的 CSP，它们都是 $U$ 的子集。我们定义：在时刻 $t$ 对 GSP $U$ 掩码 $N^*$ 只股票而得到的子池为 CSP $C_t$，其中 $|C_t|=N-N^*$。我们用**可掩码股票表征**为 CSP $C_t$ 下的 PM 建立 MDP。当 $C_t=U$ 时，现有工作中广泛使用的 MDP 形式化就是我们这套 CSP 形式化的一个特例。CSP 下 PM 作为 MDP 的具体设定如下：

> 🇬🇧 **• State.** The GSP stocks time series are composed of three components: historical OHLC prices $P_t$, classical technical indicators $Y_t$, and temporal information $D_t$. We denote the feature of GSP $U$ during the historical $D$ time steps as $X_t = [P_t, Y_t, D_t] = [x^1_t, x^2_t, \dots, x^N_t]\in\mathbb{R}^{N\times D\times F}$, $F$ refers to the feature dimension... After masked $N^*$ stocks, $X^*_t=[x_{t-D+1}, x_{t-D+2}, \dots, x_t]\in\mathbb{R}^{(N-N^*)\times D\times F}$. ... the process entails duplicating and populating with a learnable masked token $[M]$. Then, the state at time $t$ is represented as $s_t=[x^1_t, x^2_t, \dots, x^{N-N^*}_t, [M], [M], \dots, [M]]\in\mathbb{R}^{N\times D\times F}$, where the number of $[M]$ is $N^*$.

💬 **• 状态（State）**。GSP 股票的时间序列由三部分组成：历史 OHLC 价格 $P_t$、经典技术指标 $Y_t$、时间信息 $D_t$。我们把 GSP $U$ 在过去 $D$ 个时间步内的特征记为 $X_t = [P_t, Y_t, D_t] = [x^1_t, x^2_t, \dots, x^N_t]\in\mathbb{R}^{N\times D\times F}$，$F$ 为特征维度。掩码掉 $N^*$ 只股票后，余下 $N-N^*$ 只看好股票的特征为 $X^*_t\in\mathbb{R}^{(N-N^*)\times D\times F}$。为了把维度补回 GSP 的规模，过程包括「复制并用一个可学习的掩码 token $[M]$ 填充」。于是时刻 $t$ 的状态表示为
$$s_t=[x^1_t, x^2_t, \dots, x^{N-N^*}_t, \underbrace{[M], [M], \dots, [M]}_{N^*\ \text{个}}]\in\mathbb{R}^{N\times D\times F}$$
其中 $[M]$ 的数量为 $N^*$。

📌 译注：PDF 这一段公式因双栏排版被打乱，已据上下文复原。要点：把「看好的股票真实特征」+「$N^*$ 个掩码占位符」拼成固定长度 $N$ 的状态，从而让动作维度恒定、智能体可复用。

> 🇬🇧 **• Action.** Given the state, the action of the agent at time step $t$ can be entirely represented by the portfolio vector $a_t=W_t=[w^0_t; w^1_t, \dots, w^{N-N^*}_t; w^1_{[M],t}, w^2_{[M],t}, \dots, w^{N^*}_{[M],t}]\in\mathbb{R}^{N+1}$. The proportion of cash retained is represented by $w^0_t$, $[w^1_t, \dots, w^{N-N^*}_t]$ denotes the proportion of $N-N^*$ investor favorable stocks, $[w^1_{[M],t}, \dots, w^{N^*}_{[M],t}]$ denotes the proportion of $N^*$ investor unfavorable stocks.

💬 **• 动作（Action）**。给定状态，时刻 $t$ 智能体的动作可完全由组合向量表示：
$$a_t=W_t=[\,w^0_t;\ \underbrace{w^1_t, \dots, w^{N-N^*}_t}_{\text{看好股票}};\ \underbrace{w^1_{[M],t}, \dots, w^{N^*}_{[M],t}}_{\text{被掩码股票}}\,]\in\mathbb{R}^{N+1}$$
其中 $w^0_t$ 是保留的现金比例，$[w^1_t,\dots,w^{N-N^*}_t]$ 是 $N-N^*$ 只投资者看好股票的比例，$[w^1_{[M],t},\dots,w^{N^*}_{[M],t}]$ 是 $N^*$ 只不看好股票的比例（理想情况下应被压到 0）。

> 🇬🇧 **• Reward.** Drawing on prior research [19, 27, 28], for each trading time step $t$, the reward $r_t$ is the change of portfolio value $r_t=V_t-V_{t-1}$.

💬 **• 奖励（Reward）**。借鉴已有研究 [19, 27, 28]，每个交易时间步 $t$ 的奖励 $r_t$ 即组合价值的变化：$r_t=V_t-V_{t-1}$。

---

## 4 EARNMORE（方法）

> 🇬🇧 As shown in Figure 2, we present a reinforcement learning framework called EarnMore with one-time training on a GSP, enabling invest on various CSPs and achieving optimal investment portfolios. With this framework, investors have the flexibility to invest in stock pools aligned with their individual preferences. Furthermore, during the trading process, these stock pools can be dynamically adapted to construct more efficient and tailored portfolios, aligning with investors' current decisions. EarnMore consists of three main components: i) a unified approach for representing customized stock pools with different sizes and stocks, which we call maskable stock representation (§4.1); ii) reinforcement learning optimization procedures for PM with CSPs (§4.2); iii) a re-weighting mechanism that concentrates on favorable stocks and neglects unfavorable ones to rescale the distribution of portfolios (§4.3).

💬 如图 2 所示，我们提出名为 **EarnMore** 的强化学习框架：只在 GSP 上训练一次，即可投资于各种 CSP 并得到最优组合。借助该框架，投资者可灵活地投资符合自身偏好的股票池；而且在交易过程中，这些股票池还能动态调整，以构建更高效、更贴合投资者当前决策的组合。EarnMore 包含三大组件：i) 表示大小、成分各异的定制股票池的统一方法，即**可掩码股票表征**（§4.1）；ii) CSP 下 PM 的强化学习优化流程（§4.2）；iii) 聚焦看好股票、忽略不看好股票，对组合分布重新缩放的**重加权机制**（§4.3）。

**图 2**：EarnMore 整体架构。模块 (a) 从 GSP 提取**股票级**（stock-level）嵌入；模块 (b) 是「掩码—重建」过程，学习**池级**（pool-level）嵌入；模块 (c) 是具备掩码 token 感知能力的智能体。

### 4.1 面向 CSP 的可掩码股票表征（Maskable Stock Representation for CSPs）

> 🇬🇧 Consistent representation is crucial for CSPs with different sizes and stocks before portfolio decisions. If only the stocks within the CSP are embedded into the agent for decision-making, and discarding the masked-out stocks will lead to three issues. Firstly, the action dimension of agent cannot adapt to different CSP sizes. Secondly, the agent may perform poorly or even fail due to the inability to distinguish between the different CSPs when dealing with CSPs of the same size but different stocks. Finally, discarding unfavorable stocks may lead to a loss of relationships between stocks, which will negatively impact performance. To address them, we introduce a maskable stock representation that identifies the position of each stock within the GSP. We employ two levels of stock representation, which are stock-level in Module (a) and pool-level constructed through masking and reconstruction in Module (b). This process reveals hidden connections between stocks, and our maskable stock representation is based on the pool-level.

💬 在做组合决策之前，为大小、成分各异的 CSP 提供**一致的表征**至关重要。如果只把 CSP 内部的股票嵌入到智能体里做决策、并丢弃被掩码掉的股票，会带来三个问题：第一，智能体的**动作维度**无法适配不同大小的 CSP；第二，当面对大小相同但成分不同的 CSP 时，智能体因无法区分而可能表现糟糕甚至失败；第三，丢弃不看好的股票会导致**股票间关系的丢失**，从而损害性能。为解决这些问题，我们引入一种**可掩码股票表征**，它标识每只股票在 GSP 中的位置。我们采用两个层级的股票表征：模块 (a) 的**股票级**（stock-level），与模块 (b) 通过「掩码—重建」构建的**池级**（pool-level）。这一过程能揭示股票之间的隐藏关联；我们的可掩码股票表征以池级表征为基础。

> 🇬🇧 **Learning for Stock-level Representation.** In Equation 2, we exploit stock features (prices and technical indicators) and temporal characteristics for stock-level representation. Following the approach in Timesnet [33], we employ 1D convolution to produce dense embeddings for stock features and utilize an embedding layer to handle sparse temporal features. The final stock-level representation is formed by the summation of these dense and sparse embeddings. It is defined as follows:

💬 **学习股票级表征**。在式 (2) 中，我们利用股票特征（价格与技术指标）和时间特性来得到股票级表征。沿用 TimesNet [33] 的做法，我们用**一维卷积**为股票特征生成稠密嵌入（dense embedding），并用一个嵌入层处理稀疏的时间特征（sparse temporal feature）。最终的股票级表征由这两类嵌入相加得到：

$$l_s(X_t)=\psi_e(D_t;\theta_e)+\psi_c(P_t, Y_t;\theta_c) \tag{2}$$

> 🇬🇧 where $\psi_e, \psi_c$ denote the embedding layer and 1D convolutional layer, and $\theta_e, \theta_c$ are their learnable parameters respectively.

💬 其中 $\psi_e$、$\psi_c$ 分别表示嵌入层与一维卷积层，$\theta_e$、$\theta_c$ 是它们各自的可学习参数。

📌 译注：「稠密 vs 稀疏」——价格、技术指标是连续数值，适合卷积提取（稠密）；而「日期/星期几」这类时间标记是离散类别，用嵌入层查表（稀疏）。两者相加得到每只股票的综合表征。

> 🇬🇧 **Learning for Pool-level Representation.** Stock-level representation describes the vertical time series information within each individual stock without capturing the horizontal inter-stock representation. In our PM with CSPs environment, directly masking certain stocks could potentially result in losing important and valuable connections between stocks when using them as stock representations. To address this limitation, we introduce the pool-level representation, which strengthens the connections between stocks in the GSP through the masking and reconstruction process. Notably, we employ stock-level embedding as the local embedding to replace the patching embedding for historical data employed in MAEs [12] or PatchTST [21].

💬 **学习池级表征**。股票级表征刻画的是每只个股内部「纵向」的时间序列信息，并未捕捉股票之间「横向」的相互表征。在我们的 CSP 下 PM 环境中，若直接把某些股票掩码后再当作股票表征使用，可能会丢失股票间重要而宝贵的关联。为解决这一局限，我们引入**池级表征**，通过「掩码—重建」过程强化 GSP 内股票之间的关联。值得注意的是，我们用股票级嵌入作为**局部嵌入**（local embedding），取代了 MAE [12] 或 PatchTST [21] 中对历史数据所用的「补丁嵌入」（patching embedding）。

> 🇬🇧 During training, we utilize the adaptive masking strategy developed by MAGE [17] to simulate various CSPs with varying stock numbers and compositions, which improves the representational capability of the pool-level embedding and unifies the high and low-masking-ratio stock pools under the same training framework. We sample a masking ratio $r$ from a truncated Gaussian distribution:

💬 训练时，我们采用 MAGE [17] 提出的**自适应掩码策略**，以模拟股票数量与构成各异的多种 CSP。这既提升了池级嵌入的表征能力，又把「高掩码率」和「低掩码率」的股票池统一到同一个训练框架下。我们从一个**截断高斯分布**（truncated Gaussian）中采样掩码率 $r$：

$$g(r;\mu,\sigma,a,b)=\varphi\!\left(\tfrac{r-\mu}{\sigma}\right)\Big/\left(\Phi\!\left(\tfrac{b-\mu}{\sigma}\right)-\Phi\!\left(\tfrac{a-\mu}{\sigma}\right)\right) \tag{3}$$

> 🇬🇧 where $\varphi(\cdot)$ is the probability density function of the standard normal distribution, $\Phi(\cdot)$ is its cumulative distribution function, $a$ and $b$ are the lower and upper bounds.

💬 其中 $\varphi(\cdot)$ 是标准正态分布的概率密度函数，$\Phi(\cdot)$ 是其累积分布函数，$a$、$b$ 是截断的下界与上界。（附录 C 给出超参：$a=0.6, b=0.8, \mu=0.7, \sigma=0.1$。）

📌 译注：截断高斯让掩码率集中在 0.6–0.8 之间——即每次训练随机遮住约 60%–80% 的股票，模拟「投资者只挑了少量股票」的真实 CSP 场景。

> 🇬🇧 In Equation 4, the process for constructing maskable stock representation is outlined through an encoder and decoder procedure. The process starts with the encoder phase, a random masking ratio $r$ is sampled and then is used to mask a subset of stock-level embeddings selectively using the masking operation $\eta_{mo}$. Only the unmasked embeddings are retained and subsequently fed into the encoder $\psi_{enc}$ to extract latent embeddings. During the decoder phase, the latent embeddings are filled to the number of stock-level embeddings using a learnable masked token called $m$ via the mask-filled operation $\eta_{mf}$. Finally, the decoder $\psi_{dec}$ is used to reconstruct the price $\tilde{P}_t$ of masked stocks:

💬 式 (4) 通过「编码器—解码器」流程描述了可掩码股票表征的构建过程。**编码器阶段**：采样一个随机掩码率 $r$，用掩码操作 $\eta_{mo}$ 选择性地遮住一部分股票级嵌入；只保留**未被遮住**的嵌入并送入编码器 $\psi_{enc}$，提取潜在嵌入（latent embedding）。**解码器阶段**：用一个名为 $m$ 的可学习掩码 token，通过「掩码填充」操作 $\eta_{mf}$ 把潜在嵌入补回到原本的股票级嵌入数量；最后由解码器 $\psi_{dec}$ 重建被掩码股票的价格 $\tilde{P}_t$：

$$
\begin{aligned}
l_p(X_t) &= \psi_{enc}\big(\eta_{mo}(l_s(X_t), g(r;\mu,\sigma,a,b));\ \theta_{enc}\big)\\
\rho(X_t, m) &= \eta_{mf}(l_p(X_t), m)\\
\tilde{P}_t &= \psi_{dec}\big(\rho(X_t, m);\ \theta_{dec}\big)
\end{aligned}
\tag{4}
$$

> 🇬🇧 where $\psi_{enc}, \psi_{dec}$ denote the Encoder and Decoder, $\theta_{enc}, \theta_{dec}$ are their respective learnable parameters.

💬 其中 $\psi_{enc}$、$\psi_{dec}$ 分别表示编码器与解码器，$\theta_{enc}$、$\theta_{dec}$ 是它们各自的可学习参数。

> 🇬🇧 After filling in the masked token, we refer to the resulting latent embedding as maskable stock representation, and we abbreviate $\eta_{mf}(l_p(X_t), m)$ as $\rho(X_t, m)$, which will be used as state for portfolio decision making in the reinforcement learning process. The masked token makes the agent sense which stocks are unfavorable to the investor, thus catering to the investor's preferences and personal decisions, and enabling collaboration between the agent and investor, who has access to various sources of information and have some expectation of the future direction of specific stocks. It is important to mention that we retain the [CLS] token to understand how cash is allocated. This token can capture the overall sequence representation and preserve global sequence information when decoding, which is exactly what we need.

💬 填入掩码 token 后，所得的潜在嵌入即我们所称的**可掩码股票表征**，我们把 $\eta_{mf}(l_p(X_t), m)$ 简记为 $\rho(X_t, m)$，它将作为强化学习过程中做组合决策的**状态**。掩码 token 让智能体能「感知」哪些股票是投资者不看好的，从而迎合投资者的偏好与个人决策，实现智能体与投资者之间的协作——投资者能接触多种信息来源，对特定股票的未来走向有自己的预期。需要特别说明的是，我们保留了 **[CLS] token** 以理解现金如何分配：该 token 能在解码时捕捉整体的序列表征、保留全局序列信息，这正是我们所需要的。

📌 译注：[CLS] 是 BERT 里的「汇总 token」，本文借它来代表「现金」这一特殊资产，让模型有专门的通道决定持币比例。

### 4.2 CSP 下 PM 的强化学习优化（RL Optimization for PM with CSPs）

> 🇬🇧 Our reinforcement learning training process is based on the Soft Actor-Critic (SAC) [11]. There are two main components called Actor and Critic in RL optimization. The Actor utilizes the latent embeddings populated by masked tokens to generate actions that indicate the allocation ratios for cash and individual stocks, and the Actor is aware of masked token for unfavorable stocks and will avoid allocating them during decision-making. The Critic evaluates portfolio performance using populated latent embeddings with masked tokens and actions that the Actor generates. This evaluation provides a scoring mechanism that guides the learning process and helps optimize the portfolio management strategy.

💬 我们的强化学习训练过程基于 **Soft Actor-Critic（SAC）** [11]。RL 优化包含两个主要组件：**Actor**（演员，即策略网络）与 **Critic**（评论家，即价值网络）。Actor 利用填入掩码 token 后的潜在嵌入生成动作，表示现金与各股票的分配比例；Actor 能感知到不看好股票的掩码 token，会在决策时避免给它们分配资金。Critic 则用这些填充后的潜在嵌入与 Actor 生成的动作来评估组合表现；这一评估提供了一种打分机制，引导学习过程、帮助优化投资组合管理策略。

📌 译注：SAC 是一种 off-policy 的 actor-critic 算法，特点是在目标里加入**熵正则**（鼓励探索），适合连续动作空间——投资比例正是连续值，故选它。

> 🇬🇧 We utilize two contrasting strategies to penalize the actor-critic from assigning weights to the masked stocks. The first method involves adding an additional supervised loss to the actor output's portfolios. The second method involves adding a penalty term to the TD error when there is a non-zero investment portfolio for masked stocks. The first approach yields better results because the supervised loss affects the actor portfolios in a more direct way.

💬 我们用两种对比性策略来惩罚 actor-critic「给被掩码股票分配权重」的行为：第一种是在 actor 输出的组合上额外加一个**监督损失**；第二种是当被掩码股票出现非零投资时，在 **TD 误差**上加一个惩罚项。第一种方法效果更好，因为监督损失能更直接地影响 actor 的组合输出。

> 🇬🇧 **Optimization for Q-Value Network.** We use maskable stock representation $\rho(X_t, m)$ defined in Equation 4 instead of raw market data as input states $\rho$ for Actor and Critic. Let $Q_\theta(s,a)=Q_\theta(\rho,a)$ represent the $Q$-value function, and $\pi_\phi(\rho,a)$ denote the policy function. Assumed that the output of $\pi_\phi$ follows a normal distribution with expectation and variance, the $Q$-value function can be learned by minimizing the flexible Bellman residuals:

💬 **Q 值网络的优化**。我们用式 (4) 定义的可掩码股票表征 $\rho(X_t, m)$ 作为 Actor 和 Critic 的输入状态 $\rho$，而非原始市场数据。设 $Q_\theta(s,a)=Q_\theta(\rho,a)$ 为 $Q$ 值函数，$\pi_\phi(\rho,a)$ 为策略函数。假设 $\pi_\phi$ 的输出服从某个具有期望与方差的正态分布，$Q$ 值函数可通过最小化（柔性）**贝尔曼残差**来学习：

$$J_Q(\theta)=\mathbb{E}_{(s_t,a_t)\sim\mathcal{D}}\!\left[\tfrac{1}{2}\Big(Q_\theta(\rho_t,a_t)-\big(r(s_t,a_t)+\gamma\,\mathbb{E}_{s_{t+1}\sim p}[V_{\bar\theta}(\rho_{t+1})]\big)\Big)^2\right] \tag{5a}$$

$$V_{\bar\theta}(\rho_t)=\mathbb{E}_{a_t\sim\pi}\!\left[Q_{\bar\theta}(\rho(s_t,m),a_t)-\alpha\log\pi_\phi(a_t|\rho_t)\right] \tag{5b}$$

> 🇬🇧 where $\rho(s_t,m)$ abbreviated as $\rho_t$, $Q_{\bar\theta}$ represents the target Q-value network, $\bar\theta$ is the exponential moving average of the parameter $\theta$.

💬 其中 $\rho(s_t,m)$ 简记为 $\rho_t$，$Q_{\bar\theta}$ 是**目标 Q 值网络**（target network），$\bar\theta$ 是参数 $\theta$ 的指数移动平均（EMA）。式 (5b) 中的 $-\alpha\log\pi_\phi$ 即 SAC 的熵项。

> 🇬🇧 **Optimization for Policy Network.** To optimize $J_\pi(\phi)$, we utilize the reparameterization technique for the policy network $\pi_\phi$. This technique involves representing $\pi_\phi$ as a function that takes the state $s$ and standard Gaussian samples $\epsilon$ as inputs and directly outputs the action $a=f_\phi(\epsilon;s)$. Assuming $\mathcal{N}$ is the standard normal distribution, $\pi_\phi$ can be derived by minimizing KL divergence:

💬 **策略网络的优化**。为优化 $J_\pi(\phi)$，我们对策略网络 $\pi_\phi$ 使用**重参数化技巧**（reparameterization trick）：把 $\pi_\phi$ 表示为一个以状态 $s$ 和标准高斯样本 $\epsilon$ 为输入、直接输出动作 $a=f_\phi(\epsilon;s)$ 的函数。设 $\mathcal{N}$ 为标准正态分布，$\pi_\phi$ 可通过最小化 KL 散度求得：

$$J_\pi(\phi)=\mathbb{E}_{s_t\sim D,\ \epsilon_t\sim\mathcal{N}}\!\left[\alpha\log\pi_\phi\big(f_\phi(\epsilon_t;\rho_t)\,|\,s_t\big)-Q_\theta\big(\rho_t, f_\phi(\epsilon_t;\rho_t)\big)\right] \tag{6}$$

📌 译注：重参数化把「从分布中采样」改写成「确定性函数 + 独立噪声 $\epsilon$」，使梯度能穿过采样步骤进行反向传播。

> 🇬🇧 **Optimization for Parameter Alpha.** We employ an automatic entropy tuning method to adjust parameter $\alpha$ by minimizing the following loss function:

💬 **温度参数 $\alpha$ 的优化**。我们采用**自动熵调节**（automatic entropy tuning）方法，通过最小化下式来调整参数 $\alpha$：

$$J(\alpha)=\mathbb{E}_{a_t\sim\pi_t}\!\left[-\alpha\log\pi_t\big(a_t\,|\,\rho(s_t,m)\big)-\alpha\bar{\mathcal{H}}\right] \tag{7}$$

> 🇬🇧 where $\bar{\mathcal{H}}$ is the target entropy hyperparameter.

💬 其中 $\bar{\mathcal{H}}$ 是**目标熵**超参数。$\alpha$ 控制「探索（熵）」与「利用（收益）」之间的权衡，自动调节免去了手工调参。

> 🇬🇧 **Optimization for Maskable Stock Representation.** In the masking and reconstruction process, we optimize the maskable stock representation using mean-squared error. Reconstruction losses are calculated based only on the price of masked stocks:

💬 **可掩码股票表征的优化**。在「掩码—重建」过程中，我们用**均方误差**（MSE）优化可掩码股票表征。重建损失**只**基于被掩码股票的价格计算：

$$J(\theta_e,\theta_c,\theta_{enc},\theta_{dec})=\frac{1}{N^*}\sum_{i=1}^{N^*}\big(p_{i,t}-\tilde{p}_{i,t}\big)^2 \tag{8}$$

> 🇬🇧 where $N^*$ represents the number of masked stocks. Notably, pre-training the encoder on maskable stock representations faces two main drawbacks: i) creates a gap between self-supervised price prediction tasks and RL decision-making tasks; ii) potentially limits the exploration space in RL by frozen embeddings that may not always positively impact decision-making. Hence, conducting maskable stock representation optimization and RL optimization simultaneously contribute to better performance and more user-friendly, end-to-end models.

💬 其中 $N^*$ 是被掩码股票的数量。值得注意的是，若**预先（单独）训练**编码器来学可掩码表征，会有两个主要缺点：i) 在「自监督价格预测任务」与「RL 决策任务」之间产生割裂；ii) 冻结的嵌入可能并不总是有利于决策，反而限制了 RL 的探索空间。因此，**同时**进行可掩码表征优化与 RL 优化，能带来更好的性能，也使模型更友好、可端到端训练。

> 🇬🇧 Our implementation process follows the same optimization process for each batch of SAC. We first optimize the Q-value network, followed by the alpha, strategy network, and maskable stock representation. However, we find that using weighted sum loss to optimize both the maskable stock representation and the remaining three components will have a negative impact on the distribution of data sampled by the RL process and lead to unstable training.

💬 我们的实现对 SAC 的每个批次都遵循同样的优化顺序：先优化 Q 值网络，接着是 $\alpha$、策略网络，最后是可掩码股票表征。不过我们发现，若用**加权求和损失**同时优化可掩码表征与其余三个组件，会对 RL 过程所采样数据的分布产生负面影响，导致训练不稳定（故采用上述分步优化）。

### 4.3 重加权方法（Re-weighting Method）

> 🇬🇧 In a continuous decision space in portfolio management, agents face difficulties in making accurate decisions. For instance, in a constantly changing market environment, agents may overfit a fixed number of market patterns and struggle to react quickly in high-volatility markets. These issues can lead to agents micro-investing in stocks with low future returns or even result in losses.

💬 在投资组合管理的**连续决策空间**中，智能体很难做出精确决策。例如在不断变化的市场环境里，智能体可能过拟合固定数量的市场模式，难以在高波动市场中快速反应。这些问题会让智能体在低未来收益的股票上「微量投资」（micro-invest），甚至造成亏损。

> 🇬🇧 In our setting for PM with CSPs, we encounter unique problems in addition to those already present in PM: i) the state that we input to the agent is latent embeddings containing filled masked tokens, the agent may be investing in the masked stocks that these investors expect to lose money, which is precisely what investors do not want to witness; ii) due to the extent of error in decision-making, part of the investment proportion of high-yield stocks may be taken by stocks with low or negative expected future returns.

💬 在我们的 CSP 下 PM 设定中，除了 PM 固有问题外还会遇到独特问题：i) 输入给智能体的状态是含有填充掩码 token 的潜在嵌入，智能体可能会投资那些投资者预期会亏钱的被掩码股票——这正是投资者最不愿看到的；ii) 由于决策存在误差，高收益股票本应占有的部分投资比例，可能被低收益或负期望收益的股票「抢占」。

> 🇬🇧 Both problems can be solved by portfolio sparsification. Drawing inspiration from the Boltzmann distribution and Gumbel-Softmax, we introduce an additional hyperparameter $T$ to the softmax function for re-weighting portfolios to achieve sparsification of tiny investment proportions to zero:

💬 这两个问题都可以通过**组合稀疏化**（portfolio sparsification）解决。受**玻尔兹曼分布**与 **Gumbel-Softmax** 的启发，我们在 softmax 函数中引入一个额外的超参数 $T$，对组合进行重加权，把微小的投资比例稀疏化为零：

$$Re(x)=e^{x_i/T}\Big/\sum_{j=1}^{N}e^{x_j/T} \tag{9}$$

> 🇬🇧 In this context, $x$ represents the Actor's logits for portfolio, and $T\in(0,\infty)$ is a temperature parameter. Lower $T$ values lead to sparser allocations. As $T$ approaches 0, all investments tend to allocate to the asset with the highest expected return. For $T=1$, re-weighting degenerates to softmax, while for $T>1$, it reduces the allocation variance and even leads to equal allocation. Notably, re-weighting is included during RL optimization and will be used during the training and testing as shown in Appendix D.

💬 这里 $x$ 是 Actor 给出的组合 logits，$T\in(0,\infty)$ 是**温度参数**（temperature）。$T$ 越小，分配越稀疏：当 $T\to0$ 时，所有资金趋向于全部押在期望收益最高的那一个资产上；当 $T=1$ 时，重加权退化为普通 softmax；当 $T>1$ 时，它会降低分配的方差，甚至导致等额分配。值得注意的是，重加权同时纳入 RL 优化过程，并在训练与测试中都会使用（见附录 D）。本文最优 $T=0.1$（在 $\{10,5,1,0.5,0.1,0.05,0.01\}$ 中网格搜索得到）。

📌 译注：温度 $T$ 越低，softmax 输出越「尖锐」（接近 one-hot），等价于让组合更集中、更敢于重仓少数股票。这正是把「微小仓位」清零的机制。

---

## 5 实验（EXPERIMENTS）

> 🇬🇧 In this section, we conduct a series of experiments to evaluate the proposed framework. First, we demonstrate that our approach achieves better returns in two real US financial markets and substantially outperforms the baseline methods in the global stock pool. Next, we construct 6 customizable stock pools based on 3 different investor investment preferences in two US financial markets to demonstrate that our framework can effectively meet investors' preferences and decisions in the trading process. Finally, we conduct ablation studies to answer the following questions:

💬 本节通过一系列实验评估所提框架。首先，我们证明本方法在两个真实美股市场上取得更高收益，并在全局股票池（GSP）上大幅超越基线方法。接着，我们基于 3 种不同的投资者偏好、在两个美股市场上构建了 6 个可定制股票池，证明框架能在交易过程中有效满足投资者的偏好与决策。最后，通过消融实验回答以下问题：

> 🇬🇧 **RQ1:** How is the usefulness of each component of EarnMore?
> **RQ2:** Why are direct methods for PM with CSPs not working?
> **RQ3:** How is the efficiency of the EarnMore model?

💬
- **RQ1**：EarnMore 各组件各自有多大用处？
- **RQ2**：为什么处理 CSP 下 PM 的「直接方法」行不通？
- **RQ3**：EarnMore 模型的效率如何？

### 5.1 数据集与处理（Datasets and Processing）

**表 1**：数据集与日期划分

| 数据集 | SP500-GSP | SP500-CSP1 | SP500-CSP2 | SP500-CSP3 | DJ30-GSP | DJ30-CSP1 | DJ30-CSP2 | DJ30-CSP3 |
|---|---|---|---|---|---|---|---|---|
| 股票数 | 420 | 62 | 39 | 168 | 28 | 10 | 7 | 10 |
| 行业数 | 49 | 8 | 7 | 28 | 24 | 9 | 6 | 8 |

| 日期划分 | 训练（Train） | 测试（Test） |
|---|---|---|
| 划分 1 | 2007-09-26 ∼ 2018-01-25 | 2018-01-26 ∼ 2019-07-22 |
| 划分 2 | 2007-09-26 ∼ 2019-07-22 | 2019-07-23 ∼ 2021-01-08 |
| 划分 3 | 2007-09-26 ∼ 2021-01-07 | 2021-01-07 ∼ 2022-06-26 |

> 🇬🇧 In our experiment, we study daily data for 10,273 US stocks from Yahoo Finance, deriving indicators to understand market trends. After preprocessing to address data quality issues, we ended up with 3,094 US stocks and 95 technical indicators based on Qlib's Alpha158 [34]. We conduct a comprehensive evaluation to validate our framework's effectiveness and performance in various real-world scenarios. We consider two main factors in the evaluation: i) events and markets under different conditions, e.g. COVID-19, geopolitical conflicts, bull and bear; ii) customizable stock pools with different investors' investment preferences, such as one investor prefers investing in the technology and communication industries, and another one prefers investing in financial and insurance industries.

💬 实验中，我们研究来自 Yahoo Finance 的 10,273 只美股的日频数据，并衍生指标以理解市场趋势。经过预处理解决数据质量问题后，最终基于 Qlib 的 **Alpha158** [34] 得到 3,094 只美股和 95 个技术指标。我们做了全面评估，以在多种真实场景下验证框架的有效性与性能。评估考虑两大因素：i) 不同条件下的事件与市场，如新冠疫情、地缘政治冲突、牛市与熊市；ii) 体现不同投资者偏好的可定制股票池，例如一位投资者偏好科技与通信行业，另一位偏好金融与保险行业。

> 🇬🇧 We construct 8 diversified datasets from the two US stock indices, which are SP500 and DJ30. According to the Global Industry Classification Standard (GICS), we categorize SP500 and DJ30 into 49 and 24 industries at the industry level. Examples of industries include banking, insurance, software services, automotive manufacturing, and so on. Global stock pool (GSP) is the full set of stock pools containing 420 and 28 stocks respectively. Then we carefully select three CSPs based on three different investor preferences, with CSP1, CSP2, and CSP3 corresponding to the technology, financial, and service as the main industries. To better reflect the real-world demand for industry diversity in PM, we randomly add several stocks from other industries into the three CSPs. It's worth mentioning that the antagonistic and cooperative correlations between stocks has been reflected in our diversified customizable stock pool selection. For example, the CSP1 on the DJ30 includes tech giants like Apple and Microsoft, showcasing their competitive relationship. The CSP1 on the SP500 includes Intel and Microsoft, illustrating their cooperative relationship (e.g., Intel's chips used with Microsoft Windows).

💬 我们从 SP500 和 DJ30 两大美股指数构建了 8 个多样化数据集。依据**全球行业分类标准**（GICS），我们在行业层面把 SP500、DJ30 分别划为 49 和 24 个行业，例如银行、保险、软件服务、汽车制造等。全局股票池（GSP）是完整股票集，分别含 420 和 28 只股票。然后我们依据 3 种不同投资者偏好精心挑选了三个 CSP：CSP1、CSP2、CSP3 分别以**科技、金融、服务**为主要行业。为更好反映 PM 中对行业多样性的真实需求，我们又向这三个 CSP 中随机加入了若干来自其他行业的股票。值得一提的是，多样化的 CSP 选择已体现了股票间的**对抗与协作关系**：例如 DJ30 上的 CSP1 含苹果、微软等科技巨头，体现其竞争关系；SP500 上的 CSP1 含英特尔和微软，体现其协作关系（如英特尔芯片配微软 Windows 使用）。

### 5.2 评估指标（Evaluation Metrics）

> 🇬🇧 We compare EarnMore and baselines in terms of 6 financial metrics, including 1 profit criterion, 3 risk-adjusted profit criteria, and 2 risk criteria. Definitions and formulas are available in Appendix A.

💬 我们用 6 个金融指标对比 EarnMore 与各基线，包括 1 个收益指标、3 个风险调整后收益指标、2 个风险指标。定义与公式见附录 A。

### 5.3 基线方法（Baselines）

> 🇬🇧 To provide a comprehensive comparison of EarnMore, we select 14 state-of-the-art and representative stock prediction methods of 4 different types consisting of 3 rule-based methods, 2 machine learning-based methods, 2 deep learning-based methods and 7 reinforcement learning-based methods. Details of them are available in Appendix B.

💬 为全面对比 EarnMore，我们选取了 14 个最先进且具代表性的股票预测方法，分 4 类：3 个基于规则（Rule-based）、2 个基于机器学习（ML-based）、2 个基于深度学习（DL-based）、7 个基于强化学习（RL-based）。详见附录 B。

### 5.4 实现细节（Implement Details）

> 🇬🇧 Experiments are conducted on an Nvidia A6000 GPU, and we use grid search to determine the hyperparameters. The implementation of those ML-based and DL-based methods is based on Qlib [34]. As for other baselines, we use the default settings in their public implementations. We run experiments with individual 9 runs using 3 date splits × 3 random selected seeds and report the average performance. Detailed implementation is available in Appendix C.

💬 实验在一块 Nvidia A6000 GPU 上进行，用网格搜索确定超参数。ML-based 与 DL-based 方法基于 Qlib [34] 实现；其他基线使用其公开实现的默认设置。每组实验做 9 次独立运行（3 个日期划分 × 3 个随机种子），报告平均性能。详细实现见附录 C。

### 5.5 结果与分析（Results and Analysis）

> 🇬🇧 **Performance on Global Stock Pools.** We compared EarnMore with 14 baseline methods in terms of 6 financial metrics. Table 2 and Figure 3 demonstrate our framework outperforms others on portfolio management with higher returns in GSPs. For the SP500, EarnMore achieves the highest ARR of 97% and SR of 2.032, significantly higher than the second-best method. For the DJ30, EarnMore achieves improvements in terms of ARR, SR, CR, and SoR by 46.7%, 8.9%, 6.2%, and 2.4%.

💬 **全局股票池上的表现**。我们在 6 个金融指标上把 EarnMore 与 14 个基线对比。表 2 与图 3 表明，在 GSP 上我们的框架以更高收益胜出。在 SP500 上，EarnMore 取得最高的 **ARR 97%** 和 **SR 2.032**，显著高于第二名；在 DJ30 上，ARR、SR、CR、SoR 分别提升 **46.7%、8.9%、6.2%、2.4%**。

**表 2**：SP500 与 DJ30 在全局股票池（GSP）上的性能对比（↑ 越大越好，↓ 越小越好；ARR=年化收益率，SR=夏普比率，CR=卡玛比率，SOR=索提诺比率，MDD=最大回撤，VOL=波动率）。

| 类别 | 策略 | SP500-ARR% | SP500-SR | SP500-CR | SP500-SOR | SP500-MDD% | SP500-VOL | DJ30-ARR% | DJ30-SR | DJ30-CR | DJ30-SOR | DJ30-MDD% | DJ30-VOL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | Market（市场） | 9.320 | 0.556 | 0.702 | 17.120 | 26.160 | 0.014 | 6.710 | 0.458 | 0.776 | 15.560 | 22.200 | 0.013 |
| Rule | BLSW | 11.630 | 0.696 | 0.894 | 21.450 | 24.560 | 0.013 | 7.610 | 0.512 | 0.857 | 16.930 | 21.540 | 0.012 |
| Rule | CSM | 5.070 | 0.329 | 0.434 | 9.840 | 23.350 | 0.013 | 5.930 | 0.400 | 0.643 | 12.950 | 20.770 | 0.012 |
| ML | XGBoost | 10.690 | 0.377 | 0.473 | 13.650 | 19.300 | 0.016 | 10.260 | 0.343 | 0.599 | 10.420 | 14.760 | 0.013 |
| ML | LightGBM | 16.330 | 0.575 | 0.744 | 20.110 | 24.760 | 0.016 | 13.420 | 0.591 | 0.703 | 14.220 | 20.900 | 0.014 |
| DL | ALSTM | 43.50 | 1.157 | 1.367 | 22.501 | 35.820 | 0.026 | 15.030 | 1.186 | 0.590 | 14.890 | 28.070 | 0.013 |
| DL | TCN | 13.560 | 1.044 | 1.460 | 14.540 | 35.780 | 0.025 | 6.980 | 0.732 | 0.269 | 8.280 | 37.400 | 0.018 |
| RL | PG | 12.580 | 0.431 | 0.519 | 24.340 | 26.180 | 0.014 | 7.970 | 0.321 | 0.435 | 8.430 | 21.570 | 0.012 |
| RL | PPO | 15.130 | 0.537 | 0.742 | 14.770 | 24.100 | 0.013 | 9.240 | 0.385 | 0.512 | 10.140 | 20.810 | 0.012 |
| RL | SAC | 15.140 | 0.538 | 0.743 | 14.770 | 24.100 | 0.013 | 9.150 | 0.326 | 0.448 | 8.830 | 20.600 | 0.012 |
| RL | EIIE | 15.030 | 0.540 | 0.627 | 15.450 | 26.920 | 0.015 | 22.900 | 0.689 | 1.465 | 23.450 | 16.770 | 0.014 |
| RL | SARL | 21.240 | 0.756 | 0.970 | 21.230 | 24.000 | 0.013 | 21.920 | 0.786 | 1.109 | 23.020 | 20.400 | 0.012 |
| RL | IMIT | 50.300 | 1.162 | 1.949 | 35.050 | 25.420 | 0.018 | 27.640 | 0.909 | 1.593 | 27.380 | 20.050 | 0.014 |
| RL | DeepTrader | 60.290 | 1.980 | 2.195 | 34.260 | 28.580 | 0.013 | 32.230 | 1.335 | 1.440 | 27.110 | 21.190 | 0.013 |
| **RL** | **EarnMore** | **97.170** | **2.032** | **2.506** | **42.160** | 28.120 | 0.023 | **47.290** | **1.454** | **1.692** | **28.040** | 21.650 | 0.018 |
| — | 相对 SOTA 提升 | 61.171% | 2.626% | 14.169% | 20.285% | — | — | 46.727% | 8.914% | 6.215% | 2.411% | — | — |

> 🇬🇧 We can also observe that ML-based methods are optimal in controlling risk, but not outstanding in capturing returns. The reason behind it is that tree models are more robust to outliers and noise in the data, and thus can adaptively capture non-linear relationships to reduce decision risk. Specifically, ALSTM achieved a surprising 43.5% return on the SP500, due to large returns from several decisions in a large number of bad decisions, and thus we do not recommend using it. Besides, it is worth noting that the higher the potential return, the higher the risk involved in portfolio management. EarnMore is slightly inferior yet comparable to baseline methods on risk metrics, i.e., MDD and VOL. As for DJ30 dataset, EarnMore fails to perform well in MDD but achieves over 40% improvement in ARR to DeepTrader, which is the second best overall. Thus, for EarnMore, it is a slight compromise on risk control, as our priority is to maximize final portfolio values.

💬 我们还观察到：ML-based 方法在控制风险上最优，但捕捉收益的能力并不突出。原因在于树模型对数据中的离群点和噪声更鲁棒，能自适应捕捉非线性关系以降低决策风险。具体而言，ALSTM 在 SP500 上取得惊人的 43.5% 收益，但这是「大量糟糕决策中夹杂几次大赚决策」的结果，因此我们**不建议使用它**。此外值得注意：潜在收益越高，PM 中的风险也越高。EarnMore 在风险指标（MDD、VOL）上略逊于、但仍可与基线相比。在 DJ30 上，EarnMore 的 MDD 表现不佳，但 ARR 相比第二名 DeepTrader 提升超过 40%。因此对 EarnMore 而言，这是在风险控制上的轻微妥协——因为我们的首要目标是**最大化最终组合价值**。

> 🇬🇧 The global COVID-19 pandemic reached its peak between February 14 and March 20 2020, causing a significant decline in the economy and intense investor concerns. This led to a substantial fall in the stock market, with the SP500 and DJ30 indices falling by 31.81% and 34.78%, respectively. As shown in Figure 3, EarnMore is far less affected in returns than baseline methods and continues to gain returns after the market rebounded. Even during market downturns, EarnMore is able to identify stocks with the potential to generate higher returns when the market rebounds.

💬 全球新冠疫情在 2020 年 2 月 14 日至 3 月 20 日间达到高峰，引发经济大幅下滑与投资者强烈担忧，导致股市大跌——SP500 和 DJ30 指数分别下跌 31.81% 和 34.78%。如图 3 所示，EarnMore 的收益所受冲击远小于基线方法，并在市场反弹后继续获利。即使在市场下行期，EarnMore 也能识别出在市场反弹时有潜力产生更高收益的股票。

> 🇬🇧 **Performance on Customizable Stock Pools.** We illustrate the effectiveness of CSPs in two aspects. Firstly, we compare the profitability performance of CSPs-formulated according to investor preferences-with 3 state-of-the-art RL-based methods. Secondly, we demonstrate the adaptability and robustness of EarnMore to investors' personal decisions in the trading process.

💬 **可定制股票池上的表现**。我们从两方面展示 CSP 的有效性。第一，我们把按投资者偏好构建的 CSP 的盈利表现与 3 个最先进的 RL-based 方法对比。第二，我们展示 EarnMore 在交易过程中对投资者个人决策的适应性与鲁棒性。

**表 3**：SP500 与 DJ30 在可定制股票池（CSPs）上的性能对比（下划线表示最优）。

| 股票池 | 策略 | SP500-ARR% | SP500-SR | DJ30-ARR% | DJ30-SR |
|---|---|---|---|---|---|
| CSP1 | SARL | 34.330 | 0.820 | 24.140 | 0.638 |
| CSP1 | IMIT | 20.973 | 0.860 | 20.071 | 0.920 |
| CSP1 | DeepTrader | 34.030 | 0.793 | 27.740 | 0.757 |
| CSP1 | **EarnMore** | **122.610** | **2.278** | **53.990** | **1.810** |
| CSP2 | SARL | 17.000 | 0.570 | 20.020 | 0.820 |
| CSP2 | IMIT | 7.971 | 0.486 | 11.841 | 0.751 |
| CSP2 | DeepTrader | 34.030 | 0.793 | 38.470 | 0.955 |
| CSP2 | **EarnMore** | **110.110** | **2.279** | **43.400** | **1.549** |
| CSP3 | SARL | 18.090 | 0.760 | 10.910 | 0.480 |
| CSP3 | IMIT | 21.193 | 1.220 | 6.851 | 0.496 |
| CSP3 | DeepTrader | 61.320 | 1.489 | 16.840 | 0.601 |
| CSP3 | **EarnMore** | **93.670** | **2.120** | **43.460** | **1.572** |

> 🇬🇧 EarnMore achieves impressive performances on all CSPs, as shown in Table 3. Specifically, in the CSP1 stock pool, which consists of technology stocks, EarnMore's profitability is significantly higher compared to other methods. It is consistent with the notable increasing values of technology stocks over an extended period of time, and thus demonstrates that our method provides more scope for profit-seeking. In the other two CSPs, which are in the financials and services industries, EarnMore also delivers notable return improvements. Overall, our method is able to automatically adapt to investors' preferences and generate substantial returns.

💬 如表 3 所示，EarnMore 在所有 CSP 上都表现亮眼。具体而言，在以科技股为主的 CSP1 上，EarnMore 的盈利能力显著高于其他方法——这与科技股长期显著上涨一致，说明本方法为追求收益提供了更大空间。在另外两个分别属于金融与服务行业的 CSP 上，EarnMore 也带来显著的收益提升。总体而言，本方法能自动适应投资者偏好并产生可观收益。

> 🇬🇧 General Electric Company (GE) was delisted from the SP500 on June 26, 2018. As shown in Figure 4(a), after removing GE from the stock pool of the SP500 on the date of June 26, 2018, EarnMore can adapt itself to investor decisions to achieve a small return increase. Between March 2022 and June 2022, the stock price of Apple (APPL) dropped sharply by nearly 25% due to several factors, including the impact of the war and Apple's mobile phone downtime incident. As shown in Figure 4(b), excluding AAPL from DJ30 can significantly improve returns. It is important to mention that we add Microsoft (MSFT) technology stock to the CSP2 of the SP500. We purposely chose periods when the MSFT price was decreasing, to test the strength and robustness of EarnMore, in case an investor inadvertently selects an unsuitable stock. As depicted in Figure 4(c), EarnMore is able to decrease its MSFT investment by properly screening the stock with minimal to no impact on the overall returns throughout the trading procedure. Goldman Sachs (GS) announced the advancement of steel project and new energy vehicle development, which released a signal that stock price of GS would rise in 2021. Thus, our model adds GS to the stock pool, as shown in Figure 4(d), and gets more significant return growth.

💬 通用电气公司（GE）于 2018 年 6 月 26 日从 SP500 退市。如图 4(a) 所示，在该日把 GE 从 SP500 股票池移除后，EarnMore 能自适应投资者的决策，取得小幅收益提升。2022 年 3 月至 6 月，苹果（AAPL）股价因战争影响、苹果手机宕机事件等多种因素急跌近 25%；如图 4(b)，从 DJ30 中剔除 AAPL 可显著提升收益。需要说明的是，我们还把微软（MSFT）这只科技股**加入** SP500 的 CSP2，并**故意选择 MSFT 价格下跌的时段**，以测试当投资者不慎选了不合适股票时 EarnMore 的强度与鲁棒性。如图 4(c)，EarnMore 能在整个交易过程中通过恰当筛选，降低对 MSFT 的投资，对整体收益几乎无影响。高盛（GS）宣布推进钢铁项目与新能源汽车开发，释放出其股价将在 2021 年上涨的信号；于是我们的模型把 GS 加入股票池（图 4(d)），获得更显著的收益增长。

**图 4**：CSP 在动态变化下的表现（含增删 GE/AAPL/MSFT/GS 的对比曲线）。

### 5.6 消融实验（Ablation Study）

> 🇬🇧 **Effectiveness of Each Component (RQ1).** In Table 4, we study the impact of maskable stock representation, customizable stock pools, and re-weighting methods. Comparing EarnMore-w/o-M with EarnMore reveals a significant improvement due to the generalized pool-level maskable stock representation. The absence of this representation increases investment risk. Both GSP and CSPs benefit, with CSPs outperforming GSP, suggesting potential for higher returns with focused stock selection.

💬 **各组件的有效性（RQ1）**。表 4 研究了可掩码股票表征、可定制股票池、重加权方法的影响。把 **EarnMore-w/o-M**（去掉掩码表征）与完整 EarnMore 对比，可见泛化的池级可掩码股票表征带来显著提升；缺失该表征会增加投资风险。GSP 和 CSP 都从中受益，且 CSP 优于 GSP——这说明**聚焦式选股**有望带来更高收益。

> 🇬🇧 Comparing the EarnMore-w/o-MR and EarnMore-w/o-M, we find that the re-weighting method can achieve significant improvements in profits by sparsifying portfolios, somehow in increasing the MDD risk metric. Despite reducing the portfolio's diversity may decrease the chances of selecting stocks from various industries and potentially raise risks, especially in CSPs with limited industry variety, focusing on a select few industries can considerably improve returns and offset potential losses due to risk.

💬 对比 **EarnMore-w/o-MR**（去掉掩码表征和重加权）与 EarnMore-w/o-M，可见**重加权**方法通过稀疏化组合显著提升收益，但在一定程度上抬高了 MDD（最大回撤）风险指标。尽管降低组合多样性会减少从不同行业选股的机会、可能抬高风险（尤其在行业种类有限的 CSP 中），但聚焦少数几个行业能大幅提升收益，足以抵消因风险带来的潜在损失。

📌 译注：**w/o** = without（不含）。**w/o-M** = 去掉可掩码股票表征；**w/o-MR** = 同时去掉掩码表征与重加权，是最精简的版本。

**表 4**：EarnMore 消融实验（数值为 ARR / SR / MDD；红=提升，绿=下降，下划线=最优。Δ 为相对 w/o-MR 基准的变化）。

| 股票池 | 模型 | SP500-ARR | SP500-SR | SP500-MDD | DJ30-ARR | DJ30-SR | DJ30-MDD |
|---|---|---|---|---|---|---|---|
| GSP | w/o-MR | 33.2 | 1.49 | 22.7 | 10.5 | 0.71 | 21.5 |
| GSP | w/o-M | 74.8 | 1.89 | 22.5 | 31.4 | 1.14 | 19.4 |
| GSP | **EarnMore** | **97.2** | **2.03** | 28.1 | **47.3** | **1.45** | 21.7 |
| CSP1 | w/o-MR | 32.6 | 1.18 | 25.4 | 14.2 | 0.78 | 20.5 |
| CSP1 | w/o-M | 46.7 | 1.09 | 32.2 | 23.9 | 1.02 | 22.2 |
| CSP1 | **EarnMore** | **122.6** | **2.28** | 25.1 | **53.9** | **1.81** | 22.9 |
| CSP2 | w/o-MR | 8.46 | 0.62 | 28.2 | 11.02 | 0.81 | 20.1 |
| CSP2 | w/o-M | 18.14 | 0.70 | 32.1 | 22.2 | 0.02 | 21.7 |
| CSP2 | **EarnMore** | **110.1** | **2.28** | 25.8 | **43.4** | **1.55** | 24.3 |
| CSP3 | w/o-MR | 27.5 | 1.35 | 21.7 | 5.17 | 0.41 | 22.5 |
| CSP3 | w/o-M | 56.3 | 1.80 | 21.8 | 17.1 | 0.73 | 25.3 |
| CSP3 | **EarnMore** | **93.7** | **2.12** | 24.0 | **43.5** | **1.57** | 21.18 |

> 🇬🇧 **Difficulties with Direct Methods (RQ2).** There are three simple approaches to transition from PM with GSP to PM with CSPs, which are training-from-scratch, fine-tuning and action-masking. The first thing to note that both the training-from-scratch and fine-tuning approaches lack the ability to make real-time adjustments to the stock pool during trading. Nevertheless, we make a comparison between these three methods and EarnMore in terms of DJ30 ARR and SR via SAC, with the metrics representing the average performance over three investment periods. As depicted in Figure 5(a), EarnMore significantly outperforms the other direct methods. It is worth noting that the challenges of fine-tuning and training-from-scratch may be closely related in the context of PM with CSPs, and that action-masking essentially relies on logits that do not accurately reflect the agent's actual decisions.

💬 **直接方法的困难（RQ2）**。从「GSP 下 PM」过渡到「CSP 下 PM」有三种简单做法：从零训练（training-from-scratch）、微调（fine-tuning）、动作掩码（action-masking）。首先要注意，从零训练和微调都**不具备在交易中实时调整股票池**的能力。尽管如此，我们仍通过 SAC 在 DJ30 的 ARR 与 SR 上对比这三种方法与 EarnMore（指标为三个投资期的平均）。如图 5(a)，EarnMore 显著优于其他直接方法。值得注意：在 CSP 下 PM 语境中，微调与从零训练的困难可能密切相关；而动作掩码本质上依赖于「并不能准确反映智能体真实决策」的 logits。

> 🇬🇧 **Efficiency of EarnMore (RQ3).** Our framework is trained only once to meet the demand of various investors for customizable stock pools and individual decision making. As shown in Figure 5(b), we have selected several other methods to compare with EarnMore, and it can be demonstrated that as the number of CSPs increases, the efficiency of our framework shows up.

💬 **EarnMore 的效率（RQ3）**。我们的框架**只训练一次**，即可满足众多投资者对可定制股票池和个性化决策的需求。如图 5(b)，我们选了若干方法与 EarnMore 对比：随着 CSP 数量增加，本框架的效率优势愈发凸显。

**图 5**：(a) 在 DJ30 上对比 EarnMore 与直接方法的性能；(b) 在 DJ30 上对比 EarnMore 与若干方法的时间成本（纵轴为每回合训练耗时，对数刻度）。

---

## 6 结论与未来方向（CONCLUSION AND FUTURE DIRECTION）

> 🇬🇧 This paper introduces a novel RL framework for portfolio management featuring adaptive investor preference and personal decision awareness for customizable stock pools. Maskable stock representation is enhanced by masking and reconstruction process, and a re-weighting method is introduced to improve sparsified portfolios. These improvements yield superior portfolio performance compared to the benchmark methods, as evidenced by various financial criteria. For future research directions, two key areas will be prioritized. Firstly, we will focus on enhancing risk control via risk penalty optimization. Secondly, we aim to create a flexible, open customizable stock pool that allows easy stock addition or removal.

💬 本文提出了一个面向可定制股票池、具备**自适应投资者偏好与个人决策感知**能力的全新 RL 投资组合管理框架。可掩码股票表征通过「掩码—重建」过程得到增强，并引入重加权方法以改进稀疏化的组合。多种金融指标证明，这些改进带来了优于基准方法的组合表现。未来研究将优先关注两个方向：第一，通过**风险惩罚优化**增强风险控制；第二，构建一个灵活、开放、可便捷增删股票的可定制股票池。

---

## 附录 A：评估指标详情（DETAILS OF EVALUATION METRICS）

> 🇬🇧 We compared EarnMore and baselines in terms of 6 financial metrics, including 1 profit criterion, 3 risk-adjusted profit criteria, and 2 risk criteria.

💬 我们用 6 个金融指标对比 EarnMore 与各基线，包括 1 个收益指标、3 个风险调整收益指标、2 个风险指标，定义与公式如下：

> 🇬🇧 • **Annual Rate of Return (ARR)** is the annualized average return of a trading day, calculated as $ARR=\frac{V_T-V_0}{V_0}\times\frac{C}{T}$, where $T$ is the total number of trading days within a year. $V_T$ and $V_0$ represent the final and initial portfolio values.

💬 • **年化收益率（ARR, Annual Rate of Return）**：交易日的年化平均收益，$ARR=\dfrac{V_T-V_0}{V_0}\times\dfrac{C}{T}$，其中 $C=252$（一年的交易日数），$T$ 是回测期内的交易日总数，$V_T$、$V_0$ 分别为最终与初始组合价值。

> 🇬🇧 • **Sharpe Ratio (SR)** measures risk-adjusted returns of portfolios. It is defined as $SR=\frac{\mathbb{E}[r]}{\sigma[r]}$, where $\mathbb{E}[\cdot]$ is the expectation, $\sigma[\cdot]$ is the standard deviation of the return sequence, $r=[\frac{V_1-V_0}{V_0}, \frac{V_2-V_1}{V_1}, \dots, \frac{V_T-V_{T-1}}{V_{T-1}}]^T$ denotes the historical sequence of the return rate.

💬 • **夏普比率（SR, Sharpe Ratio）**：衡量组合的风险调整后收益，$SR=\dfrac{\mathbb{E}[r]}{\sigma[r]}$，其中 $\mathbb{E}[\cdot]$ 为期望，$\sigma[\cdot]$ 为收益率序列的标准差，$r=[\frac{V_1-V_0}{V_0}, \frac{V_2-V_1}{V_1}, \dots, \frac{V_T-V_{T-1}}{V_{T-1}}]^T$ 是历史收益率序列。

📌 译注：夏普比率 = 单位风险（波动）换来的超额收益，是衡量「性价比」最常用的指标，越高越好。

> 🇬🇧 • **Volatility (VOL)** is the variation in an investment's return over time, measured as the standard deviation $\sigma[r]$.
> • **Maximum Drawdown (MDD)** measures the largest loss from any peak to show the worst case. It is defined as $MDD=\max_{i=0}^{T}\frac{P_i-R_i}{P_i}$, where $R_i=\prod_{i=1}^{T}\frac{V_i}{V_{i-1}}$ and $P_i=\max_{i=1}^{T}R_i$.

💬 • **波动率（VOL, Volatility）**：投资收益随时间的变动幅度，用标准差 $\sigma[r]$ 衡量。
• **最大回撤（MDD, Maximum Drawdown）**：从任一峰值到谷底的最大损失，刻画最坏情况，$MDD=\max_{i=0}^{T}\dfrac{P_i-R_i}{P_i}$，其中 $R_i=\prod_{i=1}^{T}\frac{V_i}{V_{i-1}}$（累计净值），$P_i=\max_{i=1}^{T}R_i$（历史峰值）。

> 🇬🇧 • **Calmar Ratio (CR)** compares average annualized return to maximum drawdown, assessing risk-adjusted performance. It is defined as $CR=\frac{\mathbb{E}[r]}{MDD}$.
> • **Sortino Ratio (SoR)** is a risk-adjusted measure that focuses on the downside risk of a portfolio. It is defined as $SoR=\frac{\mathbb{E}[r]}{DD}$, where $DD$ is the standard deviation of negative return.

💬 • **卡玛比率（CR, Calmar Ratio）**：用年化平均收益除以最大回撤来评估风险调整后表现，$CR=\dfrac{\mathbb{E}[r]}{MDD}$。
• **索提诺比率（SoR, Sortino Ratio）**：只关注**下行风险**的风险调整指标，$SoR=\dfrac{\mathbb{E}[r]}{DD}$，其中 $DD$ 是负收益的标准差。

📌 译注：索提诺比率是夏普比率的改良版——它只惩罚「下跌波动」，不惩罚「上涨波动」，更贴合投资者「只怕亏、不怕赚」的心理。

---

## 附录 B：基线方法详情（DETAILS OF BASELINES）

> 🇬🇧 To provide a comprehensive comparison of EarnMore, we select 14 state-of-the-art and representative stock prediction methods of 4 different types.

💬 为全面对比 EarnMore，我们选取 14 个最先进且具代表性的股票预测方法，分 4 类。各基线描述如下：

> 🇬🇧 • **Rule-based Methods:** BLSW [22] is based on mean reversion that buys underperforming stocks and sells outperforming ones. CSM [14] is a momentum strategy that prefers assets with recently strong performance and expects short-term success.

💬 • **基于规则的方法（Rule-based）**：**BLSW** [22] 基于均值回归，买入表现欠佳的股票、卖出表现优异的股票；**CSM** [14] 是动量策略，偏好近期表现强劲的资产、期望短期延续成功。

> 🇬🇧 • **ML-based Methods:** XGBoost [3] leverages Gradient Boosting Decision Tree (GBDT) for accurate predictions in supervised learning tasks. LightGBM [16] is an efficient GBDT with gradient-based one-side sampling and exclusive feature bundling.

💬 • **基于机器学习的方法（ML-based）**：**XGBoost** [3] 利用梯度提升决策树（GBDT）在监督学习任务中做精确预测；**LightGBM** [16] 是一种高效 GBDT，采用基于梯度的单边采样（GOSS）与互斥特征捆绑（EFB）。

> 🇬🇧 • **DL-based Methods:** ALSTM [23] is a Recurrent Neural Network (RNN) that uses an external attention layer to gather information from all hidden states. TCN [1] is a Convolutional Neural Network (CNN) architecture for sequence modeling in time series analysis and natural language processing.

💬 • **基于深度学习的方法（DL-based）**：**ALSTM** [23] 是一种循环神经网络（RNN），用外部注意力层从所有隐藏状态汇聚信息；**TCN** [1] 是用于序列建模的卷积神经网络（CNN）架构，适用于时间序列分析与自然语言处理。

> 🇬🇧 • **RL-based Methods:** PG [30] optimizes policy function while considering risk and market conditions in PM without estimating value function. SAC [11] is an off-policy actor-critic algorithm that optimizes investment strategies in PM using entropy regularization and soft value functions in continuous portfolio action spaces. PPO [24] updates investment policies iteratively to balance exploration and exploitation, ensuring stability and sample efficiency in PM. EIIE [15] is the first work formulating the PM problem as an MDP, and it outperforms traditional PM methods by using CNN for feature extraction and RL for portfolio decisions. SARL [35] proposes a state-augmented RL framework, which leverages the price movement prediction as additional states based on deterministic policy gradient methods. Investor-Imitator (IMIT) [5] shows good performance as a RL-based framework in PM by replicating investor actions. DeepTrader [32] balances risk-return trade-offs by replicating inter-stock relationships and balance RL to model interactions.

💬 • **基于强化学习的方法（RL-based）**：
- **PG** [30]：策略梯度法，在 PM 中考虑风险与市场状况、优化策略函数，但不估计价值函数。
- **SAC** [11]：off-policy 的 actor-critic 算法，在连续组合动作空间中用熵正则与软价值函数优化投资策略。
- **PPO** [24]：迭代更新投资策略以平衡探索与利用，保证 PM 的稳定性与样本效率。
- **EIIE** [15]：首个把 PM 问题形式化为 MDP 的工作，用 CNN 提特征、用 RL 做组合决策，优于传统 PM 方法。
- **SARL** [35]：提出状态增强的 RL 框架，基于确定性策略梯度法，把价格涨跌预测作为额外状态。
- **Investor-Imitator（IMIT）** [5]：通过复制投资者动作，作为 RL 框架在 PM 中表现良好。
- **DeepTrader** [32]：通过刻画股票间关系并平衡 RL 建模交互，来权衡风险—收益。

---

## 附录 C：实现细节（DETAILS OF IMPLEMENTATION）

> 🇬🇧 The dimensions of our state are $(B,N,D,F)$, where $B=128$ represents the batch size, $N$ denotes the number of stocks. For DJ30, $N=28$, and for SP500, $N=420$. $D=10$ represents the number of historical data days, and $F=102$ represents the total number of features, including OHLC, technological indicators, and temporal information. It is worth mentioning that due to the large amplitude of Volume, which is not conducive to RL training, we have removed it from the features.

💬 我们状态的维度为 $(B,N,D,F)$：$B=128$ 是批大小（batch size），$N$ 是股票数（DJ30 为 28，SP500 为 420），$D=10$ 是历史数据天数，$F=102$ 是特征总数（含 OHLC、技术指标、时间信息）。值得一提的是，由于**成交量（Volume）幅度过大、不利于 RL 训练，我们已将其从特征中移除**。

> 🇬🇧 For the encoder, decoder, actor, and critic, each of them consists of 2 layers of MLP with GELU activation function. The all components embedding dimension is 64. All of our experiments were conducted on an Nvidia A6000 GPU. The horizon length was chosen from the options {32, 64, 128, 256}, and we found that 128 yielded the best results. The batch size was set to 128, and the buffer size was $1e5$. The training process consisted of 2000 episodes using the AdamW optimizer. For both the MAEs component and the RL learning component of SAC, we utilized the Mean Squared Error (MSE) Loss function.

💬 编码器、解码器、actor、critic 各由 2 层 MLP（采用 GELU 激活函数）构成。所有组件的嵌入维度均为 64。全部实验在一块 Nvidia A6000 GPU 上完成。回看窗口长度（horizon length）在 {32, 64, 128, 256} 中选择，发现 **128** 效果最佳。批大小设为 128，缓冲区大小（buffer size）为 $1\times10^5$。训练共 **2000 个回合（episode）**，使用 AdamW 优化器。MAE 组件与 SAC 的 RL 学习组件都使用**均方误差（MSE）损失**。

> 🇬🇧 During the grid search for the learning rates of the Actor, Critic, and MAEs component, we tested values within the range $\{1e-3, 1e-4, 1e-5, 1e-6, 1e-7\}$, and found that $1e-5$ yielded the best performance. The scheduler used was the multi-step learning rate scheduler with warm-up technique, starting with an initial learning rate of $1e-8$, which increased to $1e-5$ after 300 episodes, followed by subsequent multiplicative reductions by 0.1 at the 600th, 1000th, and 1400th episodes. The re-weighting parameters are given as $a=0.6, b=0.8, \mu=0.7$, and $\sigma=0.1$. The optimal temperature parameter $T$ is determined to be 0.1 from the set $\{10, 5, 1, 0.5, 0.1, 0.05, 0.01\}$. Default parameters were used for other baselines and all experiments ran with 3 seeds for computing average metrics.

💬 对 Actor、Critic、MAE 组件的学习率做网格搜索时，在 $\{1e{-}3, 1e{-}4, 1e{-}5, 1e{-}6, 1e{-}7\}$ 中测试，发现 **$1e{-}5$** 最佳。学习率调度器采用带 **warm-up（预热）** 的多步调度：初始学习率 $1e{-}8$，在第 300 回合升至 $1e{-}5$，随后在第 600、1000、1400 回合分别乘以 0.1 衰减。重加权参数为 $a=0.6, b=0.8, \mu=0.7, \sigma=0.1$。最优温度参数 $T$ 在 $\{10, 5, 1, 0.5, 0.1, 0.05, 0.01\}$ 中确定为 **0.1**。其他基线用默认参数，所有实验均以 3 个种子计算平均指标。

---

## 附录 D：训练与推理阶段伪代码（PSEUDOCODE FOR TRAINING AND INFERENCE）

> 🇬🇧 In this section, we present the detailed pseudocode for the training and testing phases of EarnMore. This includes the training phase with customizable stock pools simulated through masked token, as well as the inference phase involving an investor-customized target stock pool.

💬 本节给出 EarnMore 训练与测试阶段的详细伪代码：训练阶段通过掩码 token 模拟可定制股票池；推理阶段则使用投资者定制的目标股票池。

### 算法 1：EarnMore 的训练（Training of EarnMore）

```
输入（Require）：全局股票池 U
输出（Ensure）：参数 θ1, θ2, θe, θc, θenc, θdec, φ

  s_t = [P_t, Y_t, D_t]                       ⊳ 初始化输入数据
  θ̃1 = θ1, θ̃2 = θ2                            ⊳ 初始化目标网络权重
  D = Φ                                       ⊳ 初始化空的回放缓冲区(replay buffer)

  for 每次迭代、每个环境步 do
     l_s = ψe(D_t; θe) + ψc(P_t, Y_t; θc)     ⊳ 股票级嵌入
     r = g(r; μ, σ, a, b)                     ⊳ 采样掩码率
     l̃_s = η_mo(l_s, r)                       ⊳ 掩码操作
     l_p = ψenc(l̃_s; θenc)                    ⊳ 池级嵌入
     ρ_t = η_mf(l_p, m)                       ⊳ 可掩码股票表征
     a_t ∼ Re(π_φ(a_t | ρ_t), T)              ⊳ 从(重加权后的)策略采样动作
     s_{t+1} = p(s_{t+1} | s_t, a_t)          ⊳ 采样状态转移
     D ← D ∪ {(s_t, a_t, ρ_t, r(s_t,a_t), s_{t+1})}   ⊳ 存储转移
  end for

  for 每个梯度步 do
     θ_i ← θ_i − λ_Q ∇̂_{θi} J_Q(θ_i),  i∈{1,2}   ⊳ 更新 Q 网络
     φ ← φ − λ_π ∇̂_φ J_π(φ)                       ⊳ 更新策略网络
     α ← α − λ_α ∇̂_α J(α)                          ⊳ 调节 alpha
     θ̄_i ← τθ_i + (1−τ)θ̄_i,  i∈{1,2}              ⊳ 更新目标网络(软更新)
     θe, θc, θenc, θdec ← λ ∇̂ J(θe,θc,θenc,θdec)   ⊳ 更新可掩码表征
  end for

  return θ1, θ2, θe, θc, θenc, θdec, φ          ⊳ 优化后的参数
```

📌 译注：训练时**掩码率 $r$ 是随机采样的**（式 3），意味着每一步都在模拟一个不同的 CSP——这正是「一次训练、通用于各种 CSP」的关键。$\tau$ 是目标网络软更新系数。

### 算法 2：EarnMore 的推理（Inference of EarnMore）

```
输入（Require）：全局股票池 U，可定制股票池 CSPs = {C_t | t=1,2,...,T}
输出（Ensure）：各 CSP 的组合 {W_t | t=1,2,...,T}

  for 每个时间步 t in {1,2,...,T} do
     s_t = [P_t, Y_t, D_t]                    ⊳ 初始化状态
     l_s = ψe(D_t; θe) + ψc(P_t, Y_t; θc)     ⊳ 股票级嵌入
     根据 CSP_t 初始化掩码索引 M               ⊳ 用投资者指定的池来确定掩码
     l̃_s = η_mo(l_s, M)                       ⊳ 掩码操作
     l_p = ψenc(l̃_s; θenc)                    ⊳ 编码池级嵌入
     ρ_t = η_mf(l_p, m)                       ⊳ 可掩码股票表征
     a_t = Re(π_φ(a_t | ρ_t), T)              ⊳ 从策略预测动作
     W_t ← a_t
  end for
```

📌 译注：训练与推理的关键差异——训练时掩码索引由**随机采样**得到（模拟各种 CSP）；推理时掩码索引 $M$ 由**投资者实际选定的 CSP** 决定（哪些股票不在目标池就掩码哪些）。这就是「投资者定制」落地的方式。

---

## 附录 E：与基线的对比详情（DETAILS OF COMPARISON WITH THE BASELINES）

> 🇬🇧 In this section, we conduct a comparative analysis of our method EarnMore in comparison to 14 benchmark models. The analysis is based on 6 key performance metrics applied to SP500 and DJ30 datasets. Specifically, these metrics include Average Rate of Return (ARR) as an indicator of portfolio performance, along with risk-adjusted measures like Sharpe Ratio (SR), Calmar Ratio (CR), and Sortino Ratio (SoR). Additionally, we consider risk-related metrics, Maximum Drawdown (MDD), and Volatility (VOL), to evaluate the risk implications of the strategies. The backtesting was conducted on 3 date splits, each with 3 seed values, and the reported metrics represent the averages derived from 3×3 experiments.

💬 本节对 EarnMore 与 14 个基准模型做对比分析，基于应用于 SP500 与 DJ30 数据集的 6 个关键性能指标：ARR（组合表现）、风险调整指标 SR/CR/SoR，以及风险指标 MDD 与 VOL。回测在 3 个日期划分上进行，每个划分用 3 个种子，所报指标为 3×3 实验的平均。**表 5** 给出 GSP 与 3 个 CSP 上各方法的完整 6 指标对比（数值详见原文，EarnMore 在所有数据集上的收益类指标均为最佳，仅在风险指标上有轻微妥协）。

> 🇬🇧 As shown in Table 5, our framework performance is evaluated in comparison to all other methods. EarnMore stands out by significantly improving its return potential across all datasets, while maintaining a minimal loss in risk control. In Figure 6, we have included comparative line diagrams of EarnMore and several other methods in terms of cumulative returns. It is shown that EarnMore demonstrates the best profit potential across all datasets.

💬 如表 5 所示，EarnMore 在所有数据集上都显著提升了收益潜力，同时在风险控制上仅有极小损失。图 6 给出 EarnMore 与若干方法在累计收益上的对比折线图，显示 EarnMore 在所有数据集上都展现出最佳的盈利潜力。

> 🇬🇧 As depicted in Figure 6(a) and 6(b), in October 2018, due to the impact of the U.S. monetary policy and economic confrontation, investor confidence in the stock markets is challenged, leading to a downward trend in the U.S. stock market during this period. EarnMore is impacted, resulting in a partial loss of returns, but it is anticipated to recover quickly, maintaining its overall superiority over other methods. The global COVID-19 pandemic reached its highest point between February 14 and March 20, 2020... with the SP500 and DJ30 indices dropping by 31.81% and 34.78%, respectively. As depicted in Figures 6(c) and 6(d), it's clear that EarnMore is much less affected in terms of returns compared to the other methods. Moreover, it continues to gain profits after the market starts to recover.

💬 如图 6(a)(b)，2018 年 10 月，受美国货币政策与经济对抗影响，投资者对股市的信心受挫，美股在此期间走低。EarnMore 受到冲击、部分收益回吐，但预期能快速恢复，整体仍优于其他方法。全球新冠疫情在 2020 年 2 月 14 日至 3 月 20 日达峰，SP500 与 DJ30 指数分别下跌 31.81% 与 34.78%。如图 6(c)(d)，EarnMore 的收益所受影响明显小于其他方法，且在市场开始恢复后继续获利。

> 🇬🇧 Starting in 2021, the U.S. stock market began its recovery, with the SP500 and DJ30 indices generally showing an upward trend. From Figures 6(e) and 6(f), it's evident that EarnMore, compared to other methods, is better at identifying stocks with upward momentum, maximizing returns. In March 2022, due to geopolitical conflicts, there was a brief downturn in the U.S. stock market. During this time, EarnMore is somewhat affected, showing noticeable changes in returns in the SP500, but still demonstrating a strong upward trend in the DJ30.

💬 进入 2021 年，美股开始复苏，SP500 与 DJ30 总体上行。由图 6(e)(f) 可见，相比其他方法，EarnMore 更擅长识别具有上行动量的股票、最大化收益。2022 年 3 月，受地缘政治冲突影响，美股短暂下行；此间 EarnMore 受到一定影响，在 SP500 上收益有明显变化，但在 DJ30 上仍展现强劲上行趋势。

**图 6**：SP500 与 DJ30 在 GSP 及各 CSP 上的（累计收益）表现。

---

## 致谢（ACKNOWLEDGMENTS）

> 🇬🇧 This project is supported by the National Research Foundation, Singapore under its Industry Alignment Fund – Pre-positioning (IAF-PP) Funding Initiative. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore.

💬 本项目受新加坡国家研究基金会（National Research Foundation, Singapore）「产业对接基金—前瞻布局（IAF-PP）」资助计划支持。本文所表达的任何观点、发现、结论或建议均为作者个人立场，不代表新加坡国家研究基金会的观点。

---

> **参考文献（References）**：原文共 36 篇，涉及 MAE [12]、MAGE [17]、PatchTST [21]、TimesNet [33]、SAC [11]、PPO [24]、Qlib [34] 等关键文献。完整列表见原 PDF 第 9–10 页，此处从略。




