## PAGE 1

Reinforcement Learning with Maskable Stock Representation for
Portfolio Management in Customizable Stock Pools
Wentao Zhang Yilei Zhao Shuo Sun∗
Nanyang Technological University Zhejiang University Nanyang Technological University
Singapore China Singapore
wt.zhang@ntu.edu.sg yilei_zhao@zju.edu.cn shuo003@e.ntu.edu.sg

Jie Ying Yonggang
Nanyang Technological University Nanyang Technolog
Singapore Singapo
ying0024@e.ntu.edu.sg ying0024e.ntu
Xinrun Wang
Nanyang Technological University N
Singapore
xinrun.wang@ntu.edu.sg
ABSTRACT
Portfolio management (PM) is a fundamental financial trading task,
which explores the optimal periodical reallocation of capitals into
different stocks to pursue long-term profits. Reinforcement learning
(RL) has recently shown its potential to train profitable agents for
PM through interacting with financial markets. However, existing
work mostly focuses on fixed stock pools, which is inconsistent
with investors’ practical demand. Specifically, the target stock pool
of different investors varies dramatically due to their discrepancy
on market states and individual investors may temporally adjust
stocks they desire to trade (e.g., adding one popular stocks), which
lead to customizable stock pools (CSPs). Existing RL methods re-
quire to retrain RL agents even with a tiny change of the stock pool,
which leads to high computational cost and unstable performance.
To tackle this challenge, we propose EarnMore, a rEinforcement
leARNing framework with Maskable stOck REpresentation to han-
dle PM with CSPs through one-shot training in a global stock pool
(GSP). Specifically, we first introduce a mechanism to mask out the
representation of the stocks outside the target pool. Second, we
learn meaningful stock representations through a self-supervised
masking and reconstruction process. Third, a re-weighting mech-
anism is designed to make the portfolio concentrate on favorable
stocks and neglect the stocks outside the target pool. Through ex-
tensive experiments on 8 subset stock pools of the US stock market,
we demonstrate that EarnMore significantly outperforms 14 state-
of-the-art baselines in terms of 6 popular financial metrics with
over 40% improvement on profit. Code is available in PyTorch1.
∗Corresponding Author
1https://github.com/DVampire/EarnMore
This work is licensed under a Creative Commons Attribution
International 4.0 License.
WWW ’24, May 13–17, 2024, Singapore, Singapore
© 2024 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-0171-9/24/05.
https://doi.org/10.1145/3589334.3645615
4202
beF
72
]MP.nif-q[
4v10801.1132:viXra

ggang Xie Zitao Song
nological University Nanyang Technological University
ngapore Singapore
4e.ntu.edu.sg zitao.song@ntu.edu.sg
Bo An
Nanyang Technological University and Skywork AI
Singapore
boan@ntu.edu.sg
CCS CONCEPTS
• Information systems → Data mining; • Computing method-
ologies → Machine learning; • Applied computing → Elec-
tronic commerce.
KEYWORDS
Portfolio Management, Reinforcement Learning, Representation
Learning
ACM Reference Format:
Wentao Zhang, Yilei Zhao, Shuo Sun, Jie Ying, Yonggang Xie, Zitao Song,
Xinrun Wang, and Bo An. 2024. Reinforcement Learning with Maskable
Stock Representation for Portfolio Management in Customizable Stock
Pools. In Proceedings of the ACM Web Conference 2024 (WWW ’24), May
13–17, 2024, Singapore, Singapore. ACM, New York, NY, USA, 12 pages.
https://doi.org/10.1145/3589334.3645615
1 INTRODUCTION
The stock market, which involves over $90 trillion market cap-
italization, has attracted the attention of innumerable investors
around the world. Portfolio management, which dynamically al-
locates the proportion of capitals among different stocks, plays
a key role to make profits for investors. Reinforcement learning
(RL) has recently become a promising methodology for financial
trading tasks due to its stellar performance on solving complex
sequential decision-making problems such as Go [25] and matrix
multiplication [9]. In fact, RL has achieved significant success in
various quantitative trading tasks such as algorithmic trading [4],
portfolio management [31], order execution [8] and market mak-
ing [26]. To apply RL methods for PM, existing work [31, 32, 35]
always train RL agents to make investment decisions based on a
fixed stock pool2, which requires retraining with high computa-
tional cost when investors need to change their target stock pools.
Furthermore, investors are unable to engage in or influence the
agent’s decision-making process, as it is uncontrollable for them.
2The fixed stock pool in existing work is the global stock pool (GSP) under our setting.


## PAGE 2

WWW ’24, May 13–17, 2024, Singapore, Singapore
Global Stock Pool Investor Customizable Stock Pools
CSP1
X c
CSP2
CSP3
Cash
0.4 0.3 0.0 0.2 0.1
EarnMore
Portfolios
0.1 0.4 0.1 0.1 0.3
0.0 0.0 0.4 0.6 0.0
Figure 1: Overview of portfolio management by EarnMore in
customizable stock pools (CSPs).
Investors and agents may work better with each other in collab-
oration. On one hand, investors have a limited understanding of
stock trends, hidden connections among stocks, and the overall
market dynamics. On the other hand, the RL agent may make oc-
casional decision errors and lacks the human capacity to acquire
information in diverse ways. All of these factors can result in di-
minished or suboptimal returns. For instance, if a certain stock is
delisted in the trading process, the agent may still consider it as a
candidate for investment allocation, potentially leading to a loss of
returns. In a different scenario, when a stock exhibits significant
future profit potential, investors may want to add it to the target
pools to maximize their returns. Therefore, We introduce the task
of portfolio management with customizable stock pools (CSPs) to
meet the mentioned demands and address the above issues.
As shown in Fig. 1, the demand of PM with customizable stock
pools (CSPs) is ubiquitous for different financial practitioners in
real-world trading scenarios. For instance, stock brokerages need to
offer real-time portfolio suggestions for millions of investors with
diversified preferences on stock pools. The investors also desire
to adapt their stock pools based on different market conditions
from time to time. Therefore, an RL algorithm with the ability to
handle PM with CSPs is urgently needed. There are 3 straightfor-
ward methods for implementing PM in CSPs: i) training an agent
from scratch on each individual CSP. However, randomly picking 5
stocks from 30 for a CSP leads to 140k+ combinations, which is un-
feasible in practice; ii) adjusting the output dimensions of the policy
network and then fine-tune the agents by mapping the action space
of the GSP to CSP. Fine-tuning agent on PM with contiguous action
space might be equally time-consuming as starting from scratch,
making it impractical in reality; iii) using action-masking method
by subtracting a large constant from policy network logits that
represent unfavorable stocks to reduce their investment allocation.
Human-induced decision changes in this method do not truly ex-
press the agent’s real decision-making, as it lacks awareness of CSP
stocks, significantly impacting effectiveness. Although the existing
work considers the dynamics of the stock pool, such as the study
by Betancourt et al. [2] focuses on the changing number of assets,
training an actor and critic for each stock is resource-intensive and
time-consuming. Additionally, this approach disallows for changes
in the stock pool in terms of number of stocks and composition dur-
ing the investment process, which would suffer the same limitations
if implemented by action-masking.

Wentao Zhang et al.
To handle PM with CSPs, we face the following two major chal-
lenges: i) how to learn unified representations that are aligned for
stock pools with different sizes and stocks; ii) how to guide RL
agents to construct portfolios that concentrate on most favorable
stocks and neglect the stocks outside the target pool. To tackle
these two challenges, we have designed an RL framework that
called EarnMore with one-time training that takes into account
the distinct investment preferences of each investor and invests in
various CSPs. This framework allows for dynamic adjustments to
the pool in the investment process, contributing to a more tailored
and effective portfolio. Our contributions are four-fold:
• We introduce a learnable masked token to represent unfavorable
stocks, which enables the unified representation of stock pools
with different sizes and stocks.
• We derive meaningful embeddings using a self-supervised mask-
ing and reconstruction process that captures stock relationships.
• We propose a re-weighting mechanism to rescale the distribution
of portfolios to make it concentrate on favorable stocks and
neglect stocks outside the target pool.
• Experiments on 8 subset stock pools of the US stock market
demonstrate the superiority of EarnMore over 14 baselines in
terms of 6 popular financial metrics with one-time training.
2 RELATED WORK
2.1 Portfolio Management
Portfolio management is an essential aspect of investment and in-
volves a strategic allocation of resources to achieve optimal returns
and avoid risks simultaneously. There are two commonly used tradi-
tional rule-based methods, i.e., mean reversion [22] and momentum
[13]. The former buys low-priced stocks and sells high-priced ones,
whereas the latter relies on recent performance with the expec-
tation that trends will continue. Cross-Sectional Momentum [14]
and Time-Series Momentum [20] are two classical momentum trad-
ing methods. However, traditional rule-based methods are difficult
to capture fleeting patterns in changing market conditions and
perform well only in specific scenarios [4].
In the past few years, advanced prediction-based methods have
significantly surpassed traditional rule-based methods in perfor-
mance. These methods treat PM as a supervised learning task and
predict future returns (regression) or price movements (classifica-
tion). Then, the heuristic strategy generator allocates asset invest-
ments based on the prediction results [36] 3. Specifically, prediction-
based methods can be categorized into two kinds, machine learning
models like XGBoost [3] and LightGBM [16], along with deep learn-
ing models such as ALSTM [23] and TCN [1]. However, the volatil-
ity and noisy nature of the financial market makes it extremely
difficult to accurately predict future prices [7]. Furthermore, the
gap between prediction signals and profitable trading actions [10]
is difficult to bridge. Therefore, prediction-based methods do not
perform satisfactorily in general.
Recent years have witnessed the successful marriage of rein-
forcement learning and portfolio management [28] due to its ability
to handle sequential decision-making problems.
3For instance, top-k [36] assets are chosen for investment allocation, with the propor-
tion determined by the forecasted future returns ranked by magnitude.


## PAGE 3

Reinforcement Learning with Maskable Stock Representation for Portfolio Management in C
EIIE [15] utilizes the convolutional neural network for feature
extraction and RL for decision-making. The Investor-Imitator [5]
framework demonstrates its utility in financial investment by emu-
lating investor actions. SARL [35] leverages the price movement
prediction as additional states based on deterministic policy gra-
dient methods. DeepTrader [32] dynamically balances risk-return
with market indicators and utilizes a unique graph structure to
generate portfolios. HRPM [31] presents a hierarchical framework
addressing long-term profits and considers price slippage as part of
the trading cost. DeepScalper [29] combines intraday trading and
risk-aware tasks to capture the investment opportunities 4.
2.2 Masked Autoencoders
Masked Autoencoders (MAEs) [12] are neural networks employed
in self-supervised learning to obtain effective embeddings. An au-
toencoder is designed to learn a representation for a set of data,
which is initially used for self-supervised learning in image, video
and audio. Recently, they have been extensively studied in the field
of time series prediction. PatchTST[21] enhances multivariate time
series forecasting through self-self-supervised learning. It achieves
this by partitioning time series data into patches and utilizing sep-
arate channels for univariate time series. This approach boosts
memory efficiency and improves the model’s ability to capture his-
torical patterns. Several notable works have used MAEs to learn
time series representations, such as SimMTM [6] and Ti-MAE [18].
In financial markets, which tend to have lower signal-to-noise
ratios than typical time series data. Leveraging MAEs for self-
supervised learning, we efficiently reduce data dimensionality, filter
noise, and highlight essential information. MAEs uncover hidden
stock relationships and represent investor-unfavorable stocks with
masked tokens, improving investor-agent interactions in PM.
3 PRELIMINARIES
In this section, we present definitions and formulas for necessary
terms in PM. Next, we provide a Markov Decision Process (MDP)
model for portfolio management with CSPs.
3.1 Definitions and Formulas
Definition 1. (OHLCV). Open-High-Low-Close-Volume is a type
of bar chart obtained from the financial market. The OHLCV of stock 𝑖
at time 𝑡 is denoted as P𝑖,𝑡 = [𝑝 𝑖 𝑜 ,𝑡 , 𝑝 𝑖 ℎ ,𝑡 , 𝑝 𝑖 𝑙 ,𝑡 , 𝑝 𝑖 𝑐 ,𝑡 , 𝑣 𝑖,𝑡 ], where 𝑝 𝑖 𝑜 ,𝑡 , 𝑝 𝑖 ℎ ,𝑡 ,
𝑝 𝑖 𝑙 ,𝑡 , 𝑝 𝑖 𝑐 ,𝑡 and 𝑣 𝑖,𝑡 are open, high, low, close prices and volume.
Definition 2. (Technical indicators). A technical indicator in-
dicates a feature calculated by a formulaic combination of the his-
torical OHLCV. We denote the technical indicator vector at time 𝑡:
Y𝑡 = [𝑦 𝑡 1, 𝑦 𝑡 2, ..., 𝑦 𝑡 𝐾 ] 𝑇 ∈ R𝐾 . For 𝑘 ∈ [1, 𝐾], 𝑦 𝑡 𝑘 is represented as
𝑦 𝑡 𝑘 = 𝑓 𝑡 𝑘 (𝑥 𝑡 −𝐷+1 , ..., 𝑥 𝑡 −1 , 𝑥 𝑡 |𝜃𝑘 ), where 𝐷 denotes past time steps up
to 𝑡, and 𝜃𝑘 are hyperparameters for indicator 𝑘.
Definition 3. (Portfolio). A portfolio is a combination of finan-
cial assets, denoted as W𝑡 = [𝑤 𝑡 0, 𝑤 𝑡 1, ..., 𝑤 𝑡 𝑁 ] ∈ R𝑁 +1, with 𝑁 + 1
assets, including 1 risk-free cash and 𝑁 risky stocks. Each asset 𝑖 is
assigned a weight 𝑤 𝑡 𝑖 representing its portfolio proportion, subject to
the constraint that (cid:205) 𝑖 𝑁 =0 𝑤 𝑡 𝑖 = 1 for full investment.
4HRPM and Deepscalper are excluded from the experimental section due to the limited
order book being introduced as extra data in addition to Kline prices.

ment in Customizable Stock Pools WWW ’24, May 13–17, 2024, Singapore, Singapore
Definition 4. (Portfolio Value). Portfolio value at time step 𝑡,
denoted as 𝑉 𝑡 , represents the sum of individual asset values in the
portfolio, 𝑉 0 represents the initial cash and 𝑉 𝑡 calculated using stock
closing prices through the following formula:
𝑉𝑡 = 𝑤 𝑡 0𝑉𝑡−1 + (1 − 𝑤 𝑡 0)𝑉𝑡−1 (1 + ∑︁ 𝑖 𝑁 =1 𝑤 𝑡 𝑖 𝑝 𝑖 𝑐 ,𝑡 𝑝 − 𝑖 𝑐 ,𝑡 𝑝 − 𝑖 𝑐 1 ,𝑡−1 ). (1)
3.2 Problem Formulation
We model portfolio management as a Markov Decision Process
(MDP) and provide a detailed description of the MDP modeling
process for portfolio management with CSPs in this section.
MDP Formulation for PM. We formulate PM as an MDP fol-
lowing a standard RL scenario, where an agent (investor) interacts
with an environment (the financial markets) in discrete time to
make actions (investment decisions) and get rewards (profits). In
this work, the objective is to maximize the final portfolio value
within a long-term investment time horizon. We formulate PM as
an MDP, which is constructed by a 5-tuple (S, A,𝑇, 𝑅,𝛾). Specifi-
cally, S is a finite set of states. A is a finite set of actions. The state
transition function 𝑇 : S × A × S → [0, 1] encapsulates transition
probabilities between states based on chosen actions. The reward
function 𝑅 : S × A → 𝑅 quantifies the immediate reward of tak-
ing an action in a state. The discount factor is 𝛾 ∈ [0, 1). A policy
𝜋 : S × A → [0, 1] assigns each state 𝑠 ∈ S a distribution over
actions, where 𝑎 ∈ A has probability 𝜋 (𝑎|𝑠).
MDP Formulation for PM with CSPs. Existing work [32, 35]
focus on the GSP and lacks formal modeling in the broader context
of CSPs. We denote a GSP as 𝑈 , consisting of 𝑁 individual stocks.
By randomly masking some stocks that investors are unfavorable
to, it can generate a diverse set of CSPs, which are subsets of 𝑈 .
We define a sub-pool of GSP 𝑈 that masks 𝑁 ∗ stocks at time step
𝑡 to create CSP 𝐶 𝑡 , where |𝐶 𝑡 | = 𝑁 − 𝑁 ∗. We model an MDP for
PM with CSP 𝐶 𝑡 using maskable stock representation5. The widely
used MDP formulation in existing work is a special case in our PM
with CSPs formulation when 𝐶 𝑡 = 𝑈 . The details of the PM with
CSPs as an MDP are set as follows:
• State. The GSP stocks time series are composed of three compo-
nents: historical OHLC prices P𝑡 , classical technical indicators
Y𝑡 , and temporal information D𝑡 . We denote the feature of GSP
𝑈 during the historical 𝐷 time steps as X𝑡 = [P𝑡 , Y𝑡 , D𝑡 ] =
[𝑥 𝑡 −𝐷+1 , 𝑥 𝑡 −𝐷+2 , ..., 𝑥 𝑡 ] ∈ R𝑁 ×𝐷×𝐹 , 𝐹 represents the feature di-
mension. After masked 𝑁 ∗ stocks, the feature of CSP 𝐶 𝑡 is de-
noted as X𝑡 ∗ = [𝑥 𝑡 1, 𝑥 𝑡 2, . . . , 𝑥 𝑡 𝑁 −𝑁 ∗] ∈ R(𝑁 −𝑁 ∗)×𝐷×𝐹 . To restore
the GSP feature dimension, the process entails duplicating and
populating with a learnable masked token [𝑀]. Then, the state
is represented as 𝑠 𝑡 = [𝑥 𝑡 1, 𝑥 𝑡 2, . . . , 𝑥 𝑡 𝑁 −𝑁 ∗, [𝑀], [𝑀], . . . , [𝑀]] ∈
R𝑁 ×𝐷×𝐹 , where the number of [𝑀] is 𝑁 ∗.
• Action. Given the state, the action of the agent at time step 𝑡
can be entirely represented by the portfolio vector 𝑎 𝑡 = W𝑡 =
[𝑤 𝑡 0; 𝑤 𝑡 1, . . . , 𝑤 𝑡 𝑁 −𝑁 ∗ ; 𝑤1 [𝑀 ],𝑡 , 𝑤2 [𝑀 ],𝑡 , . . . , 𝑤𝑁 [𝑀 ∗ ],𝑡 ] ∈ R𝑁 +1. The
proportion of cash retained is represented by 𝑤 𝑡 0, [𝑤 𝑡 1, . . . , 𝑤 𝑡 𝑁 −𝑁 ∗]
denotes the proportion of 𝑁 − 𝑁 ∗ investor favorable stocks,
[𝑤1 , 𝑤2 , . . . , 𝑤𝑁 ∗ ] denotes the proportion of 𝑁 ∗ in-
[𝑀 ],𝑡 [𝑀 ],𝑡 [𝑀 ],𝑡
vestor unfavorable stocks.
5Detailed description of the maskable stock representation can be found in section 4.1.


## PAGE 4

WWW ’24, May 13–17, 2024, Singapore, Singapore
• Drawing on prior research [19, 27, 28], for each trading time step
𝑡, the reward 𝑟 𝑡 is the change of portfolio value 𝑟 𝑡 = 𝑉 𝑡 − 𝑉 𝑡 −1.
4 EARNMORE
As shown in Figure 2, we present a reinforcement learning frame-
work called EarnMore with one-time training on a GSP, enabling
invest on various CSPs and achieving optimal investment portfo-
lios. With this framework, investors have the flexibility to invest in
stock pools aligned with their individual preferences. Furthermore,
during the trading process, these stock pools can be dynamically
adapted to construct more efficient and tailored portfolios, aligning
with investors’ current decisions. EarnMore consists of three main
components: i) a unified approach for representing customized
stock pools with different sizes and stocks, which we call maskable
stock representation (§4.1); ii) reinforcement learning optimization
procedures for PM with CSPs (§4.2); iii) a re-weighting mechanism
that concentrates on favorable stocks and neglects unfavorable ones
to rescale the distribution of portfolios (§4.3).
4.1 Maskable Stock Representation for CSPs
Consistent representation is crucial for CSPs with different sizes
and stocks before portfolio decisions. If only the stocks within
the CSP are embedded into the agent for decision-making, and
discarding the masked-out stocks will lead to three issues. Firstly,
the action dimension of agent cannot adapt to different CSP sizes.
Secondly, the agent may perform poorly or even fail due to the
inability to distinguish between the different CSPs when dealing
with CSPs of the same size but different stocks. Finally, discarding
unfavorable stocks may lead to a loss of relationships between
stocks, which will negatively impact performance. To address them,
we introduce a maskable stock representation that identifies the
position of each stock within the GSP. We employ two levels of
stock representation, which are stock-level in Module (𝑎) and pool-
level constructed through masking and reconstruction in Module
(𝑏). This process reveals hidden connections between stocks, and
our maskable stock representation is based on the pool-level.
Learning for Stock-level Representation. In Equation 2, we
exploit stock features (prices and technical indicators) and tem-
poral characteristics for stock-level representation. Following the
approach in Timesnet [33], we employ 1D convolution to produce
dense embeddings for stock features and utilize an embedding layer
to handle sparse temporal features. The final stock-level repre-
sentation is formed by the summation of these dense and sparse
embeddings. It is defined as follows:
𝑙𝑠 (X𝑡 ) = 𝜓𝑒 (Dt;𝜃𝑒 ) + 𝜓𝑐 (P𝑡, Y𝑡 ;𝜃𝑐 ), (2)
where 𝜓 𝑒 , 𝜓 𝑐 denote the embedding layer and 1D convolutional
layer, and 𝜃 𝑒 , 𝜃 𝑐 are their learnable parameters respectively.
Learning for Pool-level Representation. Stock-level repre-
sentation describes the vertical time series information within each
individual stock without capturing the horizontal inter-stock rep-
resentation. In our PM with CSPs environment, directly masking
certain stocks could potentially result in losing important and valu-
able connections between stocks when using them as stock repre-
sentations. To address this limitation, we introduce the pool-level
representation, which strengthens the connections between stocks

Wentao Zhang et al.
in the GSP through the masking and reconstruction process. No-
tably, we employ stock-level embedding as the local embedding
to replace the patching embedding for historical data employed in
MAEs [12] or PatchTST [21].
During training, we utilize the adaptive masking strategy devel-
oped by MAGE [17] to simulate various CSPs with varying stock
numbers and compositions, which improves the representational
capability of the pool-level embedding and unifies the high and low-
masking-ratio stock pools under the same training framework. We
sample a masking ratio 𝑟 from a truncated Gaussian distribution:
𝑟 − 𝜇 𝑏 − 𝜇 𝑎 − 𝜇
𝑔(𝑟; 𝜇, 𝜎, 𝑎,𝑏) = 𝜑 ( )/(Φ( ) − Φ( )), (3)
𝜎 𝜎 𝜎
where 𝜑 (·) is the probability density function of the standard nor-
mal distribution, Φ(·) is its cumulative distribution function, 𝑎 and
𝑏 are the lower and upper bounds.
In Equation 4, the process for constructing maskable stock rep-
resentation is outlined through an encoder and decoder procedure.
The process starts with the encoder phase, a random masking ra-
tio 𝑟 is sampled and then is used to mask a subset of stock-level
embeddings selectively using the masking operation 𝜂 𝑚𝑜 . Only
the unmasked embeddings are retained and subsequently fed into
the encoder 𝜓 𝑒𝑛𝑐 to extract latent embeddings. During the decoder
phase, the latent embeddings are filled to the number of stock-level
embeddings using a learnable masked token called 𝑚 via the mask-
filled operation 𝜂 𝑚𝑓 . Finally, the decoder 𝜓 𝑑𝑒𝑐 is used to reconstruct
the price P˜ 𝑡 of masked stocks:
𝑙𝑝 (X𝑡 ) =𝜓𝑒𝑛𝑐 (𝜂𝑚𝑜 (𝑙𝑠 (X𝑡 ),𝑔(𝑟; 𝜇, 𝜎, 𝑎,𝑏));𝜃𝑒𝑛𝑐 )
𝜌 (X𝑡,𝑚) = 𝜂𝑚𝑓 (𝑙𝑝 (X𝑡 ),𝑚) (4)
P˜𝑡 = 𝜓𝑑𝑒𝑐 (𝜌 (X𝑡,𝑚);𝜃𝑑𝑒𝑐 ),
where 𝜓 𝑒𝑛𝑐 , 𝜓 𝑑𝑒𝑐 denote the Encoder and Decoder, 𝜃 𝑒𝑛𝑐 , 𝜃 𝑑𝑒𝑐 are
their respective learnable parameters.
After filling in the masked token, we refer to the resulting latent
embedding as maskable stock representation, and we abbreviate
𝜂 𝑚𝑓 (𝑙 𝑝 (X𝑡 ), 𝑚) as 𝜌 (X𝑡 , 𝑚), which will be used as state for port-
folio decision making in the reinforcement learning process. The
masked token makes the agent sense which stocks are unfavor-
able to the investor, thus catering to the investor’s preferences and
personal decisions, and enabling collaboration between the agent
and investor, who has access to various sources of information and
have some expectation of the future direction of specific stocks. It is
important to mention that we retain the [CLS] token to understand
how cash is allocated. This token can capture the overall sequence
representation and preserve global sequence information when
decoding, which is exactly what we need.
4.2 RL Optimization for PM with CSPs
Our reinforcement learning training process is based on the Soft
Actor-Critic (SAC) [11]. There are two main components called
Actor and Critic in RL optimization. The Actor utilizes the latent
embeddings populated by masked tokens to generate actions that
indicate the allocation ratios for cash and individual stocks, and
the Actor is aware of masked token for unfavorable stocks and will
avoid allocating them during decision-making. The Critic evaluates
portfolio performance using populated latent embeddings with
masked tokens and actions that the Actor generates. This evaluation


## PAGE 5

Reinforcement Learning with Maskable Stock Representation for Portfolio Management in C
Selected by Direct Masking (b) Extract Pool
GSP Investor Operation
[CLS]
CSP
Encoder
Date Indicator Price
𝑫 𝒀 𝑷
! ! !
Concat
Embed Conv
Masking
Operation C
Summation
(a) Extract Stock-level Embedding in GSP
Figure 2: The overall architecture of EarnMore. Module (a) is used t
masking and reconstruction process to learn pool-level embeddin
provides a scoring mechanism that guides the learning process and
helps optimize the portfolio management strategy.
We utilize two contrasting strategies to penalize the actor-critic
from assigning weights to the masked stocks. The first method
involves adding an additional supervised loss to the actor output’s
portfolios. The second method involves adding a penalty term to
the TD error when there is a non-zero investment portfolio for
masked stocks. The first approach yields better results because the
supervised loss affects the actor portfolios in a more direct way.
Optimization for Q-Value Network. We use maskable stock
representation 𝜌 (X𝑡 , 𝑚) defined in Equation 4 instead of raw market
data as input states 𝜌 for Actor and Critic. Let 𝑄 𝜃 (𝑠, 𝑎) = 𝑄 𝜃 (𝜌, 𝑎)
represent the 𝑄-value function, and 𝜋 𝜙 (𝜌, 𝑎) denote the policy func-
tion. Assumed that the output of 𝜋 𝜙 follows a normal distribution
with expectation and variance, the 𝑄-value function can be learned
by minimizing the flexible Bellman residuals:
1
𝐽𝑄 (𝜃 ) = E (𝑠𝑡,𝑎𝑡 )∼D [ 2 (𝑄𝜃 (𝜌𝑡, 𝑎𝑡 ) − (𝑟 (𝑠𝑡, 𝑎𝑡 ) + 𝛾E𝑠𝑡+1∼𝑝 [𝑉 𝜃¯ (𝜌𝑡+1 ) ]))2) ]
𝑉 𝜃¯ (𝜌𝑡 ) = E𝑎𝑡 ∼𝜋 [𝑄 𝜃¯ (𝜌 (𝑠𝑡,𝑚), 𝑎𝑡 ) − 𝛼 log 𝜋𝜙 (𝑎𝑡 |𝜌𝑡 ) ], (5)
where 𝜌 (𝑠 𝑡 , 𝑚) abbreviated as 𝜌 𝑡 , 𝑄 𝜃¯ represents the target Q-value
network, 𝜃¯ is the exponential moving average of the parameter 𝜃.
Optimization for Policy Network. To optimize 𝐽 𝜋 (𝜙), we
utilize the reparameterization technique for the policy network 𝜋 𝜙 .
This technique involves representing 𝜋 𝜙 as a function that takes
the state 𝑠 and standard Gaussian samples 𝜖 as inputs and directly
outputs the action 𝑎 = f𝜙 (𝜖; 𝑠). Assuming N is the standard normal
distribution, 𝜋 𝜙 can be derived by minimizing KL divergence:
𝐽𝜋 (𝜙) = E𝑠𝑡 ∼𝐷,𝜖𝑡 ∼N [𝛼 log 𝜋𝜙 (f𝜙 (𝜖𝑡 ; 𝜌𝑡 ) |𝑠𝑡 ) − 𝑄𝜃 (𝜌𝑡, f𝜙 (𝜖𝑡 ; 𝜌𝑡 )) ]. (6)
Optimization for Parameter Alpha. We employ an automatic
entropy tuning method to adjust parameter 𝛼 by minimizing the
following loss function:
𝐽 (𝛼 ) = 𝐸𝑎𝑡 ∼𝜋𝑡 [−𝛼 log 𝜋𝑡 (𝑎𝑡 |𝜌 (𝑠𝑡,𝑚)) − 𝛼 H¯ ], (7)
where H¯ is the target entropy hyperparameter.
Optimization for Maskable Stock Representation. In the
masking and reconstruction process, we optimize the maskable
stock representation using mean-squared error. Reconstruction
losses are calculated based only on the price of masked stocks:
1 ∑︁𝑁∗
𝐽 (𝜃𝑒,𝜃𝑐,𝜃𝑒𝑛𝑐,𝜃𝑑𝑒𝑐 ) =
𝑁 ∗ 𝑖=1
(𝑝𝑖,𝑡 − 𝑝˜𝑖,𝑡 )2, (8)

ment in Customizable Stock Pools WWW ’24, May 13–17, 2024, Singapore, Singapore
act Pool-level Embedding through Masking and Reconstruction Process
Mask-filled Operation
Decoder
Critic Actor
Cash
0.1 0.0 0.0 0.7 0.2 Re-weighting
(c) Reinforcement Learning for Portfolio Management in CSP
used to extract stock-level embeddings from GSP. Module (b) is the
beddings. Module (c) is an agent with masked token awareness.
where 𝑁 ∗ represents the number of masked stocks. Notably, pre-
training the encoder on maskable stock representations faces two
main drawbacks: i) creates a gap between self-supervised price pre-
diction tasks and RL decision-making tasks; ii) potentially limits the
exploration space in RL by frozen embeddings that may not always
positively impact decision-making. Hence, conducting maskable
stock representation optimization and RL optimization simulta-
neously contribute to better performance and more user-friendly,
end-to-end models.
Our implementation process follows the same optimization pro-
cess for each batch of SAC. We first optimize the Q-value network,
followed by the alpha, strategy network, and maskable stock rep-
resentation. However, we find that using weighted sum loss to
optimize both the maskable stock representation and the remaining
three components will have a negative impact on the distribution
of data sampled by the RL process and lead to unstable training.
4.3 Re-weighting Method
In a continuous decision space in portfolio management, agents
face difficulties in making accurate decisions. For instance, in a con-
stantly changing market environment, agents may overfit a fixed
number of market patterns and struggle to react quickly in high-
volatility markets. These issues can lead to agents micro-investing
in stocks with low future returns or even result in losses.
In our setting for PM with CSPs, we encounter unique problems
in addition to those already present in PM: i) the state that we
input to the agent is latent embeddings containing filled masked
tokens, the agent may be investing in the masked stocks that these
investors expect to lose money, which is precisely what investors
do not want to witness; ii) due to the extent of error in decision-
making, part of the investment proportion of high-yield stocks may
be taken by stocks with low or negative expected future returns.
Both problems can be solved by portfolio sparsification. Drawing
inspiration from the Boltzmann distribution and Gumbel-Softmax,
we introduce an additional hyperparameter 𝑇 to the softmax func-
tion for re-weighting portfolios to achieve sparsification of tiny
investment proportions to zero:
𝑅𝑒 (x) = 𝑒𝑥𝑖/𝑇 / ∑︁𝑁 𝑒𝑥𝑗 /𝑇 . (9)
𝑗=1


## PAGE 6

WWW ’24, May 13–17, 2024, Singapore, Singapore
In this context, x represents the Actor’s logits for portfolio, and
𝑇 ∈ (0, ∞) is a temperature parameter. Lower 𝑇 values lead to
sparser allocations. As 𝑇 approaches 0, all investments tend to
allocate to the asset with the highest expected return. For 𝑇 = 1,
re-weighting degenerates to softmax, while for 𝑇 > 1, it reduces
the allocation variance and even leads to equal allocation. Notably,
re-weighting is included during RL optimization and will be used
during the training and testing as shown in Appendix D.
5 EXPERIMENTS
In this section, we conduct a series of experiments to evaluate
the proposed framework. First, we demonstrate that our approach
achieves better returns in two real US financial markets and sub-
stantially outperforms the baseline methods in the global stock pool.
Next, we construct 6 customizable stock pools based on 3 different
investor investment preferences in two US financial markets to
demonstrate that our framework can effectively meet investors’
preferences and decisions in the trading process. Finally, we con-
duct ablation studies to answer the following questions:
RQ1: How is the usefulness of each component of EarnMore?
RQ2: Why are direct methods for PM with CSPs not working?
RQ3: How is the efficiency of the EarnMore model?
5.1 Datasets and Processing
Table 1: Datasets and Date Splits
SP500 DJ30
Dataset
GSP CSP1 CSP2 CSP3 GSP CSP1 CSP2 CSP3
Stock 420 62 39 168 28 10 7 10
Industry 49 8 7 28 24 9 6 8
Train Test
Date Split 2007-09-26 ∼ 2018-01-25 2018-01-26 ∼ 2019-07-22
2007-09-26 ∼ 2019-07-22 2019-07-23 ∼ 2021-01-08
2007-09-26 ∼ 2021-01-07 2021-01-07 ∼ 2022-06-26
In our experiment, we study daily data for 10,273 US stocks from
Yahoo Finance, deriving indicators to understand market trends.
After preprocessing to address data quality issues, we ended up
with 3,094 US stocks and 95 technical indicators based on Qlib’s Al-
pha158 [34]. We conduct a comprehensive evaluation to validate our
framework’s effectiveness and performance in various real-world
scenarios. We consider two main factors in the evaluation: i) events
and markets under different conditions, e.g. COVID-19, geopolitical
conflicts, bull and bear; ii) customizable stock pools with differ-
ent investors’ investment preferences, such as one investor prefers
investing in the technology and communication industries, and
another one prefers investing in financial and insurance industries.
Based on the above statements as shown in Table 1, we choose 3
date periods as test datasets from 2018-01-16 to 2022-06-26.
We construct 8 diversified datasets from the two US stock in-
dices, which are SP500 and DJ30. According to the Global Industry
Classification Standard6 (GICS), we categorize SP500 and DJ30 into
49 and 24 industries at the industry level. Examples of industries
include banking, insurance, software services, automotive manu-
facturing, and so on. Global stock pool (GSP) is the full set of stock
pools containing 420 and 28 stocks respectively. Then we carefully
6https://www.msci.com/our-solutions/indexes/gics

Wentao Zhang et al.
select three CSPs based on three different investor preferences, with
CSP1, CSP2, and CSP3 corresponding to the technology, financial,
and service as the main industries. To better reflect the real-world
demand for industry diversity in PM, we randomly add several
stocks from other industries into the three CSPs. It’s worth men-
tioning that the antagonistic and cooperative correlations between
stocks has been reflected in our diversified customizable stock pool
selection. For example, the CSP1 on the DJ30 includes tech giants
like Apple and Microsoft, showcasing their competitive relationship
(e.g., main businesses are tablet PCs and laptops). The CSP1 on the
SP500 includes Intel and Microsoft, illustrating their cooperative
relationship (e.g., Intel’s chips used with Microsoft Windows).
5.2 Evaluation Metrics
We compare EarnMore and baselines in terms of 6 financial metrics,
including 1 profit criterion, 3 risk-adjusted profit criteria, and 2 risk
criteria. Definitions and formulas are available in Appendix A.
5.3 Baselines
To provide a comprehensive comparison of EarnMore, we select 14
state-of-the-art and representative stock prediction methods of 4
different types consisting of 3 rule-based (Rule-based) methods,
2 machine learning-based (ML-based) methods, 2 deep learning-
based (DL-based) methods and 7 reinforcement learning-based
(RL-based) methods. Details of them are available in Appendix B.
5.4 Implement Details
Experiments are conducted on an Nvidia A6000 GPU, and we use
grid search to determine the hyperparameters. The implementation
of those ML-based and DL-based methods is based on Qlib[34].
As for other baselines, we use the default settings in their public
implementations. We run experiments with individual 9 runs using
3 date splits × 3 random selected seeds and report the average
performance7. Detailed implementation is available in Appendix C.
5.5 Results and Analysis
The performances of EarnMore and other baseline methods in GSPs
on the two US financial markets SP500 and DJ30 are shown in Table
2. Specifically, the results of cumulative return are drawn in Figure
3. We also train and test each baseline method individually for each
CSP, for the method shows no adaptive ability in customized stock
pools. We then compared the 3 state-of-the-art RL-based methods
with EarnMore on two important metrics, as shown in Table 3.
Furthermore, to evaluate EarnMore’s ability to adapt to investors’
personal decisions during trading process, we point out several real-
world scenarios of adjusting CSP and display the dynamic changes
of EarnMore in Figure 4.
Performance on Global Stock Pools. We compared EarnMore
with 14 baseline methods in terms of 6 financial metrics. Table
2 and Figure 3 demonstrate our framework outperforms others
on portfolio management with higher returns in GSPs. For the
SP500, EarnMore achieves the highest ARR of 97% and SR of 2.032,
significantly higher than the second-best method. For the DJ30,
EarnMore achieves improvements in terms of ARR, SR, CR, and SoR
7We report the experiment results of EarnMore if not particularly pointed out.


## PAGE 7

Reinforcement Learning with Maskable Stock Representation for Portfolio Management in C
Table 2: Performance comparison on SP500 and DJ30 with Global
second best and third best results on each dataset.
Stock Index SP500
Profit Risk-Adjusted Profit Risk
Categories Strategies
ARR%↑ SR↑ CR↑ SOR↑ MDD%
Market 9.320 0.556 0.702 17.120 26.160
Rule-based BLSW 11.630 0.696 0.894 21.450 24.560
CSM 5.070 0.329 0.434 9.840 23.350
ML-based XGBoost 10.690 0.377 0.473 13.650 19.300
LightGBM 16.330 0.575 0.744 20.110 24.760
DL-based ALSTM 43.50 1.157 1.367 22.501 35.820
TCN 13.560 1.044 1.460 14.540 35.780
PG 12.580 0.431 0.519 24.340 26.180
PPO 15.130 0.537 0.742 14.770 24.100
SAC 15.140 0.538 0.743 14.770 24.100
EIIE 15.030 0.540 0.627 15.450 26.920
RL-based SARL 21.240 0.756 0.970 21.230 24.000
IMIT 50.300 1.162 1.949 35.050 25.420
DeepTrader 60.290 1.980 2.195 34.260 28.580
EarnMore 97.170 2.032 2.506 42.160 28.120
Improvement over SOTA 61.171% 2.626% 14.169% 20.285% -
150
100
50
0
50
2019-07 2019-09 2019-11 2020-01 2020-03 2020-05 2020-07 2020-09 2020-11 2021-01
(a)
)% ni(
nruteR
evitalumuC
Date vs. Cumulative Return (SP500)
Market ALSTM EIIE
CSM TCN SARL 125
BLSW PG IMIT
XGBoost PPO DeepTrader
LightGBM SAC EarnMore 100
75
50
25
0
25
2019-07 2019-09 2019-11 2020-01 2020-03 2020-05 2020-07 2020-09 2020-11 2021-01
(b)
)% ni(
nruteR
evitalumuC
Date vs. Cumulative Return (DJ30)
Market ALSTM EIIE
BLSW TCN SARL
CSM PG IMIT
XGBoost PPO DeepTrader
LightGBM SAC EarnMore
Figure 3: Performance on GSP for SP500 and DJ30
by 46.7%, 8.9%, 6.2%, and 2.4%. We can also observe that ML-based
methods are optimal in controlling risk, but not outstanding in
capturing returns. The reason behind it is that tree models are more
robust to outliers and noise in the data, and thus can adaptively
capture non-linear relationships to reduce decision risk. Specifically,
ALSTM achieved a surprising 43.5% return on the SP500, due to large
returns from several decisions in a large number of bad decisions,
and thus we do not recommend using it. Besides, it is worth noting
that the higher the potential return, the higher the risk involved in
portfolio management. EarnMore is slightly inferior yet comparable
to baseline methods on risk metrics, i.e., MDD and VOL. As for DJ30
dataset, EarnMore fails to perform well in MDD but achieves over
40% improvement in ARR to DeepTrader, which is the second best
overall. Thus, for EarnMore, it is a slight compromise on risk control,
as our priority is to maximize final portfolio values.
The global COVID-19 pandemic reached its peak between Feb-
ruary 14 and March 20 2020, causing a significant decline in the
economy and intense investor concerns. This led to a substantial
fall in the stock market, with the SP500 and DJ30 indices falling by
31.81% and 34.78%, respectively. As shown in Figure 3, EarnMore
is far less affected in returns than baseline methods and continues
to gain returns after the market rebounded. Even during market
downturns, EarnMore is able to identify stocks with the potential
to generate higher returns when the market rebounds.
Performance on Customizable Stock Pools. We illustrate
the effectiveness of CSPs in two aspects. Firstly, we compare the
profitability performance of CSPs - formulated according to investor

ment in Customizable Stock Pools WWW ’24, May 13–17, 2024, Singapore, Singapore
Global Stock Pool. Results in red, yellow and green show the best,
DJ30
Risk Metrics Profit Risk-Adjusted Profit Risk Metrics
MDD%↓ VOL↓ ARR%↑ SR↑ CR↑ SOR↑ MDD%↓ VOL↓
26.160 0.014 6.710 0.458 0.776 15.560 22.200 0.013
24.560 0.013 7.610 0.512 0.857 16.930 21.540 0.012
23.350 0.013 5.930 0.400 0.643 12.950 20.770 0.012
19.300 0.016 10.260 0.343 0.599 10.420 14.760 0.013
24.760 0.016 13.420 0.591 0.703 14.220 20.900 0.014
35.820 0.026 15.030 1.186 0.590 14.890 28.070 0.013
35.780 0.025 6.980 0.732 0.269 8.280 37.400 0.018
26.180 0.014 7.970 0.321 0.435 8.430 21.570 0.012
24.100 0.013 9.240 0.385 0.512 10.140 20.810 0.012
24.100 0.013 9.150 0.326 0.448 8.830 20.600 0.012
26.920 0.015 22.900 0.689 1.465 23.450 16.770 0.014
24.000 0.013 21.920 0.786 1.109 23.020 20.400 0.012
25.420 0.018 27.640 0.909 1.593 27.380 20.050 0.014
28.580 0.013 32.230 1.335 1.440 27.110 21.190 0.013
28.120 0.023 47.290 1.454 1.692 28.040 21.650 0.018
- - 46.727% 8.914% 6.215% 2.411% - -
preferences - with 3 state-of-the-art RL-based methods. Secondly,
we demonstrate the adaptability and robustness of EarnMore to
investors’ personal decisions in the trading process.
Table 3: Performance comparison on SP500 and DJ30 with
Customizable Stock Pools. Underlined metrics indicate the
best-performing results.
Stock Index SP500 DJ30
Pool Strategies ARR%↑ SR↑ ARR%↑ SR↑
SARL 34.330 0.820 24.140 0.638
CSP1 IMIT 20.973 0.860 20.071 0.920
DeepTrader 34.030 0.793 27.740 0.757
EarnMore 122.610 2.278 53.990 1.810
SARL 17.000 0.570 20.020 0.820
CSP2 IMIT 7.971 0.486 11.841 0.751
DeepTrader 34.030 0.793 38.470 0.955
EarnMore 110.110 2.279 43.400 1.549
SARL 18.090 0.760 10.910 0.480
CSP3 IMIT 21.193 1.220 6.851 0.496
DeepTrader 61.320 1.489 16.840 0.601
EarnMore 93.670 2.120 43.460 1.572
EarnMore achieves impressive performances on all CSPs, as
shown in Table 3. Specifically, in the CSP1 stock pool, which con-
sists of technology stocks, EarnMore’s profitability is significantly
higher compared to other methods. It is consistent with the notable
increasing values of technology stocks over an extended period
of time, and thus demonstrates that our method provides more
scope for profit-seeking. In the other two CSPs, which are in the
financials and services industries, EarnMore also delivers notable
return improvements. Overall, our method is able to automatically
adapt to investors’ preferences and generate substantial returns.
General Electric Company (GE) was delisted from the SP500 on
June 26, 2018. As shown in Figure 4(a), after removing GE from
the stock pool of the SP500 on the date of June 26, 2018, EarnMore
can adapt itself to investor decisions to achieve a small return in-
crease. Between March 2022 and June 2022, the stock price of Apple
(APPL) dropped sharply by nearly 25% due to several factors, in-
cluding the impact of the war and Apple’s mobile phone downtime
incident. As shown in Figure 4(b), excluding AAPL from DJ30 can
significantly improve returns. It is important to mention that we


## PAGE 8

WWW ’24, May 13–17, 2024, Singapore, Singapore
add Microsoft (MSFT) technology stock to the CSP2 of the SP500.
We purposely chose periods when the MSFT price was decreas-
ing, to test the strength and robustness of EarnMore, in case an
investor inadvertently selects an unsuitable stock. As depicted in
Figure 4(c), EarnMore is able to decrease its MSFT investment by
properly screening the stock with minimal to no impact on the
overall returns throughout the trading procedure. Goldman Sachs
(GS) announced the advancement of steel project and new energy
vehicle development, which released a signal that stock price of GS
would rise in 2021. Thus, our model adds GS to the stock pool, as
shown in Figure 4(d), and gets more significant return growth.
80
75
70
65
60
55
50
2022-03-0
2
1 022-03-15 2022-04-0
2
1 022-04-15 2022-05-0
2
1 022-05-15 2022-06-01
(b)
)% ni(
nruteR
evitalumuC
Date vs. Cumulative Return (DJ30)
100 with_AAPL
without_AAPL 90 remove AAPL
80
70
60
2018-05-01 2018-05-15 2018-06-01 2018-06-15 2018-07-01 2018-07-15 2018-08-01
(a)
)% ni(
nruteR
evitalumuC
Date vs. Cumulative Return (SP500)
with_GE
without_GE remove GE
30
25
20
15
10
5
0
2021-01-1
2
5 021-02-
2
0
0
1 21-02-
2
1
0
5 21-03-
2
0
0
1 21-03-1
2
5 021-04-
2
0
0
1 21-04-1
2
5 021-05-01
(d)
)%
ni(
nruteR
evitalumuC
Date vs. Cumulative Return (DJ30)
without_GS
130 with_GS
120 add GS
110
100
90
80
70
2021-11
2
-0
0
1 21-11-1
2
5 021-12
2
-0
0
1 21-12-1
2
5 022-01
2
-0
0
1 22-01-1
2
5 022-02
2
-0
0
1 22-02
2
-1
0
5 22-03-01
(c)
)%
ni(
nruteR
evitalumuC
Date vs. Cumulative Return (SP500)
without_MSFT
with_MSFT
add MSFT
Figure 4: Performance on CSPs with dynamic changes
5.6 Ablation Study
Table 4: Ablation Study of EarnMore.Red, green and under-
line indicate improvement, performance decrease and best
results, respectively.
Stock Index SP500 DJ30
Pool Model ARR Δ ARR SR Δ SR MDDΔ MDD ARR Δ ARR SR Δ SR MDDΔ MDD
w/o-MR 33.2 - 1.49 - 22.7 - 10.5 - 0.71 - 21.5 -
GSP w/o-M 74.8 125.5 1.89 27.3 22.5 -1.2 31.4 198.6 1.14 60.8 19.4 -9.5
EarnMore 97.2 29.9 2.03 7.23 28.1 25.3 47.3 50.7 1.45 28.1 21.7 11.4
w/o-MR 32.6 - 1.18 - 25.4 - 14.2 - 0.78 - 20.5 -
CSP1 w/o-M 46.7 43.3 1.09 -7.6 32.2 26.5 23.9 69.5 1.02 30.9 22.2 8.50
EarnMore 122.6162.3 2.28108.4 25.1 -22.1 53.9 125.0 1.81 78.2 22.9 3.09
w/o-MR 8.46 - 0.62 - 28.2 - 11.02- 0.81 - 20.1 -
CSP2 w/o-M 18.14114.4 0.70 13.9 32.1 13.9 22.2 8.50 0.02 31.7 21.7 8.26
EarnMore 110.1507.0 2.28223.7 25.8 -19.7 43.4 95.1 1.55 42.6 24.3 11.9
w/o-MR 27.5 - 1.35 - 21.7 - 5.17 - 0.41 - 22.5 -
CSP3 w/o-M 56.3 104.7 1.80 33.2 21.8 0.37 17.1 231.0 0.73 78.3 25.3 12.7
EarnMore 93.7 66.3 2.12 17.9 24.0 10.4 43.5 153.9 1.571151 21.18 -16.23
Effectiveness of Each Component (RQ1). In Table 4, we study
the impact of maskable stock representation, customizable stock
pools, and re-weighting methods. Comparing EarnMore-w/o-M
with EarnMore reveals a significant improvement due to the gen-
eralized pool-level maskable stock representation. The absence of
this representation increases investment risk. Both GSP and CSPs
benefit, with CSPs outperforming GSP, suggesting potential for
higher returns with focused stock selection. Both GSP and CSPs
benefit, with CSPs performing better, suggesting the potential for
higher returns through a more focused stock selection.

Wentao Zhang et al.
Comparing the EarnMore-w/o-MR and EarnMore-w/o-M, we
find that the re-weighting method can achieve significant improve-
ments in profits by sparsifying portfolios, somehow in increasing
the MDD risk metric. Despite reducing the portfolio’s diversity may
decrease the chances of selecting stocks from various industries
and potentially raise risks, especially in CSPs with limited indus-
try variety, focusing on a select few industries can considerably
improve returns and offset potential losses due to risk.
1.6 1.4
1.2
1.0
0.8
0.6
0.4
0.2
0.0 ARR SR
(a)
RS
&
RRA
Performance
Train from scratch Finetuning Action masking 103 EarnMore
102
101 1 2 3
(b)
)elacS
goL
spe/s
ni(
tsoC
niarT
Number of CSPs vs. Train Cost (in s/eps Log Scale)
ALSTM EIIE TCN SARL PG IMIT PPO DeepTrader SAC EarnMore
Figure 5: (a) Comparing the performance of EarnMore with
direct methods on DJ30. (b) Comparing the time costs of
EarnMore with several other methods on DJ30.
Difficulties with Direct Methods (RQ2). There are three sim-
ple approaches to transition from PM with GSP to PM with CSPs,
which are training-from-scratch, fine-tuning and action-masking.
The first thing to note that both the training-from-scratch and fine-
tuning approaches lack the ability to make real-time adjustments
to the stock pool during trading. Nevertheless, we make a com-
parison between these three methods and EarnMore in terms of
DJ30 ARR and SR via SAC, with the metrics representing the av-
erage performance over three investment periods. As depicted in
Figure 5(a), EarnMore significantly outperforms the other direct
methods. It is worth noting that the challenges of fine-tuning and
training-from-scratch may be closely related in the context of PM
with CSPs, and that action-masking essentially relies on logits that
do not accurately reflect the agent’s actual decisions.
Efficiency of EarnMore (RQ3). Our framework is trained only
once to meet the demand of various investors for customizable stock
pools and individual decision making. As shown in Figure 5(b), we
have selected several other methods to compare with EarnMore,
and it can be demonstrated that as the number of CSPs increases,
the efficiency of our framework shows up.
6 CONCLUSION AND FUTURE DIRECTION
This paper introduces a novel RL framework for portfolio manage-
ment featuring adaptive investor preference and personal decision
awareness for customizable stock pools. Maskable stock represen-
tation is enhanced by masking and reconstruction process, and a
re-weighting method is introduced to improve sparsified portfolios.
These improvements yield superior portfolio performance com-
pared to the benchmark methods, as evidenced by various financial
criteria. For future research directions, two key areas will be pri-
oritized. Firstly, we will focus on enhancing risk control via risk
penalty optimization. Secondly, we aim to create a flexible, open
customizable stock pool that allows easy stock addition or removal.


## PAGE 9

Reinforcement Learning with Maskable Stock Representation for Portfolio Management in C
ACKNOWLEDGMENTS
This project is supported by the National Research Foundation,
Singapore under its Industry Alignment Fund – Pre-positioning
(IAF-PP) Funding Initiative. Any opinions, findings and conclu-
sions or recommendations expressed in this material are those of
the author(s) and do not reflect the views of National Research
Foundation, Singapore.
REFERENCES
[1] Shaojie Bai, J Zico Kolter, and Vladlen Koltun. 2018. An empirical evaluation
of generic convolutional and recurrent networks for sequence modeling. arXiv
preprint arXiv:1803.01271 (2018).
[2] Carlos Betancourt and Wen-Hui Chen. 2021. Deep reinforcement learning for
portfolio management of markets with a dynamic number of assets. Expert
Systems with Applications 164 (2021), 114002.
[3] Tianqi Chen and Carlos Guestrin. 2016. Xgboost: A scalable tree boosting system.
In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
Discovery and Data Mining. 785–794.
[4] Yue Deng, Feng Bao, Youyong Kong, Zhiquan Ren, and Qionghai Dai. 2016. Deep
direct reinforcement learning for financial signal representation and trading.
IEEE Transactions on Neural Networks and Learning Systems 28, 3 (2016), 653–664.
[5] Yi Ding, Weiqing Liu, Jiang Bian, Daoqiang Zhang, and Tie-Yan Liu. 2018. Investor-
imitator: A framework for trading knowledge extraction. In Proceedings of the 24th
ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.
1310–1319.
[6] Jiaxiang Dong, Haixu Wu, Haoran Zhang, Li Zhang, Jianmin Wang, and Ming-
sheng Long. 2023. SimMTM: A simple pre-training framework for masked
time-series modeling. arXiv preprint arXiv:2302.00861 (2023).
[7] Eugene F Fama. 1970. Efficient capital markets: A review of theory and empirical
work. The Journal of Finance 25, 2 (1970), 383–417.
[8] Yuchen Fang, Kan Ren, Weiqing Liu, Dong Zhou, Weinan Zhang, Jiang Bian,
Yong Yu, and Tie-Yan Liu. 2021. Universal trading for order execution with oracle
policy distillation. In AAAI Conference on Artificial Intelligence. 107–115.
[9] Alhussein Fawzi, Matej Balog, Aja Huang, Thomas Hubert, Bernardino Romera-
Paredes, Mohammadamin Barekatain, Alexander Novikov, Francisco J R Ruiz,
Julian Schrittwieser, Grzegorz Swirszcz, et al. 2022. Discovering faster matrix
multiplication algorithms with reinforcement learning. Nature 610, 7930 (2022),
47–53.
[10] Fuli Feng, Xiangnan He, Xiang Wang, Cheng Luo, Yiqun Liu, and Tat-Seng Chua.
2019. Temporal relational ranking for stock prediction. ACM Transactions on
Information Systems (TOIS) 37, 2 (2019), 1–30.
[11] Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon
Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, et al. 2018.
Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905
(2018).
[12] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick.
2022. Masked autoencoders are scalable vision learners. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16000–16009.
[13] Harrison Hong and Jeremy C Stein. 1999. A unified theory of underreaction,
momentum trading, and overreaction in asset markets. The Journal of Finance
54, 6 (1999), 2143–2184.
[14] Narasimhan Jegadeesh and Sheridan Titman. 2002. Cross-sectional and time-
series determinants of momentum returns. The Review of Financial Studies 15, 1
(2002), 143–157.
[15] Zhengyao Jiang, Dixing Xu, and Jinjun Liang. 2017. A deep reinforcement
learning framework for the financial portfolio management problem. arXiv
preprint arXiv:1706.10059 (2017).
[16] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma,
Qiwei Ye, and Tie-Yan Liu. 2017. Lightgbm: A highly efficient gradient boosting
decision tree. Advances in Neural Information Processing Systems 30 (2017).
[17] Tianhong Li, Huiwen Chang, Shlok Mishra, Han Zhang, Dina Katabi, and Dilip
Krishnan. 2023. Mage: Masked generative encoder to unify representation learn-
ing and image synthesis. In Proceedings of the IEEE/CVF Conference on Computer

ment in Customizable Stock Pools WWW ’24, May 13–17, 2024, Singapore, Singapore
Vision and Pattern Recognition. 2142–2152.
[18] Zhe Li, Zhongwen Rao, Lujia Pan, Pengyun Wang, and Zenglin Xu. 2023. Ti-MAE:
Self-supervised masked time series autoencoders. arXiv preprint arXiv:2301.08871
(2023).
[19] Xiao-Yang Liu, Hongyang Yang, Qian Chen, Runjia Zhang, Liuqing Yang, Bowen
Xiao, and Christina Dan Wang. 2020. FinRL: A deep reinforcement learning
library for automated stock trading in quantitative finance. arXiv preprint
arXiv:2011.09607 (2020).
[20] Tobias J Moskowitz, Yao Hua Ooi, and Lasse Heje Pedersen. 2012. Time series
momentum. Journal of Financial Economics 104, 2 (2012), 228–250.
[21] Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. 2022.
A time series is worth 64 words: Long-term forecasting with transformers. arXiv
preprint arXiv:2211.14730 (2022).
[22] James M Poterba and Lawrence H Summers. 1988. Mean reversion in stock prices:
Evidence and implications. Journal of Financial Economics 22, 1 (1988), 27–59.
[23] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, and Garrison
Cottrell. 2017. A dual-stage attention-based recurrent neural network for time
series prediction. arXiv preprint arXiv:1704.02971 (2017).
[24] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347
(2017).
[25] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja
Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton,
et al. 2017. Mastering the game of go without human knowledge. Nature 550,
7676 (2017).
[26] Thomas Spooner, John Fearnley, Rahul Savani, and Andreas Koukorinis. 2018.
Market making via reinforcement learning. In International Conference on Au-
tonomous Agents and MultiAgent Systems.
[27] Shuo Sun, Molei Qin, wentao zhang, Haochong Xia, Chuqiao Zong, Jie Ying,
Yonggang Xie, Lingxuan Zhao, Xinrun Wang, and Bo An. 2023. TradeMaster: A
Holistic Quantitative Trading Platform Empowered by Reinforcement Learning.
In Thirty-seventh Conference on Neural Information Processing Systems Datasets
and Benchmarks Track.
[28] Shuo Sun, Rundong Wang, and Bo An. 2023. Reinforcement learning for quan-
titative trading. ACM Transactions on Intelligent Systems and Technology 14, 3
(2023), 1–29.
[29] Shuo Sun, Wanqi Xue, Rundong Wang, Xu He, Junlei Zhu, Jian Li, and Bo An. 2022.
DeepScalper: A risk-aware reinforcement learning framework to capture fleeting
intraday trading opportunities. In Proceedings of the 31st ACM International
Conference on Information & Knowledge Management. 1858–1867.
[30] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. 1999.
Policy gradient methods for reinforcement learning with function approximation.
Advances in Neural Information Processing Systems 12 (1999).
[31] Rundong Wang, Hongxin Wei, Bo An, Zhouyan Feng, and Jun Yao. 2021. Com-
mission fee is not enough: A hierarchical reinforced framework for portfolio
management. In Proceedings of the AAAI Conference on Artificial Intelligence.
626–633.
[32] Zhicheng Wang, Biwei Huang, Shikui Tu, Kun Zhang, and Lei Xu. 2021. Deep-
Trader: A deep reinforcement learning approach for risk-return balanced portfolio
management with market conditions Embedding. In Proceedings of the AAAI
Conference on Artificial Intelligence. 643–650.
[33] Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng
Long. 2023. TimesNet: Temporal 2D-Variation Modeling for General Time Series
Analysis. In International Conference on Learning Representations.
[34] Xiao Yang, Weiqing Liu, Dong Zhou, Jiang Bian, and Tie-Yan Liu. 2020. Qlib: An
AI-oriented quantitative unvestment platform. arXiv:2009.11189 [q-fin.GN]
[35] Yunan Ye, Hengzhi Pei, Boxin Wang, Pin-Yu Chen, Yada Zhu, Ju Xiao, and Bo Li.
2020. Reinforcement-learning based portfolio management with augmented asset
movement prediction states. In Proceedings of the AAAI Conference on Artificial
Intelligence. 1112–1119.
[36] Jaemin Yoo, Yejun Soun, Yong-chan Park, and U Kang. 2021. Accurate multivariate
stock movement prediction via data-axis transformer with multi-level contexts.
In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &
Data Mining. 2037–2045.


## PAGE 10

WWW ’24, May 13–17, 2024, Singapore, Singapore
120
100
80
60
40
20
0
20
2018-01 2018-03 2018-05 2018-07 2018-09 2018-11 2019-01 2019-03 2019-05 2019-07
(a)
)% ni(
nruteR
evitalumuC
Date vs. Cumulative Return (SP500)
Market ALSTM EIIE
CSM TCN SARL B X L G S B W oost P P G PO I D M e I e T pTrader 60
LightGBM SAC EarnMore
40
20
0
20
2018-01 2018-03 2018-05 2018-07 2018-09 2018-11 2019-01 2019-03 2019-05 2019-07
(b)
)% ni(
nruteR
evitalumuC
Date vs. Cumulative Return (DJ30)
Market ALSTM EIIE
BLSW TCN SARL CSM PG IMIT XGBoost PPO DeepTrader
LightGBM SAC EarnMore
150
100
50
0
50
2019-07 2019-09 2019-11 2020-01 2020-03 2020-05 2020-07 2020-09 2020-11 2021-01
(c)
)% ni(
nruteR
evitalumuC
Date vs. Cumulative Return (SP500)
Market ALSTM EIIE
CSM TCN SARL 125
BLSW PG IMIT XGBoost PPO DeepTrader
LightGBM SAC EarnMore 100
75
50
25
0
25
2019-07 2019-09 2019-11 2020-01 2020-03 2020-05 2020-07 2020-09 2020-11 2021-01
(d)
)% ni(
nruteR
evitalumuC
Date vs. Cumulative Return (DJ30)
Market ALSTM EIIE
BLSW TCN SARL
CSM PG IMIT XGBoost PPO DeepTrader
LightGBM SAC EarnMore
140
120
100
80
60
40
20
0
2021-01 2021-03 2021-05 2021-07 2021-09 2021-11 2022-01 2022-03 2022-05 2022-07
(e)
)%
ni(
nruteR
evitalumuC
Market Date vs A . L C ST u M mulative R EI e IE turn (SP500) 80
CSM TCN SARL
BLSW PG IMIT
X Li G gh B t o G o B s M t P S P A O C D Ea e r e n p M T o ra re der 60
40
20
0
2021-01 2021-03 2021-05 2021-07 2021-09 2021-11 2022-01 2022-03 2022-05 2022-07
(f)
)%
ni(
nruteR
evitalumuC
Date vs. Cumulative Return (DJ30) Market ALSTM EIIE
BLSW TCN SARL
CSM PG IMIT
XGBoost PPO DeepTrader LightGBM SAC EarnMore
Figure 6: Performance on GSP and CSPs for SP500 and DJ30
A DETAILS OF EVALUATION METRICS
We compared EarnMore and baselines in terms of 6 financial metrics,
including 1 profit criterion, 3 risk-adjusted profit criteria, and 2 risk
criteria. Definitions and formulas are as follows:
• Annual Rate of Return (ARR) is the annualized average return
rate, calculated as 𝐴𝑅𝑅 = 𝑉𝑇 𝑉 −𝑉 0 × 𝑇 𝐶 , where 𝑇 is the total number
of trading days, and 𝐶 = 252 i 0 s the number of trading days within
a year. 𝑉 𝑇 and 𝑉 0 represent the final and initial portfolio values.
• Sharpe Ratio (SR) measures risk-adjusted returns of portfolios.
It is defined as 𝑆𝑅 = E[r] , where E[·] is the expectation, 𝜎 [·] is
𝜎 [r]
the standard deviation, r = [ 𝑉 1 𝑉 − 0 𝑉 0 , 𝑉 2 𝑉 − 1 𝑉 1 , ..., 𝑉𝑇 𝑉 − 𝑇 𝑉 − 𝑇 1 −1 ]𝑇 denotes
the historical sequence of the return rate.
• Volatility (VOL) is the variation in an investment’s return over
time, measured as the standard deviation 𝜎 [r].
• Maximum Drawdown (MDD) measures the largest loss from
any peak to show the worst case. It is defined as: 𝑀𝐷𝐷 = max 𝑇 𝑖=0 𝑃𝑖 𝑃 − 𝑖 𝑅𝑖 ,
where 𝑅 𝑖 = (cid:206)𝑇 𝑖=1 𝑉 𝑉 𝑖− 𝑖 1 and 𝑃 𝑖 = max 𝑇 𝑖=1 𝑅 𝑖 .
• Calmar Ratio (CR) compares average annualized return to
maximum drawdown, assessing risk-adjusted performance. It is
defined as 𝐶𝑅 = 𝑀 E 𝐷 [r 𝐷 ] .
• Sortino Ratio (SoR) is a risk-adjusted measure that focuses
on the downside risk of a portfolio. It is defined as 𝑆𝑜𝑅 = E 𝐷 [ 𝐷 r] ,
where 𝐷𝐷 is the standard deviation of negative return.

Wentao Zhang et al.
B DETAILS OF BASELINES
To provide a comprehensive comparison of EarnMore, we select 14
state-of-the-art and representative stock prediction methods of 4
different types consisting of 3 rule-based (Rule-based) methods,
2 machine learning-based (ML-based) methods, 2 deep learning-
based (DL-based) methods and 7 reinforcement learning-based
(RL-based) methods. Descriptions of baselines are as follows.
• Rule-based Methods: BLSW [22] is based on mean reversion
that buys underperforming stocks and sells outperforming ones.
CSM [14] is a momentum strategy that prefers assets with re-
cently strong performance and expects short-term success.
• ML-based Methods: XGBoost [3] leverages Gradient Boost-
ing Decision Tree (GBDT) for accurate predictions in super-
vised learning tasks. LightGBM [16] is an efficient GBDT with
gradient-based one-side sampling and exclusive feature bundling.
• DL-based Methods: ALSTM [23] ALSTM is a Recurrent Neural
Network (RNN) that uses an external attention layer to gather
information from all hidden states. TCN [1] is a Convolutional
Neural Network (CNN) architecture for sequence modeling in
time series analysis and natural language processing.
• RL-based Methods: PG [30] optimizes policy function while
considering risk and market conditions in PM without estimating
value function. SAC [11] is an off-policy actor-critic algorithm
that optimizes investment strategies in PM using entropy regu-
larization and soft value functions in continuous portfolio action
spaces. PPO [24] updates investment policies iteratively to bal-
ance exploration and exploitation, ensuring stability and sample
efficiency in PM. EIIE [15] is the first work formulating the PM
problem as an MDP, and it outperforms traditional PM methods
by using CNN for feature extraction and RL for portfolio deci-
sions. SARL [35] proposes a state-augmented RL framework,
which leverages the price movement prediction as additional
states based on deterministic policy gradient methods. Investor-
Imitator (IMIT) [5] shows good performance as a RL-based
framework in PM by replicating investor actions. DeepTrader
[32] leverages RL to model inter-stock relationships and balance
the risk-return trade-offs.
C DETAILS OF IMPLEMENTATION
The dimensions of our state are (𝐵, 𝑁, 𝐷, 𝐹 ), where 𝐵 = 128 rep-
resents the batch size, 𝑁 denotes the number of stocks. For DJ30,
𝑁 = 28, and for SP500, 𝑁 = 420. 𝐷 = 10 represents the number of
historical data days, and 𝐹 = 102 represents the total number of
features, including OHLC, technological indicators, and temporal
information. It is worth mentioning that due to the large amplitude
𝑖 𝑃 − 𝑖 𝑅𝑖 , of Volume, which is not conducive to RL training, we have removed
it from the features. For the encoder, decoder, actor, and critic, each
of them consists of 2 layers of MLP with GELU activation function.
The all components embedding dimension is 64.
All of our experiments were conducted on an Nvidia A6000 GPU,
and we used grid search to determine the hyperparameters. The
horizon length was chosen from the options {32, 64, 128, 256}, and
we found that 128 yielded the best results. The batch size was set
to 128, and the buffer size was 1𝑒5. The training process consisted
of 2000 episodes using the AdamW optimizer. For both the MAEs
component and the RL learning component of SAC, we utilized the
Mean Squared Error (MSE) Loss function. During the grid search


## PAGE 11

Reinforcement Learning with Maskable Stock Representation for Portfolio Management in C
Table 5: Performance comparison on SP500 and DJ30. Results in re
results on each dataset.
Stock Index SP500
Profit Risk-Adjusted Profit Risk Me
Pool Strategies
ARR%↑ SR↑ CR↑ SOR↑ MDD%↓
Market 9.320 0.556 0.702 17.120 26.160
BLSW 11.630 0.696 0.894 21.450 24.560
CSM 5.070 0.329 0.434 9.840 23.350
XGBoost 10.690 0.377 0.473 13.650 19.300
LightGBM 16.330 0.575 0.744 20.110 24.760
ALSTM 43.50 1.157 1.367 22.501 35.820
TCN 13.560 1.044 1.460 14.540 35.780
PG 12.580 0.431 0.519 24.340 26.180
GSP
PPO 15.130 0.537 0.742 14.770 24.100
SAC 15.140 0.538 0.743 14.770 24.100
EIIE 15.030 0.540 0.627 15.450 26.920
SARL 21.240 0.756 0.970 21.230 24.000
IMIT 50.300 1.162 1.949 35.050 25.420
DeepTrader 60.290 1.980 2.195 34.260 28.580
EarnMore 97.170 2.032 2.506 42.160 28.120
Improvement 61.171% 2.626% 14.169% 20.285% -
Market 16.581 0.722 1.242 26.728 28.703
BLSW 16.446 0.714 1.230 26.328 28.655
CSM 15.772 0.691 1.194 25.658 28.974
XGBoost 17.265 0.371 0.582 15.720 22.988
LightGBM 26.530 0.651 0.940 22.337 28.147
ALSTM 18.710 0.796 0.817 18.002 36.762
TCN 19.120 0.717 1.558 20.176 36.777
PG 7.130 0.278 0.382 29.330 30.350
CSP1
PPO 22.200 0.557 0.817 15.150 28.330
SAC 22.280 0.560 0.820 15.210 28.320
EIIE 39.090 0.635 1.107 17.940 33.340
SARL 34.330 0.820 1.159 22.330 27.860
IMIT 34.030 0.793 1.241 22.620 27.950
DeepTrader 20.973 0.860 1.516 31.222 28.389
EarnMore 122.610 2.278 2.957 48.430 25.060
Improvement 211.59% 164.89% 89.794% 55.115% -
Market 6.032 0.389 0.381 9.029 30.820
BLSW 6.218 0.396 0.393 9.317 30.717
CSM 5.924 0.379 0.367 8.660 30.584
XGBoost 9.275 0.356 0.570 16.240 21.440
LightGBM 10.687 0.383 0.506 18.060 30.120
ALSTM 11.959 0.607 1.127 15.019 34.388
TCN 12.192 0.456 0.408 10.729 36.163
PG 7.030 0.275 0.378 29.290 30.370
CSP2
PPO 6.940 0.271 0.374 8.110 30.440
SAC 6.930 0.272 0.374 8.130 30.450
EIIE 22.260 0.848 1.491 26.630 25.170
SARL 17.000 0.570 0.724 16.740 29.790
IMIT 24.890 0.534 0.797 15.690 31.470
DeepTrader 7.971 0.486 0.528 12.692 29.041
EarnMore 110.110 2.279 3.116 46.990 25.770
Improvement 342.39% 168.75% 108.79% 60.430% -
Market 7.048 0.497 0.578 14.183 24.235
BLSW 7.493 0.535 0.628 15.348 23.723
CSM 5.104 0.373 0.422 9.797 21.810
XGboost 9.528 0.401 0.572 14.130 16.938
LightGBM 12.863 0.512 0.710 19.083 22.863
ALSTM 13.849 0.519 0.470 11.074 34.208
TCN 9.733 0.854 1.184 16.498 34.958
PG 8.970 0.370 0.456 22.570 24.230
CSP3
PPO 11.400 0.402 0.593 13.390 22.270
SAC 11.420 0.487 0.732 13.420 22.270
EIIE 17.590 0.742 1.068 20.390 21.260
SARL 18.090 0.760 1.075 21.290 22.100
IMIT 61.320 1.489 2.913 32.050 22.220
DeepTrader 21.193 1.220 1.911 38.111 19.038
EarnMore 93.670 2.120 2.720 42.810 24.030
Improvement 52.756% 42.377% - 12.330% -
for the learning rates of the Actor, Critic, and MAEs component, we
tested values within the range {1𝑒 −3, 1𝑒 −4, 1𝑒 −5, 1𝑒 −6, 1𝑒 −7}, and
found that 1𝑒 − 5 yielded the best performance. The scheduler used
was the multi-step learning rate scheduler with warm-up technique,
starting with an initial learning rate of 1𝑒 − 8, which increased to

ment in Customizable Stock Pools WWW ’24, May 13–17, 2024, Singapore, Singapore
s in red, yellow and green show the best, second best and third best
DJ30
Risk Metrics Profit Risk-Adjusted Profit Risk Metrics
DD%↓ VOL↓ ARR%↑ SR↑ CR↑ SOR↑ MDD%↓ VOL↓
160 0.014 6.710 0.458 0.776 15.560 22.200 0.013
560 0.013 7.610 0.512 0.857 16.930 21.540 0.012
350 0.013 5.930 0.400 0.643 12.950 20.770 0.012
300 0.016 10.260 0.343 0.599 10.420 14.760 0.013
760 0.016 13.420 0.591 0.703 14.220 20.900 0.014
820 0.026 15.030 1.186 0.590 14.890 28.070 0.013
780 0.025 6.980 0.732 0.269 8.280 37.400 0.018
180 0.014 7.970 0.321 0.435 8.430 21.570 0.012
100 0.013 9.240 0.385 0.512 10.140 20.810 0.012
100 0.013 9.150 0.326 0.448 8.830 20.600 0.012
920 0.015 22.900 0.689 1.465 23.450 16.770 0.014
000 0.013 21.920 0.786 1.109 23.020 20.400 0.012
420 0.018 27.640 0.909 1.593 27.380 20.050 0.014
580 0.013 32.230 1.335 1.440 27.110 21.190 0.013
120 0.023 47.290 1.454 1.692 28.040 21.650 0.018
- 46.727% 8.914% 6.215% 2.411% - -
703 0.017 10.784 0.599 1.203 23.793 21.869 0.013
655 0.017 11.078 0.607 1.217 24.030 21.919 0.013
974 0.017 10.453 0.580 1.166 23.281 22.036 0.013
988 0.017 10.293 0.339 0.605 6.953 14.390 0.013
147 0.016 17.167 0.516 0.819 9.493 20.837 0.013
762 0.025 22.786 1.212 0.765 19.502 23.875 0.013
777 0.023 38.154 1.218 1.190 22.678 23.090 0.019
350 0.016 12.180 0.410 0.650 0.220 19.850 0.012
330 0.016 12.070 0.404 0.634 10.810 20.060 0.012
320 0.024 11.300 0.377 0.609 10.040 20.170 0.012
340 0.019 15.250 0.478 1.050 14.880 17.720 0.012
860 0.016 24.140 0.638 1.047 17.450 22.470 0.014
950 0.018 27.740 0.757 1.251 21.490 21.350 0.014
389 0.017 20.071 0.920 1.781 33.901 22.203 0.015
060 0.023 53.990 1.810 2.165 35.570 22.930 0.018
- 41.505 % 48.604 % 21.561 % 4.923% - -
820 0.017 8.930 0.593 0.730 15.274 22.502 0.014
717 0.017 9.094 0.600 0.750 15.610 22.443 0.014
584 0.017 8.388 0.572 0.705 14.701 22.329 0.014
440 0.017 8.560 0.360 0.591 10.823 16.508 0.013
120 0.016 13.610 0.561 0.831 16.717 20.553 0.012
388 0.017 14.903 1.121 0.526 13.876 29.800 0.014
163 0.024 31.249 1.186 1.079 19.243 27.634 0.019
370 0.016 8.970 0.366 0.546 10.510 20.700 0.012
440 0.016 8.490 0.355 0.458 10.350 20.300 0.012
450 0.016 8.750 0.366 0.566 10.720 20.180 0.012
170 0.016 14.180 0.595 1.062 18.800 25.890 0.017
790 0.016 20.020 0.820 1.234 25.050 21.460 0.013
470 0.018 38.470 0.955 1.646 30.450 25.520 0.014
041 0.016 11.841 0.751 1.009 20.978 21.711 0.013
770 0.024 43.400 1.549 1.744 28.010 24.330 0.019
- 46.727% 12.815% 30.607% 5.953 % - -
235 0.013 1.068 0.115 0.329 6.730 25.023 0.013
723 0.012 1.292 0.133 0.357 7.298 24.748 0.013
810 0.012 0.844 0.103 0.310 6.440 24.888 0.013
938 0.017 -0.935 0.023 0.049 0.415 19.203 0.014
863 0.012 3.907 0.185 0.288 5.080 23.440 0.013
208 0.015 7.804 0.729 0.403 9.923 30.900 0.013
958 0.023 7.228 0.570 0.401 10.098 36.633 0.018
230 0.013 -0.550 0.005 0.038 -0.300 23.520 0.012
270 0.012 -1.110 -0.028 0.003 -1.320 24.020 0.012
270 0.012 -0.750 -0.005 0.020 -0.610 23.740 0.012
260 0.012 -4.050 0.022 0.057 0.650 29.930 0.016
100 0.012 10.910 0.480 0.708 14.070 23.520 0.013
220 0.016 16.840 0.601 0.891 17.820 25.810 0.015
038 0.012 6.851 0.496 0.881 16.979 22.121 0.013
030 0.022 43.460 1.572 1.969 29.670 21.180 0.018
- 158.08% 115.64% 89.675% 66.498% - -
1𝑒 − 5 after 300 episodes, followed by subsequent multiplicative
reductions by 0.1 at the 600th, 1000th, and 1400th episodes. The
re-weighting parameters are given as 𝑎 = 0.6, 𝑏 = 0.8, 𝜇 = 0.7, and
𝜎 = 0.1. The optimal temperature parameter 𝑇 is determined to
be 0.1 from the set {10, 5, 1, 0.5, 0.1, 0.05, 0.01}. Default parameters


## PAGE 12

WWW ’24, May 13–17, 2024, Singapore, Singapore
were used for other baselines and all experiments ran with 3 seeds
for computing average metrics.
D PSEUDOCODE FOR THE TRAINING AND
INFERENCES PHASES OF EARNMORE
In this section, we present the detailed pseudocode for the training
and testing phases of EarnMore. This includes the training phase
with customizable stock pools simulated through masked token,
as well as the inference phase involving an investor-customized
target stock pool. Refer to Algorithm 1 and Algorithm 2.
Algorithm 1 Training of EarnMore
Require: Global Stock Pool 𝑈
Ensure: 𝜃 1 , 𝜃 2 , 𝜃 𝑒 , 𝜃 𝑐 , 𝜃 𝑒𝑛𝑐 , 𝜃 𝑑𝑒𝑐 , 𝜙 ⊲ Parameters
𝑠 𝑡 = [P𝑡 , Y𝑡 , D𝑡 ] ⊲ Initialize input data
𝜃˜ 1 = 𝜃 1 , 𝜃˜ 2 = 𝜃 2 ⊲ Initialize target network weights
D = Φ ⊲ Initialize an empty replay buffer
for each iteration and each environment step do
𝑙 𝑠 = 𝜓 𝑒 (Dt; 𝜃 𝑒 ) + 𝜓 𝑐 (P𝑡 , Y𝑡 ; 𝜃 𝑐 ) ⊲ Stock-level embedding
𝑟 = 𝑔(𝑟; 𝜇, 𝜃, 𝑎, 𝑏) ⊲ Sample a mask ratio
𝑙˜ 𝑠 = 𝜂 𝑚𝑜 (𝑙 𝑠 , 𝑟) ⊲ Masking operation
𝑙 𝑝 = 𝜓 𝑒𝑛𝑐 (𝑙˜ 𝑠 ; 𝜃 𝑒𝑛𝑐 ) ⊲ Pool-level embedding
𝜌 𝑡 = 𝜂 𝑚𝑓 (𝑙 𝑝 , 𝑚) ⊲ Maskable stock representation
𝑎 𝑡 ∼ 𝑅𝑒 (𝜋 𝜙 (𝑎 𝑡 |𝜌 𝑡 ),𝑇 ) ⊲ Sample action from the policy
𝑠 𝑡+1 = 𝑝 (𝑠 𝑡+1 |𝑠 𝑡 , 𝑎 𝑡 ) ⊲ Sample transition
D ∼ D ∪ {(𝑠 𝑡 , 𝑎 𝑡 , 𝜌 𝑡 , 𝑟 (𝑠 𝑡 , 𝑎 𝑡 ), 𝑠 𝑡+1 )} ⊲ Store transition
end for
for each gradient step do
𝜃 𝑖 ← 𝜃 𝑖 − 𝜆 𝑄 ∇ˆ 𝜃𝑖 𝐽 𝑄 (𝜃 𝑖 ) for 𝑖 ∈ 1, 2 ⊲ Update Q network
𝜙 ← 𝜙 − 𝜆 𝜋 ∇ˆ 𝜙 𝐽 𝜋 (𝜙) ⊲ Update policy network
𝛼 ← 𝛼 − 𝜆 𝛼 ∇ˆ 𝛼 𝐽 (𝛼) ⊲ Adjust alpha
𝜃¯ 𝑖 ← 𝜏𝜃 𝑖 + (1 − 𝜏)𝜃¯ 𝑖 for 𝑖 ∈ 1, 2 ⊲ Update target networks
𝜃 𝑒 , 𝜃 𝑐 , 𝜃 𝑒𝑛𝑐 , 𝜃 𝑑𝑒𝑐 ← 𝜆∇ˆ 𝐽 (𝜃 𝑒 , 𝜃 𝑐 , 𝜃 𝑒𝑛𝑐 , 𝜃 𝑑𝑒𝑐 )
end for
return 𝜃 1 , 𝜃 2 , 𝜃 𝑒 , 𝜃 𝑐 , 𝜃 𝑒𝑛𝑐 , 𝜃 𝑑𝑒𝑐 , 𝜙 ⊲ Optimized parameters
Algorithm 2 Inference of EarnMore
Require: Global Stock Pool 𝑈 , Customizable Stock Pools 𝐶𝑆𝑃𝑠 =
{𝐶 𝑡 |𝑡 = 1, 2, ...,𝑇 } ⊲ Input data
Ensure: {𝑊 𝑡 |𝑡 = 1, 2, ...,𝑇 } ⊲ Portfolios of CSPs
for each time step 𝑡 in {1, 2, ...,𝑇 } do
𝑠 𝑡 = [P𝑡 , Y𝑡 , D𝑡 ] ⊲ Initialize state
𝑙 𝑠 = 𝜓 𝑒 (Dt; 𝜃 𝑒 ) + 𝜓 𝑐 (P𝑡 , Y𝑡 ; 𝜃 𝑐 ) ⊲ Stock-level embedding
Initialize mask index 𝑀 according to CSP 𝑡
𝑙˜ 𝑠 = 𝜂 𝑚𝑜 (𝑙 𝑠 , 𝑀) ⊲ Masking operation
𝑙 𝑝 = 𝜓 𝑒𝑛𝑐 (𝑙˜ 𝑠 ; 𝜃 𝑒𝑛𝑐 ) ⊲ Encode pool-level embedding
𝜌 𝑡 = 𝜂 𝑚𝑓 (𝑙 𝑝 , 𝑚) ⊲ Maskable stock representation
𝑎 𝑡 = 𝑅𝑒 (𝜋 𝜙 (𝑎 𝑡 |𝜌 𝑡 ),𝑇 ) ⊲ Predict action from the policy
𝑊 𝑡 ← 𝑎 𝑡
end for

Wentao Zhang et al.
E DETAILS OF COMPARISON WITH THE
BASELINES
In this section, we conduct a comparative analysis of our method
EarnMore in comparison to 14 benchmark models. The analysis
is based on 6 key performance metrics applied to SP500 and DJ30
datasets. Specifically, these metrics include Average Rate of Return
(ARR) as an indicator of portfolio performance, along with risk-
adjusted measures like Sharpe Ratio (SR), Calmar Ratio (CR), and
Sortino Ratio (SoR). Additionally, we consider risk-related metrics,
Maximum Drawdown (MDD), and Volatility (VOL), to evaluate the
risk implications of the strategies. The backtesting was conducted
on 3 date splits, each with 3 seed values, and the reported metrics
represent the averages derived from 3 × 3 experiments. For detailed
information, please refer to Table ?? and Figure 6.
As shown in Table 5, our framework performance is evaluated
in comparison to all other methods. EarnMore stands out by sig-
nificantly improving its return potential across all datasets, while
maintaining a minimal loss in risk control. In Figure 6, we have
included comparative line diagrams of EarnMore and several other
methods in terms of cumulative returns. It is shown that EarnMore
demonstrates the best profit potential across all datasets.
As depicted in Figure 6(a) and 6(b), in October 2018, due to the
impact of the U.S. monetary policy and economic confrontation,
investor confidence in the stock markets is challenged, leading
to a downward trend in the U.S. stock market during this period.
EarnMore is impacted, resulting in a partial loss of returns, but it is
anticipated to recover quickly, maintaining its overall superiority
over other methods. The global COVID-19 pandemic reached its
highest point between February 14 and March 20, 2020, causing a
significant economic downturn and raising serious concerns among
investors. This resulted in a substantial decline in the stock market,
with the SP500 and DJ30 indices dropping by 31.81% and 34.78%,
respectively. As depicted in Figures 6(c) and 6(d), it’s clear that
EarnMore is much less affected in terms of returns compared to
the other methods. Moreover, it continues to gain profits after the
market starts to recover. Even during market downturns, EarnMore
shows an ability to pinpoint stocks with the potential for higher
returns when the market bounces back. Starting in 2021, the U.S.
stock market began its recovery, with the SP500 and DJ30 indices
generally showing an upward trend. From Figures 6(e) and 6(f),
it’s evident that EarnMore, compared to other methods, is better at
identifying stocks with upward momentum, maximizing returns. In
March 2022, due to geopolitical conflicts, there was a brief downturn
in the U.S. stock market. During this time, EarnMore is somewhat
affected, showing noticeable changes in returns in the SP500, but
still demonstrating a strong upward trend in the DJ30.
