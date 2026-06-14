# `experiments/` 工程化工具链 · 注释说明

> 本子目录配合 `05_experiments工程化工具详解.md` 阅读。`experiments/` 是围绕 `tools/train.py` 的**大规模实验自动化基础设施**（生成实验→并发调度→监控早停→可视化→选最优），不是论文算法。

## 注释覆盖策略

`experiments/` 共 9 个文件、约 6000 行，其中大量是绘图/CSV 读写/分析的工程代码。为高效学习，本目录采用**两级注释**：

| 级别 | 范围 | 形式 |
|---|---|---|
| **逐行注释** | 2 个最核心、最可读的入口文件 | 本目录的 `*_注释版.py` |
| **类→方法→关键块注释** | 其余 7 个大文件（绘图/分析为主） | `05_experiments工程化工具详解.md` 第 2 节（含关键代码片段注释） |

### 逐行注释文件（本目录）
| 注释文件 | 对应原文件 | 角色 |
|---|---|---|
| `monitored_trainer_注释版.py` | `experiments/monitored_trainer.py` | train.py ↔ 监控器 的桥接（最先看） |
| `experiment_manager_注释版.py` | `experiments/experiment_manager.py` | 实验生成器（笛卡尔积展开 + 字符串替换注参） |

### 在 `05` 文档里详解（含关键代码注释）的文件
`hyperopt.py`（超参搜索）、`scheduler.py`（并发调度）、`monitor.py` + `alpha_monitor.py`（监控/早停）、`best_model_finder.py`（选最优）、`visualizer.py` + `alpha_visualizer.py`（可视化）。

## 阅读顺序
1. 先读 `05` 文档第 1 节（流水线全景）+ 第 3 节（端到端工作流）。
2. 看 `monitored_trainer_注释版.py`——理解训练与监控如何挂钩。
3. 看 `experiment_manager_注释版.py`——理解"一个模板怎么变成一批实验"。
4. 回 `05` 第 2 节，按"类→方法"读懂 scheduler / hyperopt / monitor / best_model_finder。
5. 想动手跑：转 `04_运行调试测试指南.md` 第 4 节。

> **最大的坑**（务必记住）：`experiment_manager._generate_python_config` 用**字符串精确替换**注入超参——基础配置 `configs/mask_sac_portfolio_management.py` 里必须保留 `lr = 5e-5`、`workdir = "workdir"` 等"锚点行"的原样写法，否则注参会悄悄失败、实验用默认值。
