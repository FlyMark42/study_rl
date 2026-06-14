# EarnMore 新版 `experiments/` 工程化工具链详解

> 面向想看懂并使用 `EarnMore-main_v20260610/experiments/` 这套**大规模超参实验工具链**的读者。配合 `04_运行调试测试指南.md`（怎么跑）一起看。
>
> **一句话定位**：`experiments/` 不是论文算法，而是一套**围绕 `tools/train.py` 的"实验自动化基础设施"**——批量生成实验、并发调度训练、实时监控早停、可视化对比、自动选出最优稳定模型。学算法可跳过；要做正经实验/调参必须懂它。

---

## 1. 全局：9 个文件 + 一条流水线

```
                ┌──────────────────────────────────────────────────────────┐
                │                  实验生命周期(从生成到选优)                  │
                └──────────────────────────────────────────────────────────┘

 ① 生成实验        ② 调度训练            ③ 监控/早停           ④ 可视化        ⑤ 选最优
┌───────────┐   ┌─────────────┐   ┌──────────────────┐   ┌───────────┐  ┌────────────────┐
│hyperopt.py│   │scheduler.py │   │monitor.py        │   │visualizer │  │best_model_     │
│experiment_│──▶│(并发跑很多   │──▶│alpha_monitor.py  │──▶│alpha_     │─▶│finder.py       │
│manager.py │   │ train.py子进程)│   │monitored_trainer │   │visualizer │  │(扫描/评稳定/排名)│
└───────────┘   └─────────────┘   └──────────────────┘   └───────────┘  └────────────────┘
   生成 exp_xxx/      读 exp_xxx/         train.py 把指标       读 logs/*.csv    读各 exp 指标,
   目录+config.py     config.py 起进程     写进 logs/*.csv       和 train_log     算稳定性, 选 best
```

| 文件 | 行数 | 角色 | 你会直接用的命令 |
|---|---|---|---|
| `experiment_manager.py` | 482 | 实验**生成/管理**：基于模板批量生成 `exp_xxx/config.py` | `list-experiments` / `generate-configs` |
| `hyperopt.py` | 1299 | **超参搜索**：随机搜索 + 贝叶斯优化(skopt) + 稳定性分析 | `random-search` / `bayesian-search` |
| `scheduler.py` | 567 | **调度器**：并发跑多个实验、资源控制、失败重试、断点恢复 | `run` / `status` |
| `monitor.py` | 931 | **训练监控**：记录指标到 CSV、价值函数/稳定性分析、早停、画训练曲线 | `status` / `comprehensive-analysis` |
| `alpha_monitor.py` | 596 | `monitor` 的**子类**，针对 ASR/ARR 指标定制早停与状态分析 | （被 monitored_trainer 内部用） |
| `monitored_trainer.py` | 130 | **桥接** `train.py` ↔ 监控器（记录指标、判早停） | （`train.py --monitor` 时自动用） |
| `best_model_finder.py` | 769 | **选最优**：扫描所有实验、评估稳定性、按验证指标排名、给报告 | `report` / `find-best` |
| `visualizer.py` | 882 | **可视化**：单实验出图 + 跨实验对比/排名/超参分析 | `plot-all-experiments` / `compare-experiments` |
| `alpha_visualizer.py` | 338 | `visualizer` 的**子类**，针对 ASR 指标定制可视化 | （被 visualizer 选用） |

> 设计模式：`alpha_*` 都是对应基类的子类（`TrainingMonitor_Alpha(TrainingMonitor)`、`ExperimentComparator_Alpha(ExperimentComparator)`），把"通用监控/可视化"特化到本项目的 ASR/ARR 指标上。读懂基类即懂子类。

---

## 2. 逐文件详解（含关键代码注释）

### 2.1 `experiment_manager.py` —— 实验的"生成器与花名册"
**核心类 `ExperimentManager`**，把"一个模板 + 若干超参取值"展开成一批独立实验，每个实验 = 一个目录 `experiments/exp_xxx/` + 一份 `config.py`。

关键方法：
- `generate_configs(template, **kwargs)`：核心。读模板的 `fixed_parameters`/`variable_parameters`，用 `itertools.product` 对变量参数**求笛卡尔积**，每个组合生成一个实验：
  ```python
  combinations = list(itertools.product(*param_values))   # 所有超参组合
  for i, combination in enumerate(combinations, 1):
      exp_name = "exp_{:03d}_{}_{}".format(i, template_name, "_".join(param_strs))  # exp_001_..._lr_5e-5_...
      exp_dir = os.path.join(self.experiments_dir, exp_name)
      os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
      self._generate_python_config(base_config, exp_config, exp_name, ...)  # 生成 config.py
  ```
- `_generate_python_config(...)`：**字符串替换法**生成 config.py——读基础配置 `configs/mask_sac_portfolio_management.py`，把里面的 `lr = 5e-5`、`batch_size = 32`、`num_episodes = 200`、各日期、`workdir`、`tag` 等**按本实验的取值替换**：
  ```python
  replacements['lr = 5e-5'] = 'lr = {}'.format(lr_val)   # 把基础配置里的 lr 改成本实验的
  replacements['workdir = "workdir"'] = 'workdir = "experiments/experiments/{}"'.format(exp_name)
  modified_content = base_content
  for old, new in replacements.items():
      modified_content = modified_content.replace(old, new)   # 逐项替换
  ```
  > 注意：这是**靠精确匹配字符串**实现的——所以基础配置里那几行必须长得跟 `replacements` 里的"旧值"一模一样，否则替换不上。改基础配置时要小心别破坏这些锚点。
- `list_experiments()` / `show_experiments_tree()`：花名册——读每个 `exp_xxx/results.json` 汇总状态(created/running/completed/failed)。
- 每个实验目录产出：`config.py`(训练用)、`config.json`(参数记录)、`results.json`(状态)、`logs/`(指标 CSV)。

**CLI**：
```bash
python experiment_manager.py list-templates           # 看可用模板
python experiment_manager.py show-template --name X    # 看某模板的固定/可变参数
python experiment_manager.py generate-configs --template mask_sac_base --lr 1e-4,5e-5 --seed 123,456
python experiment_manager.py list-experiments [--tree] # 看所有实验(可树状)
```

### 2.2 `hyperopt.py` —— 超参搜索（随机 + 贝叶斯）
比 `experiment_manager` 更上层：不用你手填超参，而是**自动采样**超参 + **自动切分训练/验证日期窗口**，再调用配置生成逻辑产出一批实验。四个核心类：

- **`ParameterSpaceManager`**：定义搜索空间（哪些超参、各自范围），并能 `detect_environment()` 按 GPU/CPU 给不同空间。
- **`RandomSearchOptimizer.random_search(...)`**：随机采样 `num_samples` 组超参；`_generate_date_splits(...)` 按"训练窗口/验证窗口/滑动步长"生成滚动日期切分（滚动回测）；`_sample_parameters(...)` 从空间里随机取值。
- **`BayesianOptimizer.bayesian_search(...)`**：用 `skopt`（scikit-optimize）做贝叶斯优化——`_build_skopt_space` 建搜索空间，按采集函数(EI 等)迭代，`resume_from_search` 可从某次搜索断点续跑。
- **`StabilityAnalyzer`**：把"同配置不同种子"的实验分组，算 ASR 方差、评估稳定性等级、出报告与图。

**CLI**（见 `quickhelp.txt`）：
```bash
python hyperopt.py detect-environment
python hyperopt.py show-space
python hyperopt.py random-search   --template mask_sac_base --num-samples 2 \
       --train-start-date 2021-10-01 --train-window-months 10 \
       --val-start-date 2022-10-01 --val-window-months 6 --slide-step-months 2
python hyperopt.py bayesian-search --template mask_sac_base --max-iter 2 --acquisition EI ...
python hyperopt.py resume-from --search-id bayesian_..._2 --template mask_sac_base
```
> 依赖 `scikit-optimize`（贝叶斯）。它最终也是落到"生成一堆 `exp_xxx/config.py`"，交给 `scheduler` 去跑。

### 2.3 `scheduler.py` —— 并发调度器
把 `experiments/` 下所有"待跑"实验排队，**并发地为每个实验起一个 `python train.py` 子进程**，并管理资源/重试/恢复。五个核心类：

- **`TaskQueue`**：`scan_experiments()` 扫描所有 `exp_xxx/` 的状态文件，维护 pending/running/completed/failed 队列；`update_task_status()` 改状态。
- **`ResourceManager`**：`check_available_slots()` 按 `max_concurrent` 和 CPU/内存阈值(默认 80%)决定还能起几个进程。
- **`RetryManager`**：失败实验按 `max_retries` 重试。
- **`RecoveryManager`**：启动时 `recover_interrupted_experiments()`——把上次中断(状态卡在 running 但进程已死)的实验重置为 pending。
- **`ExperimentScheduler`**：主循环：
  ```python
  def run(self):
      self.recovery_manager.recover_interrupted_experiments()   # 先恢复中断的
      while not self.should_stop:
          self.task_queue.scan_experiments()                    # 扫描状态
          self._process_running_experiments()                   # 收割已结束进程(成功→completed, 失败→重试/failed)
          self._start_new_experiments()                         # 有空位就起新实验
          time.sleep(10)
  def _start_single_experiment(self, task):
      cmd = [sys.executable, g_strTrainCmd, "--config", config_file, "--root", g_strRoot, "--monitor"]
      process = subprocess.Popen(cmd, stdout=log_file, stderr=STDOUT)   # ★起一个 train.py 子进程
      self.running_processes[exp_name] = process
  ```

**CLI**：
```bash
python scheduler.py --traincmd "<绝对路径>/tools/train.py" run --max-concurrent 1 --max-retries 3
python scheduler.py status                      # 看队列状态
python scheduler.py set-priority --exp X --priority 1
python scheduler.py rerunall                    # 重跑所有
```
> ⚠️ `g_strTrainCmd`(train.py 路径)、`g_strRoot`(项目根)需改成你机器的实际**绝对路径**。它给每个子进程都加了 `--monitor`，所以训练会自动接监控器。

### 2.4 `monitor.py` + `alpha_monitor.py` —— 训练监控与早停
**`TrainingMonitor`（基类）**：
- `record_metrics(exp_name, episode, metrics_dict)`：把每个 episode 的训练指标**追加写到** `exp_dir/logs/training_metrics.csv`。
- `record_value_function_metrics(...)`：记录 Q 值/TD 误差/策略熵/权重范数等到 `value_function_metrics.csv`。
- `analyze_value_function_quality()` / `analyze_stability()`：读 CSV 做事后分析。
- `early_stop_check()`：基于指标趋势判断是否该早停。
- `TrainingVisualizer.plot_training_curves()`：画训练曲线。

**`TrainingMonitor_Alpha(TrainingMonitor)`（子类，实际被用）**：重写 `early_stop_check(exp_name, patience, metrics='ASR')` / `status` / `analyze_stability`，针对本项目的 **ASR/ARR** 指标定制——`patience` 轮内验证指标不创新高就建议早停。

**CLI**：
```bash
python monitor.py status --experiment exp_001_xxx
python monitor.py comprehensive-analysis --experiment exp_001_xxx
python monitor.py analyze-stability --experiment exp_001_xxx
python monitor.py early-stop-check --experiment exp_001_xxx
python monitor.py plot-training --experiment exp_001_xxx
```

### 2.5 `monitored_trainer.py` —— train.py 与监控器的桥
**`MonitoredTrainer`**：`train.py --monitor` 时被实例化（内部用 `TrainingMonitor_Alpha`）。三个接口在 `train.py` 主循环里被调用：
- `log_episode_metrics(metrics_dict)` → 训练每 episode 调，写指标。
- `log_value_function_metrics(episode, agent)` → 从 agent 提取 Q 值/TD 误差/熵/权重范数。
- `should_stop_early(...)` → 验证后调，返回是否早停。
> 逐行注释见 `带详细注释的核心代码/experiments_注释版/monitored_trainer_注释版.py`。

### 2.6 `best_model_finder.py` —— 自动选出"最稳定最优"模型
训练全部跑完后用它。**`StableBestModelFinder`** 流程：
- `scan_all_experiments()`：扫描所有 `exp_xxx/`。
- `evaluate_stability(exp_name, path)` → `_calculate_training_health` + `_calculate_stability_score`：用"训练健康度 + 早停数据"给每个实验打**稳定性分**（避免选到"昙花一现"的不稳定模型）。
- `filter_stable_models(min_score)`：筛掉不稳定的。
- `collect_validation_metrics(...)` + `rank_by_metric(metric='val_ASR', ...)`：在稳定模型里按验证指标排名。
- `find_best_stable_model(...)` / `generate_best_models_report(...)`：给出最优模型与报告。

**CLI**（运行前需设 `g_episodeCount/g_patience/g_metrics` 等前提变量，见 `quickhelp.txt`）：
```bash
python best_model_finder.py evaluate-all
python best_model_finder.py list-stable
python best_model_finder.py find-best
python best_model_finder.py --min-stability 30 report     # 生成完整分析报告
```

### 2.7 `visualizer.py` + `alpha_visualizer.py` —— 可视化
- **`TrainingVisualizer`（单实验）**：读 `logs/*.csv` 与 `train_log.txt`，画 loss / reward / validation / learning / value_function / portfolio 曲线；`generate_all_plots()` 一次出全。
- **`ExperimentComparator`（跨实验）**：`plot_experiments_comparison(metric)`、`generate_performance_ranking()`、`generate_hyperparameter_analysis()`（从实验名解析超参，分析"超参 vs 性能"）。
- **`alpha_visualizer`**：上述两者的 `_Alpha` 子类，针对 ASR 指标定制。

**CLI**：
```bash
python visualizer.py plot-all-experiments --experiments-dir experiments     # 每个实验都出图
python visualizer.py compare-experiments --experiments-dir experiments       # 跨实验对比/排名/超参分析
```

---

## 3. 端到端工作流（把命令串起来）

```bash
cd experiments

# ① 生成一批实验(随机搜索: 2 组超参 × 滚动日期切分)
python hyperopt.py random-search --template mask_sac_base --num-samples 2 \
       --train-start-date 2021-10-01 --train-window-months 10 \
       --val-start-date 2022-10-01 --val-window-months 6 --slide-step-months 2

# ② 并发调度训练(每个实验一个 train.py 子进程, 自动重试/恢复)
python scheduler.py --traincmd "<abs>/tools/train.py" run --max-concurrent 1 --max-retries 3

# ③ 训练中: 实时监控 / 综合分析
python monitor.py status --experiment exp_001_xxx
python monitor.py comprehensive-analysis --experiment exp_001_xxx

# ④ 训练后: 批量可视化对比
python visualizer.py plot-all-experiments --experiments-dir experiments
python visualizer.py compare-experiments --experiments-dir experiments

# ⑤ 选出最稳定最优模型
python best_model_finder.py --min-stability 30 report
```
> 想继续下一轮：再 `random-search` 加样本，或 `hyperopt.py resume-from --search-id <贝叶斯搜索id>` 继续贝叶斯优化。

---

## 4. 实验目录长什么样（产出物）

```
experiments/
├── templates/                     # 实验模板(json: fixed_parameters/variable_parameters)
├── summary.json                   # 所有实验汇总(experiment_manager 维护)
├── exp_001_mask_sac_base_lr_5e-5_..../
│   ├── config.py                  # ★本实验的训练配置(scheduler 用它起 train.py)
│   ├── config.json                # 参数记录
│   ├── results.json               # 状态(created/running/completed/failed)
│   ├── training.log               # 子进程 stdout/stderr
│   ├── best.pth                   # 本实验验证最优模型
│   ├── train_log.txt / train_infos.txt
│   └── logs/
│       ├── training_metrics.csv           # monitor 写: 每 episode 训练指标
│       ├── value_function_metrics.csv     # Q值/TD误差/熵/权重范数
│       └── portfolio_daily_data.csv       # 组合日数据
└── exp_002_.../ ...
```

---

## 5. 学习与使用建议

- **只想学算法**：跳过本目录，看 `03` + `带详细注释的核心代码/` 即可。
- **要做实验/调参**：先用 `04` 文档跑通**单训练**，再按上面"端到端工作流"走一遍 `hyperopt → scheduler → monitor → visualizer → best_model_finder`。
- **改基础配置时小心**：`experiment_manager._generate_python_config` 用**字符串精确替换**注参，别动那几行的写法（如 `lr = 5e-5`、`workdir = "workdir"`），否则注参失败。
- **路径问题是最大坑**：`scheduler` 的 `g_strTrainCmd/g_strRoot`、`hyperopt` 的日期参数都要按你机器改；`best_model_finder` 运行前要设前提变量。

> 逐行注释：`带详细注释的核心代码/experiments_注释版/` 下提供 `monitored_trainer_注释版.py`、`experiment_manager_注释版.py` 的完整注释；其余大文件(plot/分析为主)按"类→方法→关键块"在本文档第 2 节注释，配合源码阅读即可。
