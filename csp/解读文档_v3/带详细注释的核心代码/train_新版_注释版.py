# -*- coding: utf-8 -*-
"""
==========================================================================================
训练主循环 tools/train.py —— 逐行详细中文注释版（学习用）
对应原文件：EarnMore-main_v20260610/tools/train.py
==========================================================================================

【这份文件做什么】
它是整个工程的"训练总指挥"，把"数据集 → 环境 → 智能体 → 回放缓冲区"四大件造出来，
再反复执行"采样 → 更新网络 → 验证 → 存最优模型"的循环，对应论文【算法 1：训练】。

【新版相对旧版的改动（约 19 行）】
  ① 去掉了旧版末尾用 send_email(...) 包裹 main() 的 try/except（训练完/出错发邮件），新版直接 main(args)，更干净。
  ② 新增"根路径解析"：把相对 --root 转成绝对路径（第 67~71 行）。
  ③ MonitoredTrainer 从 experiments.monitored_trainer 导入（旧版从 pm.utils；配合工程工具搬进 experiments/）。
  ④ 工具函数从 tools.tools_utils 导入（旧版叫 tools.utils）。
  主循环逻辑本身没变，仍是论文算法 1 的实现。

【对应论文】
  - 整个 main() 的 for episode 循环 = 算法 1 的外层循环。
  - train_one_episode 里的 explore_env + update_net = 算法 1 的"采样"与"梯度更新"两个 for。
==========================================================================================
"""

import warnings
warnings.filterwarnings("ignore")               # 屏蔽各种告警，让训练日志干净
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 显存碎片整理，缓解大模型 OOM
import sys
from pathlib import Path
import torch
from mmengine.config import Config, DictAction  # mmengine 的配置系统（Registry+Config 那一套）
import numpy as np
import random
import json
import argparse
from iopath.common.file_io import g_pathmgr as pathmgr  # 统一的文件读写管理器
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter        # TensorBoard 记录训练曲线
import gym
from copy import deepcopy

ROOT = str(Path(__file__).resolve().parents[1])  # 项目根目录（train.py 上一级的上一级）
sys.path.append(ROOT)

from tools.tools_utils import check_operating_system   # ← 新版：工具模块改名 tools_utils（旧版 utils）

# 从注册器导入三大件的"工厂"
from pm.registry import ENVIRONMENT
from pm.registry import AGENT
from pm.registry import DATASET
from pm.utils import update_data_root
from pm.utils import ReplayBuffer
from pm.utils import load_checkpoint
from pm.utils import save_checkpoint
from pm.utils import find_latest_checkpoint
from pm.utils import print_table
from pm.utils import plot_metrics


def helloworld():
    print("Hello, World!")


def init_before_training(seed=3407):
    """固定所有随机种子 + 设定 cudnn 确定性，保证实验可复现。"""
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)              # CPU 随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False        # 关掉自动寻优算法（否则结果不确定）
    torch.backends.cudnn.deterministic = True     # 确定性模式
    torch.set_default_dtype(torch.float32)


def make_env(env_id, env_params):
    """gym 向量环境要求的"环境工厂闭包"：返回一个无参函数，调用时才真正造环境。"""
    def thunk():
        env = gym.make(env_id, **env_params)
        return env
    return thunk


def main(args):
    # ===================== ① 读配置 + 处理命令行覆盖项 =====================
    cfg = Config.fromfile(args.config)            # 把 configs/xxx.py 读成配置对象

    if args.cfg_options is None:
        args.cfg_options = dict()
    if args.root is not None:
        # ← 新版新增：把相对根路径转成绝对路径
        root_path = Path(args.root)
        if not root_path.is_absolute():
            args.root = str((Path(ROOT) / root_path).resolve())
        args.cfg_options["root"] = args.root
    if args.workdir is not None:
        args.cfg_options["workdir"] = args.workdir
    if args.tag is not None:
        args.cfg_options["tag"] = args.tag
    cfg.merge_from_dict(args.cfg_options)         # 把命令行覆盖项合并进配置
    print(cfg)

    # ===================== ② 可选：初始化训练监控器 =====================
    monitor = None
    if args.monitor:
        sys.path.append(ROOT)
        from experiments.monitored_trainer import MonitoredTrainer   # ← 新版：从 experiments/ 导入
        exp_name = os.path.splitext(os.path.basename(args.config))[0]
        if 'exp_' not in exp_name:
            exp_name = os.path.basename(os.path.dirname(args.config))
        monitor = MonitoredTrainer(exp_name, enable_monitoring=True)
        print("Monitoring enabled for experiment: {}".format(exp_name))

    update_data_root(cfg, root=args.root)         # 把配置里的相对数据路径补成绝对路径
    init_before_training(cfg.seed)                # 固定随机种子

    exp_path = os.path.join(cfg.root, cfg.workdir)   # 本次实验的工作目录（存日志/模型）
    os.makedirs(exp_path, exist_ok=True)
    writer = SummaryWriter(exp_path)              # TensorBoard

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ===================== ③ 造数据集 =====================
    print(50 * "-" + "build dataset" + "-" * 50)
    dataset = DATASET.build(cfg.dataset)          # → PortfolioManagementDataset（读股票/特征/子池mask）

    # ===================== ④ 造训练环境（向量化，可并行多子池） =====================
    print(50 * "-" + "build train enviroment" + "-" * 50)
    cfg.environment.update(dict(                  # 给环境补上训练相关参数
        mode="train",
        if_norm=True,
        dataset=dataset,
        start_date=cfg.train_start_date,
        end_date=cfg.train_end_date
    ))
    train_environment = ENVIRONMENT.build(cfg.environment)   # → EnvironmentASR（★新版=A股真实交易模拟器）
    # 用 SyncVectorEnv 把 num_envs 个环境副本包成"向量环境"，一次 step 推进所有副本
    train_envs = gym.vector.SyncVectorEnv(
        [make_env("PortfolioManagement-v0",
                  env_params=dict(env=deepcopy(train_environment),
                                  transition_shape=cfg.transition_shape,
                                  seed=cfg.seed + i)) for i in range(cfg.num_envs)]
    )

    # ===================== ⑤ 造验证环境（每个 CSP 子池一个环境） =====================
    print(50 * "-" + "build val enviroment" + "-" * 50)
    cfg.environment.update(dict(
        mode="val",
        if_norm=True,
        dataset=dataset,
        scaler=train_environment.scaler,          # ★复用训练集 scaler，防止数据泄漏
        start_date=cfg.val_start_date,
        end_date=cfg.val_end_date
    ))
    val_environment = ENVIRONMENT.build(cfg.environment)
    # 验证环境数量 = 子池数量（len(aux_stocks)）：每个 CSP 各建一个环境，分别评估模型对不同子池的适应力
    val_envs = gym.vector.SyncVectorEnv(
        [make_env("PortfolioManagement-v0",
                  env_params=dict(env=deepcopy(val_environment),
                                  transition_shape=cfg.transition_shape)) for i in range(len(val_environment.aux_stocks))]
    )
    # （测试环境的构建被注释掉了，需要时打开）

    # ===================== ⑥ 造智能体 =====================
    print(50 * "-" + "build agent" + "-" * 50)
    cfg.agent.update(dict(device=device))
    agent = AGENT.build(cfg.agent)                # → AgentMaskSAC（GPU）或 AgentSAC（CPU）

    # 初始化 agent.last_state（智能体记住"当前状态"，供 explore_env 连续交互）
    state, _ = train_envs.reset(seed=cfg.seed)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    agent.last_state = state

    # ===================== ⑦ 造回放缓冲区 + 预填充 =====================
    buffer = ReplayBuffer(
        buffer_size=cfg.buffer_size,
        transition=cfg.transition,
        transition_shape=cfg.transition_shape,
        if_use_per=cfg.if_use_per,                # 是否用优先经验回放（默认关）
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    buffer_items = agent.explore_env(train_envs, cfg.horizon_len)   # 先采一批交互数据
    buffer.update(buffer_items)                                     # 填进缓冲区

    max_metrics = -np.inf                         # 记录验证集最佳指标（用于存最优模型）

    # ===================== ⑧ 断点续训：找最新 checkpoint =====================
    latest_path = find_latest_checkpoint(exp_path, suffix="pth")
    if latest_path:
        start_episode = load_checkpoint(agent, latest_path)
    else:
        start_episode = 0

    # 清掉旧的 CSV 日志，保证本次实验写新文件
    logs_dir = os.path.join(exp_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    csv_files = [
        os.path.join(logs_dir, "training_metrics.csv"),
        os.path.join(logs_dir, "value_function_metrics.csv"),
        os.path.join(logs_dir, "portfolio_daily_data.csv")
    ]
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            os.remove(csv_file)
            print(f"删除旧数据文件: {csv_file}")

    print("start episode {}, end episode {}".format(start_episode + 1, cfg.num_episodes))

    # ===================== ⑨ 主循环：逐 episode 训练 + 验证 =====================
    horizon_step = 0
    for episode in range(start_episode + 1, cfg.num_episodes + 1):
        infos = {"episode": [episode]}
        episode_stats_log = {"episode": [episode]}

        # ---------- 训练一个 episode ----------
        print("Train Episode: [{}/{}]".format(episode, cfg.num_episodes))
        train_stats, train_infos = train_one_episode(train_envs, buffer, agent, cfg.horizon_len)

        horizon_stats = train_stats["horizon_stats"]   # 每个 horizon 段的统计（loss 等）
        episode_stats = train_stats["episode_stats"]   # 整个 episode 的平均统计

        # 写 TensorBoard
        for k, v in horizon_stats.items():
            for item in v:
                writer.add_scalar("train/horizon_{}".format(k), item, horizon_step)
                horizon_step += 1
        for k, v in episode_stats.items():
            writer.add_scalar("train/episode_{}".format(k), v, episode)

        # 可选：把训练指标（含学习率/梯度范数/beta/rep 损失等）喂给监控器
        if monitor:
            enhanced_train_metrics = {"episode": episode}
            enhanced_train_metrics.update({"train_{}".format(k): v for k, v in episode_stats.items()})
            try:
                if 'obj_critics' in episode_stats:
                    enhanced_train_metrics['reward'] = -episode_stats['obj_critics']   # 用 -critic_loss 当 reward 代理
                if 'alphas' in episode_stats:
                    enhanced_train_metrics['exploration_rate'] = episode_stats['alphas']   # alpha≈探索强度
                if hasattr(agent, 'act_optimizer'):
                    enhanced_train_metrics['actor_learning_rate'] = agent.act_optimizer.param_groups[0]['lr']
                if hasattr(agent, 'cri_optimizer'):
                    enhanced_train_metrics['critic_learning_rate'] = agent.cri_optimizer.param_groups[0]['lr']
                if 'gradient_norms' in episode_stats:
                    enhanced_train_metrics['gradient_norm'] = episode_stats['gradient_norms']
                if 'beta_losses' in episode_stats:     # 掩码监督惩罚损失（论文 §4.2）
                    enhanced_train_metrics['beta_loss'] = episode_stats['beta_losses']
                if 'rep_losses' in episode_stats:      # MAE 重构损失（论文公式 8）
                    enhanced_train_metrics['representation_loss'] = episode_stats['rep_losses']
            except Exception as e:
                print("Warning: Could not collect additional training metrics: {}".format(e))
            monitor.log_episode_metrics(enhanced_train_metrics)
            monitor.log_value_function_metrics(episode, agent)

        # 打印训练指标表
        train_episode_stats_log = OrderedDict({
            "episode": [episode],
            **{f"train_{k}": ["{:04f}".format(v)] for k, v in episode_stats.items()},
        })
        episode_stats_log.update(train_episode_stats_log)
        infos.update(train_infos)
        print(print_table(train_episode_stats_log))

        # ---------- 验证一个 episode ----------
        print("Validate Episode: [{}/{}]".format(episode, cfg.num_episodes))
        val_stats, val_infos = validate(val_envs, agent)
        episode_stats = val_stats["episode_stats"]

        # 兼容不同 Agent 的指标键名：MaskSAC 用 ASR_env0（夏普），标准 SAC 用 ARR%_env0
        # env0 即第 0 个验证环境 = 完整 GSP（dataset 里 aux_stocks[0]=All）
        if "ASR_env0" in episode_stats:
            metric = np.mean([episode_stats["ASR_env0"]])
        elif "ARR%_env0" in episode_stats:
            metric = np.mean([episode_stats["ARR%_env0"]])
        elif "ARR_env0" in episode_stats:
            metric = np.mean([episode_stats["ARR_env0"]])
        else:
            assert(False)
            available_metrics = [k for k in episode_stats.keys() if k.endswith("_env0")]
            metric = np.mean([episode_stats[available_metrics[0]]]) if available_metrics else 0.0

        # ★验证指标创新高 → 存最优模型 best.pth（论文算法 1 之外的工程实践）
        if metric > max_metrics:
            max_metrics = metric
            save_checkpoint(episode, agent, exp_path, if_best=True)

        # 可选：早停（连续若干 episode 不提升则停）
        if monitor and monitor.should_stop_early(min_episodes=50):
            print("Early stopping triggered at episode {}".format(episode))
            break

        # 写验证指标到 TensorBoard / 监控器 / 日志表
        episode_stats = val_stats["episode_stats"]
        for k, v in episode_stats.items():
            writer.add_scalar("val/episode_{}".format(k), v, episode)
        if monitor:
            enhanced_val_metrics = {}
            enhanced_val_metrics.update({"val_{}".format(k): v for k, v in episode_stats.items()})
            if "ASR_env0" in episode_stats:
                enhanced_val_metrics['validation_asr'] = episode_stats["ASR_env0"]
            elif "ARR%_env0" in episode_stats:
                enhanced_val_metrics['validation_asr'] = episode_stats["ARR%_env0"]
            elif "ARR_env0" in episode_stats:
                enhanced_val_metrics['validation_asr'] = episode_stats["ARR_env0"]
            if "SR_env0" in episode_stats:
                enhanced_val_metrics['validation_reward'] = episode_stats["SR_env0"]
            monitor.log_episode_metrics(enhanced_val_metrics)

        val_episode_log_stats = OrderedDict({
            "episode": [episode],
            **{f"val_{k}": ["{:04f}".format(v)] for k, v in episode_stats.items()},
        })
        episode_stats_log.update(val_episode_log_stats)
        infos.update(val_infos)
        print(print_table(val_episode_log_stats))

        # ---------- 把本 episode 的统计/详情写进日志文件 ----------
        def to_json_serializable(obj):
            """递归把 numpy.ndarray / torch.Tensor 转成 list，方便 json.dumps。"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            try:
                import torch
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().numpy().tolist()
            except ImportError:
                pass
            if isinstance(obj, dict):
                return {k: to_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_json_serializable(v) for v in obj]
            return obj

        with pathmgr.open(os.path.join(exp_path, "train_log.txt"), "a") as op:
            op.write(json.dumps(to_json_serializable(episode_stats_log), ensure_ascii=False) + "\n")
        with pathmgr.open(os.path.join(exp_path, "train_infos.txt"), "a") as op:
            op.write(json.dumps(to_json_serializable(infos), ensure_ascii=False) + "\n")
    # （末尾用 best.pth 跑测试集的代码被注释掉了，需要时打开）


def train_one_episode(environment, buffer, agent, horizon_len):
    """训练一个 episode：反复"采数据→更新网络"，直到环境结束(done)。对应论文算法 1 的两个 for。"""
    infos = dict()
    stats = {"episode_stats": {}, "horizon_stats": {}}

    environment.reset()                           # 重置环境到新起点

    while True:
        # ① 采样：智能体与环境交互 horizon_len 步，产出一批 transition
        buffer_items = agent.explore_env(environment, horizon_len)
        buffer.update(buffer_items)               # 存入缓冲区

        # ② 学习：从缓冲区抽样、更新五件套网络（Critic→Alpha→Actor→Beta→Rep）
        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)  # ★核心，对应论文公式 (5)(6)(7)(8)
        torch.set_grad_enabled(False)

        # ③ 判断这批数据里是否出现 done（环境到末尾）。buffer_items[-2] 是 dones
        positive_indices = torch.nonzero(buffer_items[-2] > 0)
        if positive_indices.numel() == 0:
            min_row_index = horizon_len - 1       # 没 done
        else:
            min_row_index = torch.min(positive_indices[:, 0]).item()   # 第一个 done 的位置

        # 收集本段的各项 loss
        for k, v in logging_tuple.items():
            stats["horizon_stats"].setdefault("{}".format(k), []).append(v)

        if min_row_index < horizon_len - 1:       # 出现 done → 本 episode 结束
            break

    # 把每段统计取平均，作为 episode 级统计
    for k, v in stats["horizon_stats"].items():
        stats["episode_stats"][k] = np.mean(v)
    return stats, infos


def validate(environment, agent):
    """验证：在验证环境(各 CSP 子池)上跑一遍，让 agent.validate_net 算 ARR/SR/MDD 等指标。对应论文算法 2(推理)。"""
    stats = {"episode_stats": {}}
    logging_tuple, infos = agent.validate_net(environment)
    for k, v in logging_tuple.items():
        stats["episode_stats"][k] = v
    return stats, infos


if __name__ == '__main__':
    # 命令行入口：python tools/train.py --config configs/mask_sac_portfolio_management.py --root .
    parser = argparse.ArgumentParser(description='训练脚本')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--monitor', action='store_true', help='启用训练监控')
    parser.add_argument('--root', help='项目根目录')
    parser.add_argument('--workdir', help='工作目录')
    parser.add_argument('--tag', help='实验标签')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='配置选项')
    args = parser.parse_args()
    main(args)            # ← 新版：直接调用（旧版这里用 try/except 包了 send_email 通知）


# ==========================================================================================
# 【训练全景图（一句话串起来）】
#   读配置 → 造 Dataset → 造 train/val 向量环境 → 造 AgentMaskSAC → 造 ReplayBuffer → 预填充
#   for episode:
#       train_one_episode:  while 未done: explore_env(采样) → buffer.update → update_net(五件套更新)
#       validate:           在各 CSP 子池上 validate_net，算 ASR/ARR/SR/MDD
#       若 env0(完整GSP) 指标创新高 → 存 best.pth
#   → 全程写 TensorBoard + 日志 + 可选监控器
#
# 【对应论文】这整段就是算法 1(训练) + 算法 2(推理/验证) 的工程实现；
#            真正体现论文公式的核心在 agent.update_net()（见 mask_sac_注释版.py）。
# ==========================================================================================
