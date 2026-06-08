import warnings

warnings.filterwarnings("ignore")
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
from pathlib import Path
import torch
from mmengine.config import Config, DictAction
import numpy as np
import random
import json
import argparse
from iopath.common.file_io import g_pathmgr as pathmgr
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import gym
from copy import deepcopy

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from tools.tools_utils import check_operating_system

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
from utils import send_email

def helloworld():
    print("Hello, World!")

def init_before_training(seed = 3407):
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        # 设置CPU的随机种子
        torch.manual_seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)

def make_env(env_id, env_params):
    def thunk():
        env = gym.make(env_id, **env_params)
        return env
    return thunk

def main(args):

    cfg = Config.fromfile(args.config)

    if args.cfg_options is None:
        args.cfg_options = dict()
    if args.root is not None:
        args.cfg_options["root"] = args.root
    if args.workdir is not None:
        args.cfg_options["workdir"] = args.workdir
    if args.tag is not None:
        args.cfg_options["tag"] = args.tag
    cfg.merge_from_dict(args.cfg_options)
    print(cfg)
    
    # 初始化监控器
    monitor = None
    if args.monitor:
        sys.path.append(ROOT)
        from pm.utils.monitored_trainer import MonitoredTrainer
        # 从配置文件路径推导实验名称
        exp_name = os.path.splitext(os.path.basename(args.config))[0]
        if 'exp_' not in exp_name:
            exp_name = os.path.basename(os.path.dirname(args.config))
        monitor = MonitoredTrainer(exp_name, enable_monitoring=True)
        print("Monitoring enabled for experiment: {}".format(exp_name))

    update_data_root(cfg, root=args.root)

    init_before_training(cfg.seed)

    exp_path = os.path.join(cfg.root, cfg.workdir)
    os.makedirs(exp_path, exist_ok=True)

    writer = SummaryWriter(exp_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(50 * "-" + "build dataset" + "-" * 50)
    dataset = DATASET.build(cfg.dataset)

    print(50 * "-" + "build train enviroment" + "-" * 50)
    cfg.environment.update(dict(
        mode = "train",
        if_norm = True,
        dataset = dataset,
        start_date = cfg.train_start_date,
        end_date = cfg.train_end_date
    ))
    train_environment = ENVIRONMENT.build(cfg.environment)
    train_envs = gym.vector.SyncVectorEnv(
                                          [make_env("PortfolioManagement-v0",
                                           env_params=dict(env = deepcopy(train_environment),
                                           transition_shape = cfg.transition_shape, seed = cfg.seed + i)) for i in range(cfg.num_envs)]
    )

    print(50 * "-" + "build val enviroment" + "-" * 50)
    cfg.environment.update(dict(
        mode="val",
        if_norm = True,
        dataset = dataset,
        scaler = train_environment.scaler,
        start_date=cfg.val_start_date,
        end_date=cfg.val_end_date
    ))
    val_environment = ENVIRONMENT.build(cfg.environment)
    val_envs = gym.vector.SyncVectorEnv(
        [make_env("PortfolioManagement-v0",
                  env_params=dict(env=deepcopy(val_environment),
                                  transition_shape=cfg.transition_shape)) for i in range(len(val_environment.aux_stocks))]
    )
    #print(len(val_environment.aux_stocks))

    # print(50 * "-" + "build test enviroment" + "-" * 50)
    # cfg.environment.update(dict(
    #     mode="test",
    #     if_norm=True,
    #     dataset=dataset,
    #     scaler=train_environment.scaler,
    #     start_date=cfg.test_start_date,
    #     end_date=getattr(cfg, "test_end_date", None)
    # ))
    # test_environment = ENVIRONMENT.build(cfg.environment)
    # test_envs = gym.vector.SyncVectorEnv(
    #     [make_env("PortfolioManagement-v0",
    #               env_params=dict(env=deepcopy(test_environment),
    #                               transition_shape=cfg.transition_shape)) for i in range(len(test_environment.aux_stocks))]
    # )

    print(50 * "-" + "build agent" + "-" * 50)
    cfg.agent.update(dict(device = device))
    agent = AGENT.build(cfg.agent)

    '''init agent.last_state'''
    state, _ = train_envs.reset(seed = cfg.seed)
    state = torch.tensor(state, dtype=torch.float32, device = device).unsqueeze(0)
    agent.last_state = state

    '''init buffer'''
    buffer = ReplayBuffer(
        buffer_size = cfg.buffer_size,
        transition = cfg.transition,
        transition_shape = cfg.transition_shape,
        if_use_per = cfg.if_use_per,
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # device=torch.device("cpu")
    )

    buffer_items = agent.explore_env(train_envs, cfg.horizon_len)
    buffer.update(buffer_items)

    max_metrics = -np.inf

    #print(cfg)

    latest_path = find_latest_checkpoint(exp_path, suffix="pth")
    if latest_path:
        start_episode = load_checkpoint(agent, latest_path)
    else:
        start_episode = 0

    # 清除旧的CSV文件，确保实验开始时创建新文件
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

    horizon_step = 0
    for episode in range(start_episode + 1, cfg.num_episodes + 1):
        infos = {"episode": [episode]}
        episode_stats_log = {"episode": [episode]}

        ######################train######################
        print("Train Episode: [{}/{}]".format(episode, cfg.num_episodes))
        train_stats, train_infos = train_one_episode(train_envs, buffer, agent, cfg.horizon_len)

        horizon_stats = train_stats["horizon_stats"]
        episode_stats = train_stats["episode_stats"]

        for k, v in horizon_stats.items():
            for item in v:
                writer.add_scalar("train/horizon_{}".format(k), item, horizon_step)
                horizon_step += 1
        for k, v in episode_stats.items():
            writer.add_scalar("train/episode_{}".format(k), v, episode)

        # 记录训练指标到监控器
        if monitor:
            # 基础训练指标
            enhanced_train_metrics = {"episode": episode}
            enhanced_train_metrics.update({"train_{}".format(k): v for k, v in episode_stats.items()})

            # 添加额外指标
            try:
                # 尝试获取reward信息
                if 'obj_critics' in episode_stats:
                    enhanced_train_metrics['reward'] = -episode_stats['obj_critics']  # critic loss的负值作为reward代理

                # 尝试获取探索率信息（从alphas字段获取）
                if 'alphas' in episode_stats:
                    enhanced_train_metrics['exploration_rate'] = episode_stats['alphas']

                # 尝试获取学习率信息
                if hasattr(agent, 'act_optimizer') and hasattr(agent.act_optimizer, 'param_groups'):
                    enhanced_train_metrics['actor_learning_rate'] = agent.act_optimizer.param_groups[0]['lr']
                if hasattr(agent, 'cri_optimizer') and hasattr(agent.cri_optimizer, 'param_groups'):
                    enhanced_train_metrics['critic_learning_rate'] = agent.cri_optimizer.param_groups[0]['lr']

                # 尝试获取梯度范数信息
                if 'gradient_norms' in episode_stats:
                    enhanced_train_metrics['gradient_norm'] = episode_stats['gradient_norms']

                # 掩码算法的特殊指标
                if 'beta_losses' in episode_stats:
                    enhanced_train_metrics['beta_loss'] = episode_stats['beta_losses']
                if 'rep_losses' in episode_stats:
                    enhanced_train_metrics['representation_loss'] = episode_stats['rep_losses']

            except Exception as e:
                print("Warning: Could not collect additional training metrics: {}".format(e))

            monitor.log_episode_metrics(enhanced_train_metrics)

            # 记录价值函数指标
            monitor.log_value_function_metrics(episode, agent)

        train_episode_stats_log = OrderedDict({
            "episode": [episode],
            **{f"train_{k}": ["{:04f}".format(v)] for k, v in episode_stats.items()},
        })
        episode_stats_log.update(train_episode_stats_log)
        infos.update(train_infos)

        table = print_table(train_episode_stats_log)
        print(table)
        ###################################################
        """
        if episode % cfg.save_freq == 0:
            save_checkpoint(episode, agent, exp_path, if_best=False)
        """

        ######################val#########################
        print("Validate Episode: [{}/{}]".format(episode, cfg.num_episodes))
        # val_stats, val_fig_list = validate(val_environment, agent, if_visualize=False)
        # plot_figures(val_fig_list, os.path.join(visualize_path, "val_episode_{:04d}.pdf".format(episode)))
        val_stats, val_infos = validate(val_envs, agent)

        # 兼容不同Agent类型的指标键名差异
        # MaskSAC使用ASR_env0，标准SAC使用ARR%_env0
        episode_stats = val_stats["episode_stats"]
        if "ASR_env0" in episode_stats:
            metric = np.mean([episode_stats["ASR_env0"]])  # MaskSAC的ASR指标
        elif "ARR%_env0" in episode_stats:
            metric = np.mean([episode_stats["ARR%_env0"]])  # 标准SAC的ARR%指标
        elif "ARR_env0" in episode_stats:
            metric = np.mean([episode_stats["ARR_env0"]])   # 备选ARR指标
        else:
            # 如果都没有，使用可用的第一个数值指标
            assert(False)
            available_metrics = [k for k in episode_stats.keys() if k.endswith("_env0")]
            if available_metrics:
                metric = np.mean([episode_stats[available_metrics[0]]])
            else:
                metric = 0.0  # 默认值
        if metric > max_metrics:
            max_metrics = metric
            save_checkpoint(episode, agent, exp_path, if_best=True)

        # 检查早停条件
        if monitor and monitor.should_stop_early(min_episodes=50):
            print("Early stopping triggered at episode {}".format(episode))
            break

        episode_stats = val_stats["episode_stats"]

        for k, v in episode_stats.items():
            writer.add_scalar("val/episode_{}".format(k), v, episode)

        # 记录验证指标到监控器
        if monitor:
            enhanced_val_metrics = {}
            enhanced_val_metrics.update({"val_{}".format(k): v for k, v in episode_stats.items()})

            # 添加validation_asr指标（使用主要的性能指标）
            if "ASR_env0" in episode_stats:
                enhanced_val_metrics['validation_asr'] = episode_stats["ASR_env0"]
            elif "ARR%_env0" in episode_stats:
                enhanced_val_metrics['validation_asr'] = episode_stats["ARR%_env0"]
            elif "ARR_env0" in episode_stats:
                enhanced_val_metrics['validation_asr'] = episode_stats["ARR_env0"]

            # 添加validation_reward指标
            if "SR_env0" in episode_stats:
                enhanced_val_metrics['validation_reward'] = episode_stats["SR_env0"]

            monitor.log_episode_metrics(enhanced_val_metrics)

        val_episode_log_stats = OrderedDict({
            "episode": [episode],
            **{f"val_{k}": ["{:04f}".format(v)] for k, v in episode_stats.items()},
        })
        episode_stats_log.update(val_episode_log_stats)
        infos.update(val_infos)

        table = print_table(val_episode_log_stats)
        print(table)
        ###################################################

        # ######################test########################
        # print("Test Episode: [{}/{}]".format(episode, cfg.num_episodes))
        # # test_stats, test_fig_list = validate(test_environment, agent, if_visualize=True)
        # # plot_figures(test_fig_list, os.path.join(visualize_path, "test_episode_{:04d}.pdf".format(episode)))
        # test_stats = validate(test_envs, agent, if_visualize=False)
        #
        # episode_stats = test_stats["episode_stats"]
        #
        # for k, v in episode_stats.items():
        #     writer.add_scalar("test/episode_{}".format(k), v, episode)
        #
        # test_episode_log_stats = OrderedDict({
        #     "episode": [episode],
        #     **{f"test_{k}": ["{:04f}".format(v)] for k, v in episode_stats.items()},
        # })
        #
        # table = print_table(test_episode_log_stats)
        # print(table)
        # ###################################################
        def to_json_serializable(obj):
            """
            递归地将 numpy.ndarray 和 torch.Tensor 转成 list，
            其他常见 Python 原生类型保持不变。
            """
            # numpy array → list
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # torch Tensor → list
            try:
                import torch
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().numpy().tolist()
            except ImportError:
                pass
            # dict → 递归处理
            if isinstance(obj, dict):
                return {k: to_json_serializable(v) for k, v in obj.items()}
            # list/tuple → 递归处理
            if isinstance(obj, (list, tuple)):
                return [to_json_serializable(v) for v in obj]
            # 其他（int, float, str, bool, None）直接返回
            return obj

        with pathmgr.open(os.path.join(exp_path, "train_log.txt"), "a") as op:
            clean_episode = to_json_serializable(episode_stats_log)
            op.write(json.dumps(clean_episode, ensure_ascii=False) + "\n")

        # 写入详细信息
        with pathmgr.open(os.path.join(exp_path, "train_infos.txt"), "a") as op:
            clean_infos = to_json_serializable(infos)
            op.write(json.dumps(clean_infos, ensure_ascii=False) + "\n")

    # max_episode = load_checkpoint(agent, os.path.join(exp_path, "best.pth"))
    # print("Test Max Episode: [{}/{}]".format(max_episode, cfg.num_episodes))
    # test_stats = validate(test_envs, agent)
    #
    # episode_stats = test_stats["episode_stats"]
    #
    # test_log_stats = OrderedDict({
    #     "episode": [max_episode],
    #     **{f"{k}": ["{:04f}".format(v)] for k, v in episode_stats.items()},
    # })
    #
    # table = print_table(test_log_stats)
    # print(table)
    #
    # with pathmgr.open(os.path.join(exp_path, "test_log.txt"), "a") as op:
    #     op.write(json.dumps(test_log_stats) + "\n")
    #
    # # plot metrics
    # plot_metrics(exp_path)

def train_one_episode(environment, buffer, agent, horizon_len):

    infos = dict()

    stats = {
        "episode_stats": {},
        "horizon_stats": {},
    }

    # reset environment
    environment.reset()

    while True:
        buffer_items = agent.explore_env(environment, horizon_len)
        buffer.update(buffer_items)

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        # if done is True in dones, find the min row index
        positive_indices = torch.nonzero(buffer_items[-2] > 0)
        if positive_indices.numel() == 0:
            min_row_index = horizon_len - 1
        else:
            min_row_index = torch.min(positive_indices[:, 0]).item()

        for k, v in logging_tuple.items():
            stats["horizon_stats"].setdefault("{}".format(k), []).append(v)

        if min_row_index < horizon_len - 1: # done is True in dones
            break

    # update episode stats
    for k, v in stats["horizon_stats"].items():
        stats["episode_stats"][k] = np.mean(v)

    return stats, infos

def validate(environment, agent):
    stats = {
        "episode_stats": {},
    }

    logging_tuple, infos = agent.validate_net(environment)

    # update episode stats
    for k, v in logging_tuple.items():
        stats["episode_stats"][k] = v

    return stats, infos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练脚本')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--monitor', action='store_true', help='启用训练监控')
    parser.add_argument('--root', help='项目根目录')
    parser.add_argument('--workdir', help='工作目录')
    parser.add_argument('--tag', help='实验标签')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='配置选项')
    
    args = parser.parse_args()
    
    
    try:
        main(args)
        
        send_email("yinxiaofei1986@126.com", "WUnUmJ4FxmDz5TEe", "yinxiaofei1986@126.com", 
    f'csp noopt train done !!!', 'success!', attachment=None)   
    except Exception as e:
        send_email("yinxiaofei1986@126.com", "WUnUmJ4FxmDz5TEe", "yinxiaofei1986@126.com", 
    f'csp noopt train error !!!', 'error!', attachment=None)   
        pass
