import argparse
import os
import parser
import sys
import json
from pathlib import Path
import torch
from mmengine import DictAction
from mmengine.config import Config
import numpy as np
from iopath.common.file_io import g_pathmgr as pathmgr
from collections import OrderedDict
import gym
from copy import deepcopy
import random
import pandas as pd
from gym.vector import SyncVectorEnv
import copy

# 添加项目根路径
ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from pm.registry import ENVIRONMENT, AGENT, DATASET
from pm.utils import update_data_root, load_checkpoint
def validate(environment, agent):
    stats = {
        "episode_stats": {},
    }

    logging_tuple, infos = agent.validate_net(environment)

    # update episode stats
    for k, v in logging_tuple.items():
        stats["episode_stats"][k] = v

    return stats, infos

def make_env(env_id, env_params):
    def thunk():
        env = gym.make(env_id, **env_params)
        return env
    return thunk

def test_main(args):
    # 加载配置
    cfg = Config.fromfile(args.config)
    SEED = cfg.seed
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    else:
        torch.manual_seed(SEED)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cfg_options is None:
        args.cfg_options = {}
    if args.root:
        root_path = Path(args.root)
        if not root_path.is_absolute():
            args.root = str((Path(ROOT) / root_path).resolve())
        args.cfg_options['root'] = args.root
    if args.workdir:
        args.cfg_options['workdir'] = args.workdir
    if args.tag:
        args.cfg_options['tag'] = args.tag
    cfg.merge_from_dict(args.cfg_options)
    print(cfg)

    # 数据路径更新
    update_data_root(cfg, root=args.root)

    # 构建数据集
    dataset = DATASET.build(cfg.dataset)

    # 构建训练环境
    cfg.environment.update(dict(
        mode="train",
        if_norm=True,
        dataset=dataset,
        start_date=cfg.train_start_date,
        end_date=cfg.train_end_date
    ))
    train_env = ENVIRONMENT.build(cfg.environment)
    trained_scalers = train_env.scaler  # <— 这里

    # 构造 test 环境，传入 scaler
    cfg.environment.update(dict(
        mode="test",
        if_norm=True,
        dataset=dataset,
        scaler=trained_scalers,  # <— 一定要带上这一行
        start_date=cfg.test_start_date,
        end_date=cfg.test_end_date
    ))
    test_env = ENVIRONMENT.build(cfg.environment)
    test_envs = gym.vector.SyncVectorEnv(
        [make_env("PortfolioManagement-v0",
                  env_params=dict(
                      env=deepcopy(test_env),
                      transition_shape=cfg.transition_shape,
                      seed=SEED + i
                  ))
         for i in range(len(test_env.aux_stocks))]
    )
    test_envs.reset(seed=SEED)

    # 加载模型
    # exp_path = os.path.join(cfg.root, cfg.workdir, cfg.tag)
    exp_path = os.path.join(cfg.root, cfg.workdir)
    best_model = os.path.join(exp_path, 'best.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.agent.update(dict(device=device))
    agent = AGENT.build(cfg.agent)
    agent.device = device
    if os.path.exists(best_model):
        load_checkpoint(agent, best_model)
        print(f'Loaded model from {best_model}')
    else:
        print('Best model not found, exiting.')
        return

    # 运行测试
    stats, infos = validate(test_envs, agent)
    # print(infos.keys())

    ep_stats = stats['episode_stats']
    print('====== Test Results ======')
    for k,v in ep_stats.items():
        print(f'{k}: {v:.4f}')

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
        if isinstance(obj, (list, tuple)):
            return [to_json_serializable(v) for v in obj]
        return obj

    # 写入日志
    log_path = os.path.join(exp_path, 'test_log.txt')
    with pathmgr.open(log_path, 'a') as f:
        entry = OrderedDict({'episode': ['test'], **{k: [f'{v:.4f}'] for k,v in ep_stats.items()}})
        f.write(json.dumps(to_json_serializable(entry), ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # 使用硬编码参数替代 argparse，之前的代码
    # class Args:
    #     config = "../configs/mask_sac_portfolio_management.py"
    #     cfg_options = dict()
    #     root = "/root/autodl-tmp"
    #     workdir = "EM_original"
    #     tag = "2021_original"
    # args = Args()
    parser = argparse.ArgumentParser(description='训练脚本')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--root', help='项目根目录')
    parser.add_argument('--workdir', help='工作目录')
    parser.add_argument('--tag', help='实验标签')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='配置选项')
    args = parser.parse_args()
    test_main(args)

