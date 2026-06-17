# -*- coding: utf-8 -*-
# base parameters (do not modify)
import torch
from tools.tools_utils import check_gpu_available

# GPU检测配置
is_gpu_available = check_gpu_available()

root = None
workdir = "workdir"
tag = "mask_sac"
# GPU环境配置切换
if is_gpu_available:
    # GPU环境，原始配置
    num_stocks = 1000
    batch_size = 32
    buffer_size = 4096
    horizon_len = 64
    embed_dim = 128
    decoder_embed_dim = 128
    repeat_times = 128
    days = 10
    if_use_rep = True
    if_use_beta = True
else:
    # CPU优化配置，减少计算量
    num_stocks = 5
    batch_size = 8
    buffer_size = 512
    horizon_len = 16
    embed_dim = 64
    decoder_embed_dim = 64
    repeat_times = 16
    days = 5
    if_use_rep = False
    if_use_beta = False
num_envs = 1
num_features = 67 # 72+3
temporal_dim = 3 # weekday, day, month
if_norm = True
if_norm_temporal = False
train_start_date = "2021-01-01"
train_end_date = "2024-06-30"
val_start_date = "2024-07-01"
val_end_date = "2024-12-31"
test_start_date = "2025-01-01"
test_end_date = "2025-03-31"
if_use_per = False
save_freq = 20
action_wrapper_method = "reweight"
T = 0.01
n_steps_per_episode = 1024

# train parameters (adjust mainly)
# num_episodes = 200
num_episodes = 5
depth = 1 # 1 transformer
decoder_depth = 1
lr = 5e-5 # act_lr, cri_lr
act_lr = 5e-5
cri_lr = 5e-5
rep_lr = 5e-5
beta_lr = 5e-5
rep_loss_weight = 0.01
beta_loss_weight = 0.01
seed = 20258888

# size
feature_size = (days, num_features)
patch_size = (days, num_features)

# CPU环境简化transition结构  
if is_gpu_available:
    transition = ["state", "action", "mask", "ids_restore", "reward", "done", "next_state"]
    transition_shape = dict(
        state = dict(shape = (num_envs, num_stocks, days, num_features), type = "float32"),
        action = dict(shape = (num_envs, num_stocks + 1), type = "float32"),
        mask = dict(shape = (num_envs, num_stocks), type = "int32"),
        ids_restore = dict(shape = (num_envs, num_stocks), type = "int64"),
        reward = dict(shape = (num_envs, ), type = "float32"),
        done = dict(shape = (num_envs, ), type = "float32"),
        next_state  = dict(shape = (num_envs, num_stocks, days, num_features), type = "float32")
    )
else:
    transition = ["state", "action", "reward", "done", "next_state"]
    transition_shape = dict(
        state = dict(shape = (num_envs, num_stocks, days, num_features), type = "float32"),
        action = dict(shape = (num_envs, num_stocks + 1), type = "float32"),
        reward = dict(shape = (num_envs, ), type = "float32"),
        done = dict(shape = (num_envs, ), type = "float32"),
        next_state  = dict(shape = (num_envs, num_stocks, days, num_features), type = "float32")
    )

dataset = dict(
    type = "PortfolioManagementDataset",
    root = root,
    #临时改成
    data_path= "datasets/csp_data_cut/features",
    # data_path= "datasets/csp_data/features",
    stocks_path = "datasets/csp_data_cut/stocks_cut.txt",
    aux_stocks_path = "datasets/csp_data_cut/aux_stocks_files_cleaned",
    # 72
    # features_name=[
    #     "vol_price_bias_6", "vol_up_momentum_pct", "price_relative_pos_20d", "vol_mom_9_26",
    #     "price_change_rate_6", "neg_corr_open_vol_10", "cov_rank_high_vol_5", "price_delta_indicator_9",
    #     "amt_1m_3m", "cov_ret_amt", "cov_volume_1m", "mean_amt", "mean_ret_amt", "ret_amt", "turn_1m_3m",
    #     "turn_mean_1m", "daily_ret_1m", "daily_ret_3m", "daily_ret_6m", "mom_3m_1m", "mom_3m", "mom_1m",
    #     "high60_close", "min_low_idx_20", "close_rank_5d", "loss_sum_5d_ratio", "gain_sum_5d_ratio",
    #     "up_days_5d_ratio", "low20_close", "high20_close", "close20_close", "sum_turnover_120d",
    #     "avg_volume_30d_ratio", "maxmin_time_distance_20d", "std_turnover_30d", "std_daily_ret_30d",
    #     "price_range_1m", "std_volume_30d", "std_volume_5d", "volume_ratio", "turnover_rate",
    #     "turnover_rate_30d", "vol_active_5d", "vol_active_30d", "macd", "macd_dea", "macd_diff",
    #     "kdj_k", "kdj_d", "kdj_j", "rsi_6", "rsi_12", "rsi_24", "clo_5d_60d", "clo_maxmin_3m",
    #     "clo_ratemean_1m", "corr_clo_turn", "diff", "dist_high", "high_low_mean", "long_short_ema",
    #     "max_ret_1m", "ret_max_1m", "ret_std_1m", "skew_ret_1m", "turn_cov_1m", "open", "high",
    #     "low", "close", "vol", "amount"
    # ],
    # 72-8
    features_name=[
        "vol_price_bias_6", "vol_up_momentum_pct", "price_relative_pos_20d", "vol_mom_9_26",
        "price_change_rate_6", "neg_corr_open_vol_10", "cov_rank_high_vol_5",
        "amt_1m_3m", "cov_ret_amt", "cov_volume_1m", "mean_amt", "mean_ret_amt", "ret_amt", "turn_1m_3m",
        "turn_mean_1m",
        "high60_close", "min_low_idx_20", "close_rank_5d", "loss_sum_5d_ratio", "gain_sum_5d_ratio",
        "up_days_5d_ratio", "low20_close", "high20_close",  "sum_turnover_120d",
        "avg_volume_30d_ratio", "maxmin_time_distance_20d", "std_turnover_30d", "std_daily_ret_30d",
        "price_range_1m", "std_volume_30d", "std_volume_5d", "volume_ratio", "turnover_rate",
        "turnover_rate_30d", "vol_active_5d", "vol_active_30d", "macd", "macd_dea", "macd_diff",
        "kdj_k", "kdj_d", "kdj_j", "rsi_6", "rsi_12", "rsi_24", "clo_5d_60d", "clo_maxmin_3m",
        "clo_ratemean_1m", "corr_clo_turn", "diff", "dist_high", "high_low_mean", "long_short_ema",
        "max_ret_1m", "ret_max_1m", "ret_std_1m", "skew_ret_1m", "turn_cov_1m", "open", "high",
        "low", "close", "vol", "amount"
    ],
    temporals_name = [
        "weekday",
        "day",
        "month",
    ],
    labels_name = [
        'ret1',
        'mov1',
    ]
)

environment = dict(
    type = "EnvironmentASR",
    dataset = None,
    mode = "train",
    if_norm = if_norm,
    if_norm_temporal = if_norm_temporal,
    scaler = None,
    days = days,
    start_date = None,
    end_date = None,
    initial_amount = 1e5,
    transaction_cost_pct = 1e-3,
)

rep_net = dict(
    type = "MaskTimeState",
    embed_type = "TimesEmbed",
    feature_size = feature_size,
    patch_size = patch_size,
    t_patch_size = 1,
    num_stocks = num_stocks,
    pred_num_stocks = num_stocks,
    in_chans = 1,
    input_dim=num_features,
    temporal_dim=temporal_dim,
    embed_dim = embed_dim,
    depth = depth,
    num_heads = 4,
    decoder_embed_dim = decoder_embed_dim,
    decoder_depth = decoder_depth,
    decoder_num_heads = 8,
    mlp_ratio = 4.0,
    norm_pix_loss = False,
    cls_embed = True,
    sep_pos_embed = True,
    trunc_init = False,
    no_qkv_bias = False,
    mask_ratio_min = 0.6,
    mask_ratio_max = 0.8,
    mask_ratio_mu = 0.7,
    mask_ratio_std = 0.1,
)

act_net = dict(
        type = "ActorMaskSAC",
        embed_dim = decoder_embed_dim,
        depth = depth,
        cls_embed = True,
    )

cri_net = dict(
        type = "CriticMaskSAC",
        embed_dim = decoder_embed_dim,
        depth = depth,
        cls_embed = True,
    )

criterion = dict(type='MSELoss', reduction="none")
scheduler = dict(type='MultiStepLRScheduler',
                 multi_steps=[600 * n_steps_per_episode, 1000 * n_steps_per_episode, 1400 * n_steps_per_episode],
                 t_initial = num_episodes * n_steps_per_episode,
                 decay_t = 500 * n_steps_per_episode,
                 gamma = 0.1,
                 t_mul = 1.,
                 lr_min = 0.,
                 decay_rate = 1.,
                 warmup_t = 300 * n_steps_per_episode,
                 warmup_lr_init = 1e-8,
                 warmup_prefix = False,
                 cycle_limit = 0,
                 t_in_epochs = False,
                 noise_range_t = None,
                 noise_pct = 0.67,
                 noise_std = 1.0,
                 noise_seed = 42,
                 initialize = True)
optimizer = dict(type='AdamW',
                 params = None,
                 lr=lr)

# Agent配置自动切换
if is_gpu_available:
    agent = dict(
        type = "AgentMaskSAC",
        act_lr=act_lr,
        cri_lr=cri_lr,
        rep_lr=rep_lr,
        beta_lr=beta_lr,
        rep_net = rep_net,
        act_net = act_net,
        cri_net = cri_net,
        criterion = criterion,
        optimizer = optimizer,
        scheduler = scheduler,
        if_use_per=if_use_per,
        if_use_rep=if_use_rep,
        if_use_beta=if_use_beta,
        rep_loss_weight=rep_loss_weight,
        beta_loss_weight=beta_loss_weight,
        num_envs = num_envs,
        transition_shape = transition_shape,
        max_step = 1e4,
        gamma = 0.99,
        reward_scale = 2**0,
        repeat_times = repeat_times,
        batch_size = batch_size,
        clip_grad_norm = 3.0,
        soft_update_tau = 5e-3,
        state_value_tau = 0,
        device = None,
        action_wrapper_method = action_wrapper_method,
        T = T,
    )
else:
    agent = dict(
        type = "AgentSAC",
        act_lr=act_lr,
        cri_lr=cri_lr,
        act_net = dict(
            type = "ActorSAC",
            feature_size = feature_size,
            num_stocks = num_stocks,
            input_dim = num_features,
            temporal_dim = temporal_dim,
            embed_dim = decoder_embed_dim,
            depth = depth,
            cls_embed = True,
        ),
        cri_net = dict(
            type = "CriticSAC", 
            feature_size = feature_size,
            num_stocks = num_stocks,
            input_dim = num_features,
            temporal_dim = temporal_dim,
            embed_dim = decoder_embed_dim,
            depth = depth,
            cls_embed = True,
        ),
        criterion = criterion,
        optimizer = optimizer,
        scheduler = scheduler,
        if_use_per=if_use_per,
        num_envs = num_envs,
        transition_shape = transition_shape,
        max_step = 1e4,
        gamma = 0.99,
        reward_scale = 2**0,
        repeat_times = repeat_times,
        batch_size = batch_size,
        clip_grad_norm = 3.0,
        soft_update_tau = 5e-3,
        state_value_tau = 0,
        device = None,
        action_wrapper_method = action_wrapper_method,
        T = T,
    )