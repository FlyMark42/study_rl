# -*- coding: utf-8 -*-
# ============================================================================
#  文件: pm/agent/sac/mask_sac.py  （带详细中文注释版）★最核心的文件★
#  作用: EarnMore 的智能体 AgentMaskSAC —— 把"表征网络 + Actor + Critic + 优化器 +
#        训练逻辑"组装成完整的 Soft Actor-Critic 算法。对应论文算法1 和公式(5)(6)(7)(8)。
#
#  本文件最该看懂的 3 个方法:
#    1) explore_env()        —— 与环境交互、采集数据(论文算法1第一个 for)
#    2) update_net()         —— 更新 5 个组件(论文算法1第二个 for) ★最重要★
#    3) get_obj_critic_raw() —— 计算 TD 误差(贝尔曼方程的代码落地, 论文公式5)
#
#  五个被更新的组件(对照学习清单):
#    Critic(Q网络)  → Alpha(温度)  → Actor(策略)  → Beta(掩码惩罚)  → Rep(MAE表征)
# ============================================================================
import torch
from typing import Tuple
from copy import deepcopy
from torch import Tensor
from types import MethodType
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from datetime import datetime
import pandas as pd
from einops import rearrange

from pm.registry import AGENT, NET, CRITERION, OPTIMIZER, SCHEDULER
from pm.utils import (ReplayBuffer, build_storage, get_optim_param,
                      get_action_wrapper, forward_action_wrapper, get_action_logprob_wrapper)
from pm.metrics.metric_new import ARR, VOL, DD, MDD, SR, CR, SOR, IR, ASR


@AGENT.register_module()   # 注册名 "AgentMaskSAC"(配置里 agent.type 用它)
class AgentMaskSAC():
    def __init__(self,
                 act_lr=None, cri_lr=None, rep_lr=None, beta_lr=None,   # 4 个学习率
                 rep_net=None, act_net=None, cri_net=None,             # 3 个网络的"配方"
                 criterion=None, optimizer=None, scheduler=None,        # 损失/优化器/调度器配方
                 if_use_per=False,        # 是否用优先经验回放(默认否)
                 if_use_beta=True,        # 是否用掩码惩罚 beta_loss(论文§4.2监督损失)
                 if_use_rep=True,         # 是否更新表征网络(论文公式8重构损失)
                 rep_loss_weight=1.0, beta_loss_weight=0.01,
                 num_envs=1, max_step=1e4, transition_shape=None,
                 gamma=0.99,              # ★ 折扣因子 γ (学习清单2.2)
                 reward_scale=2**0,       # 奖励缩放
                 repeat_times=1.0,        # 每次 update_net 重复更新多少个 batch
                 batch_size=512,
                 clip_grad_norm=3.0,      # 梯度裁剪(防梯度爆炸)
                 soft_update_tau=.0,      # ★ 目标网络软更新系数 τ (学习清单8.2)
                 state_value_tau=5e-3,
                 device=torch.device("cpu"),
                 action_wrapper_method="reweight",   # 动作包装方式(论文§4.3重加权)
                 T=1.0,                                # 重加权温度参数
                 ):
        # ---- 保存超参 ----
        self.act_lr, self.cri_lr, self.rep_lr, self.beta_lr = act_lr, cri_lr, rep_lr, beta_lr
        self.num_envs = num_envs
        self.device = torch.device("cpu") if not device else device
        self.max_step = max_step
        self.if_use_per, self.if_use_beta, self.if_use_rep = if_use_per, if_use_beta, if_use_rep
        self.rep_loss_weight, self.beta_loss_weight = rep_loss_weight, beta_loss_weight
        self.transition_shape = transition_shape
        self.gamma, self.reward_scale = gamma, reward_scale
        self.repeat_times, self.batch_size = repeat_times, batch_size
        self.clip_grad_norm, self.soft_update_tau, self.state_value_tau = clip_grad_norm, soft_update_tau, state_value_tau
        self.last_state = None
        # 监控用的中间量
        self.last_q_values = None
        self.last_td_error = None
        self.last_policy_entropy = None

        # ---- 构造 3 个网络(用注册器从配方造出来) ----
        self.rep = NET.build(rep_net).to(self.device)               # 表征网络 MaskTimeState
        self.act = self.act_target = NET.build(act_net).to(self.device)   # Actor(策略)
        self.cri = self.cri_target = NET.build(cri_net).to(self.device) if cri_net else self.act  # Critic
        self.cri_target = deepcopy(self.cri)   # ★ Critic 的目标网络(算TD目标时用, 缓慢更新)

        # ---- 为每个网络各建一个优化器(注意: 表征/Actor/Critic/Beta/Alpha 分开优化, 互不干扰) ----
        # Beta 优化器和 Actor 共享参数(都是更新 act 网络), 但用不同的损失和学习率
        rep_optimizer = deepcopy(optimizer); rep_optimizer.update(dict(params=self.rep.parameters(), lr=self.rep_lr))
        self.rep_optimizer = OPTIMIZER.build(rep_optimizer); self.rep_optimizer.parameters = MethodType(get_optim_param, self.rep_optimizer)
        act_optimizer = deepcopy(optimizer); act_optimizer.update(dict(params=self.act.parameters(), lr=self.act_lr))
        self.act_optimizer = OPTIMIZER.build(act_optimizer); self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        cri_optimizer = deepcopy(optimizer); cri_optimizer.update(dict(params=self.cri.parameters(), lr=self.cri_lr))
        self.cri_optimizer = OPTIMIZER.build(cri_optimizer) if cri_net else self.act_optimizer
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)
        beta_optimizer = deepcopy(optimizer); beta_optimizer.update(dict(params=self.act.parameters(), lr=self.beta_lr))
        self.beta_optimizer = OPTIMIZER.build(beta_optimizer); self.beta_optimizer.parameters = MethodType(get_optim_param, self.beta_optimizer)

        # ---- 温度系数 α(论文公式7): 学的是 log(α), 保证 α>0 ----
        alpha_optimizer = deepcopy(optimizer)
        self.alpha_log = torch.tensor((-1,), dtype=torch.float32, requires_grad=True, device=self.device)
        alpha_optimizer.update(dict(params=(self.alpha_log,), lr=alpha_optimizer["lr"]))
        self.alpha_optimizer = OPTIMIZER.build(alpha_optimizer); self.alpha_optimizer.parameters = MethodType(get_optim_param, self.alpha_optimizer)

        # ---- 为每个优化器各建一个学习率调度器(带预热的多步衰减, 论文附录C) ----
        for opt_name, opt in [("rep", self.rep_optimizer), ("act", self.act_optimizer), ("cri", self.cri_optimizer),
                              ("beta", self.beta_optimizer), ("alpha", self.alpha_optimizer)]:
            scheduler.update(dict(optimizer=opt))
            setattr(self, f"{opt_name}_scheduler", SCHEDULER.build(scheduler))

        # ---- ★ 把 Actor 的方法用"重加权包装器"包一层(论文§4.3公式9) ★ ----
        # 之后调用 self.get_action(ρ) 时, 会自动做"温度 softmax + 掩码忽略"
        self.get_action = get_action_wrapper(self.act.get_action, method=action_wrapper_method, T=T)
        self.forward_action = forward_action_wrapper(self.act.forward, method=action_wrapper_method, T=T)
        self.get_action_logprob = get_action_logprob_wrapper(self.act.get_action_logprob, method=action_wrapper_method, T=T)

        # ---- 损失函数(MSE), 以及是否用优先经验回放的分支 ----
        self.if_use_per = if_use_per
        if self.if_use_per:
            criterion.update(dict(reduction="none"))
            self.get_obj_critic = self.get_obj_critic_per     # 优先经验回放版
        else:
            criterion.update(dict(reduction="mean"))
            self.get_obj_critic = self.get_obj_critic_raw     # 普通版(主线)
        self.criterion = CRITERION.build(criterion)

        # 目标熵 = 动作维度(SAC 自动熵调节的目标值, 论文公式7的 H̄)
        self.target_entropy = transition_shape["action"]["shape"][-1]
        self.global_step = 0

    # ------------------------------------------------------------------
    #  检查点: 保存/加载所有网络、优化器、调度器的状态
    # ------------------------------------------------------------------
    def get_state_dict(self):
        print("get state dict")
        state_dict = {
            "alpha_log": self.alpha_log,
            "rep": self.rep.state_dict(), "act": self.act.state_dict(), "cri": self.cri.state_dict(),
            "act_target": self.act_target.state_dict(), "cri_target": self.cri_target.state_dict(),
            "rep_optimizer": self.rep_optimizer.state_dict(), "act_optimizer": self.act_optimizer.state_dict(),
            "cri_optimizer": self.cri_optimizer.state_dict(), "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "beta_optimizer": self.beta_optimizer.state_dict(),
            "rep_scheduler": self.rep_scheduler.state_dict(), "act_scheduler": self.act_scheduler.state_dict(),
            "cri_scheduler": self.cri_scheduler.state_dict(), "alpha_scheduler": self.alpha_scheduler.state_dict(),
            "beta_scheduler": self.beta_scheduler.state_dict(),
        }
        print("get state dict success")
        return state_dict

    def set_state_dict(self, state_dict):
        print("set state dict")
        # 注意: alpha_log 这一行被注释掉了(加载时不恢复 α, 重新自适应)
        self.act.load_state_dict(state_dict["act"]); self.cri.load_state_dict(state_dict["cri"])
        self.rep.load_state_dict(state_dict["rep"])
        self.act_target.load_state_dict(state_dict["act_target"]); self.cri_target.load_state_dict(state_dict["cri_target"])
        self.rep_optimizer.load_state_dict(state_dict["rep_optimizer"]); self.act_optimizer.load_state_dict(state_dict["act_optimizer"])
        self.cri_optimizer.load_state_dict(state_dict["cri_optimizer"]); self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        self.beta_optimizer.load_state_dict(state_dict["beta_optimizer"])
        self.rep_scheduler.load_state_dict(state_dict["rep_scheduler"]); self.act_scheduler.load_state_dict(state_dict["act_scheduler"])
        self.cri_scheduler.load_state_dict(state_dict["cri_scheduler"]); self.alpha_scheduler.load_state_dict(state_dict["alpha_scheduler"])
        self.beta_scheduler.load_state_dict(state_dict["beta_scheduler"])
        print("set state dict success")

    # ==================================================================
    #  ★方法1: explore_env —— 与环境交互、采集数据 (论文算法1 第一个 for)★
    # ==================================================================
    def explore_env(self, env, horizon_len: int) -> Tuple[Tensor, ...]:
        """
        让智能体在环境里走 horizon_len 步, 把每步的 (状态,动作,掩码,还原索引,奖励,done,下一状态)
        收集起来, 返回给回放缓冲区。这就是 RL 的"采样"阶段。
        """
        # 预先分配存储空间(每种数据一个张量)
        states = build_storage((horizon_len, *self.transition_shape["state"]["shape"]), self.transition_shape["state"]["type"], self.device)
        actions = build_storage((horizon_len, *self.transition_shape["action"]["shape"]), self.transition_shape["action"]["type"], self.device)
        masks = build_storage((horizon_len, *self.transition_shape["mask"]["shape"]), self.transition_shape["mask"]["type"], self.device)
        ids_restores = build_storage((horizon_len, *self.transition_shape["ids_restore"]["shape"]), self.transition_shape["ids_restore"]["type"], self.device)
        rewards = build_storage((horizon_len, *self.transition_shape["reward"]["shape"]), self.transition_shape["reward"]["type"], self.device)
        dones = build_storage((horizon_len, *self.transition_shape["done"]["shape"]), self.transition_shape["done"]["type"], self.device)
        next_states = build_storage((horizon_len, *self.transition_shape["next_state"]["shape"]), self.transition_shape["next_state"]["type"], self.device)

        state = self.last_state   # 从上次结束的状态接着走

        for t in range(horizon_len):
            b, e, n, d, f = state.shape
            # ① 原始状态 → 可掩码股票表征 ρ (随机掩码, 模拟一个随机子池 CSP)
            rep_state, mask, ids_restore = self.rep.forward_state(
                rearrange(state, "b e n d f -> (b e) n d f", b=b, e=e))
            # ② 表征 → 动作(经重加权): 得到投资组合权重
            action = self.get_action(rep_state)

            states[t] = state
            ary_action = action.detach().cpu().numpy()
            # ③ 在环境里执行动作, 拿到反馈
            next_state, reward, terminated, truncated, info = env.step(ary_action)
            done = np.logical_or(terminated, truncated)   # 合成 done 标志

            # 转成张量
            state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0)
            done = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(0)

            # ④ 存储这一步的所有数据(注意: mask 和 ids_restore 也存, 更新时要复用)
            actions[t] = action.unsqueeze(0)
            masks[t] = mask
            ids_restores[t] = ids_restore
            rewards[t] = reward
            dones[t] = done
            next_states[t] = state

        self.last_state = state          # 记住最后状态, 下次接着走
        rewards *= self.reward_scale     # 奖励缩放
        dones = dones.type(torch.float32)
        return states, actions, masks, ids_restores, rewards, dones, next_states

    # ------------------------------------------------------------------
    #  更新表征网络(MAE 重构损失, 论文公式8)
    # ------------------------------------------------------------------
    def update_rep_net(self, states, masks, ids_restores, lr_scheduler=None, step=None):
        rep_loss, _, _ = self.rep(states, masks, ids_restores)   # 调 MaskTimeState.forward() 拿重构损失
        self.optimizer_update(self.rep_optimizer, rep_loss, lr_scheduler=lr_scheduler, step=step)
        return rep_loss

    # ------------------------------------------------------------------
    #  通用的"优化一步"工具: 清零梯度 → 反向传播 → 梯度裁剪 → 更新 → 调度器步进
    # ------------------------------------------------------------------
    def optimizer_update(self, optimizer, objective, lr_scheduler=None, step=None):
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()
        lr_scheduler.step_update(step)

    def optimizer_update_amp(self, optimizer, objective, lr_scheduler=None, step=None):
        """混合精度版(AMP), GPU 上加速用。逻辑同上, 多了 GradScaler。(细节, 略)"""
        optimizer.zero_grad()
        if torch.cuda.is_available():
            amp_scale = torch.cuda.amp.GradScaler()
            amp_scale.scale(objective).backward()
            amp_scale.unscale_(optimizer)
            clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
            amp_scale.step(optimizer)
            amp_scale.update()
        else:
            objective.backward()
            clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
            optimizer.step()
        lr_scheduler.step_update(step)

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """
        ★ 目标网络软更新 (学习清单8.2, 论文公式5的 θ̄) ★
        target = target*(1-τ) + current*τ  —— 让目标网络"慢半拍"地跟上主网络, 训练更稳。
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    # ==================================================================
    #  ★★★ 方法2: update_net —— 更新 5 个组件 (论文算法1 第二个 for) ★★★
    #  这是整个 EarnMore 训练的心脏。每个 batch 严格按论文顺序更新:
    #    Critic → Alpha → Actor → Beta → Rep
    # ==================================================================
    def update_net(self, buffer: ReplayBuffer) -> dict:
        obj_critics = obj_actors = alphas = rep_losses = beta_losses = gradient_norms = 0.0
        update_times = int(self.repeat_times)
        assert update_times >= 1

        for _ in tqdm(range(update_times), bar_format="update net batch " + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"):
            # ---------- ① 更新 Critic (Q网络): 最小化 TD 误差 ----------
            # get_obj_critic 内部算好了 TD 误差(论文公式5), 并顺便返回当前 batch 的表征
            obj_critic, state, mask, ids_restore, rep_states = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            # 监控: 记录当前 Q 值
            with torch.no_grad():
                q1, q2 = self.cri.get_q1_q2(rep_states, self.get_action(rep_states))
                self.last_q_values = torch.min(q1, q2)
            self.optimizer_update(self.cri_optimizer, obj_critic, self.cri_scheduler, self.global_step)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)   # 目标网络软更新

            # ---------- ② 更新 Alpha (温度系数): 自动熵调节 (论文公式7) ----------
            action_pg, log_prob = self.get_action_logprob(rep_states)   # 当前策略的动作和对数概率
            # α 的损失: 让"实际熵"靠近"目标熵"。当熵太低(探索不足)时增大 α, 反之减小
            obj_alpha = (self.alpha_log * (self.target_entropy - log_prob).detach()).mean()
            self.last_policy_entropy = -log_prob.mean()   # 监控: 策略熵
            self.optimizer_update(self.alpha_optimizer, obj_alpha, self.alpha_scheduler, self.global_step)

            # ---------- ③ 更新 Actor (策略网络): 最大化 Q - α·logπ (论文公式6) ----------
            alpha = self.alpha_log.exp().detach()   # α = exp(log α)
            alphas += alpha.item()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)   # 限制 α 范围, 防止数值爆炸
            q_value_pg = self.cri_target(rep_states, action_pg).mean()      # Critic 对当前动作的打分
            obj_actor = (q_value_pg - log_prob * alpha).mean()              # ★ 目标: Q - α·logπ(最大熵RL) ★
            obj_actors += obj_actor.item()
            # 注意这里传 -obj_actor: 优化器默认最小化, 加负号 = 最大化 obj_actor(梯度上升)
            self.optimizer_update(self.act_optimizer, -obj_actor, self.act_scheduler, self.global_step)

            # 监控: 计算 Actor 的梯度范数
            total_norm = 0.0
            for param in self.act.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            if total_norm > 0:
                gradient_norms += (total_norm ** 0.5)

            # ---------- ④ 更新 Beta (掩码惩罚): 论文§4.2 的"监督损失" ----------
            # 惩罚 Actor 给"被掩盖股票"分配权重, 让池外股票权重趋近 0
            if self.if_use_beta:
                beta_loss = self.get_object_beta(rep_states, mask, ids_restore)
                self.optimizer_update(self.beta_optimizer, beta_loss, lr_scheduler=self.beta_scheduler, step=self.global_step)
                beta_losses += beta_loss.item()

            # ---------- ⑤ 更新 Rep (MAE表征网络): 最小化重构损失 (论文公式8) ----------
            # 论文强调: 表征学习与RL"同时进行"(端到端), 不预训练
            if self.if_use_rep:
                rep_loss = self.update_rep_net(state, mask, ids_restore, lr_scheduler=self.rep_scheduler, step=self.global_step)
                rep_losses += rep_loss.item()

            self.global_step += 1

        # 收集本次更新的统计量(用于日志/监控)
        act_lr = self.act_optimizer.param_groups[0]["lr"]
        cri_lr = self.cri_optimizer.param_groups[0]["lr"]
        alpha_lr = self.alpha_optimizer.param_groups[0]["lr"]
        stats = {
            "obj_critics": obj_critics / update_times, "obj_actors": obj_actors / update_times,
            "alphas": alphas / update_times, "act_lr": act_lr, "cri_lr": cri_lr, "alpha_lr": alpha_lr,
            "rep_losses": rep_losses / update_times, "beta_losses": beta_losses / update_times,
            "gradient_norms": gradient_norms / update_times,
        }
        return stats

    def get_q_values(self):
        """返回当前 Q 值(给外部监控用)。"""
        return self.last_q_values

    # ==================================================================
    #  验证/推理: 在验证环境跑一整回合, 算各种金融指标 (论文算法2)
    # ==================================================================
    def validate_net(self, environment):
        state, _ = environment.reset()
        num_envs = environment.num_envs
        infos = {"portfolio_rets": [[] for _ in range(num_envs)], "portfolio_values": [[] for _ in range(num_envs)],
                 "date": [[] for _ in range(num_envs)], "portfolios": []}

        while True:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            b, e, n, d, f = state.shape
            state = rearrange(state, "b e n d f -> (b e) n d f", b=b, e=e)
            rep_state, _, _ = self.rep.forward_state(state)        # 状态 → 表征
            action = self.forward_action(x=rep_state)               # ★ 用确定性 forward(推理稳定)
            ary_action = action.detach().cpu().numpy()
            infos["portfolios"].append(ary_action[0])
            state, reward, terminated, truncated, info = environment.step(ary_action)
            done = np.logical_or(terminated, truncated)
            # 收集每步的收益/市值/日期(普通步 or 最后一步)
            if 'portfolio_ret' in info:
                for i in range(num_envs):
                    infos["portfolio_rets"][i].append(info['portfolio_ret'][i])
                    infos["portfolio_values"][i].append(info['portfolio_value'][i])
                    infos["date"][i].append(info['date'][i])
            elif 'final_info' in info:
                for i in range(num_envs):
                    fin = info['final_info'][i]
                    infos["portfolio_rets"][i].append(fin.get('portfolio_ret', None))
                    infos["portfolio_values"][i].append(fin.get('portfolio_value', None))
                    infos["date"][i].append(fin.get('date', None))
            if np.sum(done) > 0:
                break

        # 用收益序列算 6+ 个金融指标(论文附录A)
        metrics = {}
        for i in range(num_envs):
            rets = infos["portfolio_rets"][i]
            dates = infos["date"][i]
            rets_ary = pd.Series(rets, index=pd.to_datetime(dates))
            metrics.update({
                "ARR_env{}".format(i): ARR(rets_ary),   # 年化收益率
                "ASR_env{}".format(i): ASR(rets_ary),   # 调整夏普(本仓库主指标)
                "SR_env{}".format(i): SR(rets_ary),     # 夏普比率
                "CR_env{}".format(i): CR(rets_ary),     # 卡玛比率
                "MDD_env{}".format(i): MDD(rets_ary),   # 最大回撤
                "VOL_env{}".format(i): VOL(rets_ary),   # 波动率
                "DD_env{}".format(i): DD(rets_ary),     # 下行偏差
                "SOR_env{}".format(i): SOR(rets_ary),   # 索提诺比率
                "IR_env{}".format(i): IR(rets_ary),     # 信息比率
            })
        self._save_portfolio_data(infos)   # 把每日持仓存成 CSV(便于可视化)
        return metrics, infos

    def _save_portfolio_data(self, infos):
        """把每日 portfolio(收益/市值/各股权重)存成 CSV, 供后续可视化分析。(工程细节)"""
        import os, glob
        exp_dirs = glob.glob("experiments/exp_*")
        if exp_dirs:
            exp_dir = max(exp_dirs, key=os.path.getmtime)
            logs_dir = os.path.join(exp_dir, "logs")
        else:
            logs_dir = os.path.join("experiments", "default", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        portfolio_data = []
        num_envs = len(infos["portfolio_rets"])
        for env_idx in range(num_envs):
            rets = infos["portfolio_rets"][env_idx]
            values = infos["portfolio_values"][env_idx]
            dates = infos["date"][env_idx]
            weights_list = infos.get("portfolios", [])
            for step, (ret, value, date) in enumerate(zip(rets, values, dates)):
                weights = weights_list[step] if step < len(weights_list) else None
                cash_ratio = weights[0] if weights is not None else None
                record = {'env': env_idx, 'step': step, 'date': date, 'portfolio_return': ret,
                          'portfolio_value': value, 'cash_ratio': cash_ratio}
                if weights is not None:
                    for i, weight in enumerate(weights[1:], 1):
                        record[f'stock_{i}_weight'] = weight
                portfolio_data.append(record)
        if portfolio_data:
            df = pd.DataFrame(portfolio_data)
            csv_path = os.path.join(logs_dir, "portfolio_daily_data.csv")
            df.to_csv(csv_path, index=False)
            print(f"每日portfolio数据已保存到: {csv_path}")

    # ------------------------------------------------------------------
    #  Beta 损失: 论文§4.2 的"监督损失" —— 惩罚给被掩盖股票分配权重
    # ------------------------------------------------------------------
    def get_object_beta(self, rep_states, mask, ids_restore):
        # mask_bool: 第0位现金补0(不惩罚), 其余沿用 mask(1=被掩盖)
        mask_bool = torch.concat(
            [torch.zeros((rep_states.shape[0], 1), dtype=torch.bool, device=self.device), mask], dim=1).bool()
        weight = self.forward_action(rep_states)            # Actor 给出的权重
        # 把"分配给被掩盖股票的权重"加起来, 作为损失 → 希望它趋近 0
        beta_loss = (weight * mask_bool).sum(dim=1).mean()
        return beta_loss

    # ==================================================================
    #  ★方法3: get_obj_critic_raw —— 计算 TD 误差 (贝尔曼方程的代码落地)★
    #  对应论文公式(5)、学习清单第3部分(贝尔曼)和第6部分(TD)。务必逐行看懂!
    # ==================================================================
    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():   # 算 TD 目标时不需要梯度
            # 从回放缓冲区随机抽一批(异策略 RL 的标志, 学习清单第5、8部分)
            states, actions, masks, ids_restores, rewards, dones, next_states = buffer.sample(batch_size)
            # 设备搬运(若缓冲区在 CPU 而训练在 GPU)
            if states.device != self.device:
                states = states.to(self.device); actions = actions.to(self.device)
                masks = masks.to(self.device); ids_restores = ids_restores.to(self.device)
                rewards = rewards.to(self.device); dones = dones.to(self.device); next_states = next_states.to(self.device)

            # 下一状态 → 表征(复用存好的 mask/ids_restore, 保证编码一致)
            rep_next_states, _, _ = self.rep.forward_state(next_states, mask=masks, ids_restore=ids_restores)
            # 下一状态的动作和对数概率(用当前策略)
            next_as, next_logprobs = self.get_action_logprob(rep_next_states)
            # 下一状态的 Q 值(用目标网络, 双Q取min抑制高估)
            next_qs = self.cri_target.get_q_min(rep_next_states, next_as)
            alpha = self.alpha_log.exp().detach()
            # ★★★ TD 目标 = r + γ(1-done)(Q_target(s',a') - α·logπ(a'|s')) ★★★
            #     这一行 = 论文公式(5) = 软化的贝尔曼方程(比普通贝尔曼多减一个熵项)
            q_labels = rewards + (1.0 - dones) * self.gamma * (next_qs - next_logprobs * alpha)

            # 当前状态 → 表征
            rep_states, _, _ = self.rep.forward_state(states, mask=masks, ids_restore=ids_restores)

        # 当前 Q 值(两个 Q 都算, 训练 Critic)
        q1, q2 = self.cri.get_q1_q2(rep_states, actions)
        # ★ TD 误差 = (Q1 - 目标)² + (Q2 - 目标)²  → Critic 要最小化它 ★ (孪生 Critic)
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)

        # 监控: 记录 TD 误差
        with torch.no_grad():
            td_error1 = self.criterion(q1, q_labels)
            td_error2 = self.criterion(q2, q_labels)
            self.last_td_error = (td_error1 + td_error2) / 2.0
        return obj_critic, states, masks, ids_restores, rep_states

    def get_obj_critic_per(self, buffer, batch_size):
        """优先经验回放(PER)版的 TD 误差计算。逻辑同上, 多了重要性采样权重 is_weights。(默认不用)"""
        with torch.no_grad():
            states, actions, masks, ids_restores, rewards, dones, next_states, is_weights, is_indices = buffer.sample_for_per(batch_size)
            if states.device != self.device:
                states = states.to(self.device); actions = actions.to(self.device)
                masks = masks.to(self.device); ids_restores = ids_restores.to(self.device)
                rewards = rewards.to(self.device); dones = dones.to(self.device); next_states = next_states.to(self.device)
                is_weights = is_weights.to(self.device); is_indices = is_indices.to(self.device)
            rep_next_states, _, _ = self.rep.forward_state(next_states, mask=masks, ids_restore=ids_restores)
            next_as, next_logprobs = self.get_action_logprob(rep_next_states)
            next_qs = self.cri_target.get_q_min(rep_next_states, next_as)
            alpha = self.alpha_log.exp().detach()
            q_labels = rewards + (1 - dones) * self.gamma * (next_qs - next_logprobs * alpha)
            rep_states, _, _ = self.rep.forward_state(states, mask=masks, ids_restore=ids_restores)
        q1, q2 = self.cri.get_q1_q2(rep_states, actions)
        td_errors = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        obj_critic = (td_errors * is_weights).mean()   # 用重要性权重加权
        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())  # 回写 TD 误差(供 PER 调整优先级)
        return obj_critic, states, masks, ids_restores, rep_states

# ============================================================================
#  ★ 通关检验 ★ 如果你能对着 update_net() 说出下面这张表, 就真正掌握了 EarnMore:
#
#   更新顺序 | 组件          | 损失/目标                       | 论文公式 | 学习清单
#   --------|--------------|--------------------------------|---------|--------
#     ①     | Critic(Q网络) | min (Q - TD目标)²               | 公式(5) | 第3,6部分
#     ②     | Alpha(温度)   | 让策略熵靠近目标熵               | 公式(7) | 第10部分
#     ③     | Actor(策略)   | max (Q - α·logπ)               | 公式(6) | 第9,10部分
#     ④     | Beta(掩码惩罚) | min 给被掩盖股票的权重之和        | §4.2    | 第11部分
#     ⑤     | Rep(MAE表征)  | min 被掩盖股票的重构MSE          | 公式(8) | 第11部分
#
#  其中 ①(TD误差)是 RL 的引擎, ③(Actor)是决策的来源,
#  ②(熵)让探索充分, ④⑤是 EarnMore 相对普通 SAC 的两大创新(掩码感知 + 表征学习)。
# ============================================================================
