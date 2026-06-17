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

from pm.registry import AGENT
from pm.registry import NET
from pm.registry import CRITERION
from pm.registry import OPTIMIZER
from pm.registry import SCHEDULER
from pm.utils import ReplayBuffer, build_storage, get_optim_param, get_action_wrapper, forward_action_wrapper, get_action_logprob_wrapper
from pm.metrics.metric_new import ARR, VOL, DD, MDD, SR, CR, SOR, IR, ASR

@AGENT.register_module()
class AgentMaskSAC():
    def __init__(self,
                 act_lr: float = None,
                 cri_lr: float = None,
                 rep_lr: float = None,
                 beta_lr: float = None,
                 rep_net: dict = None,
                 act_net: dict = None,
                 cri_net: dict = None,
                 criterion: dict = None,
                 optimizer: dict = None,
                 scheduler: dict = None,
                 if_use_per: bool = False,
                 if_use_beta: bool = True,
                 if_use_rep: bool = True,
                 rep_loss_weight: float = 1.0,
                 beta_loss_weight: float = 0.01,
                 num_envs: int = 1,
                 max_step: int = 1e4,
                 transition_shape: dict = None,
                 gamma: float = 0.99,
                 reward_scale: int = 2**0,
                 repeat_times: float = 1.0,
                 batch_size: int = 512,
                 clip_grad_norm: float = 3.0,
                 soft_update_tau: float = .0,
                 state_value_tau: float = 5e-3,
                 device: torch.device = torch.device("cpu"),
                 action_wrapper_method: str = "reweight",#默认是reweight
                 T: float = 1.0,#默认是1
                 ):

        self.act_lr = act_lr
        self.cri_lr = cri_lr
        self.rep_lr = rep_lr
        self.beta_lr = beta_lr

        self.num_envs = num_envs
        self.device = torch.device("cpu") if not device else device
        self.max_step = max_step

        self.if_use_per = if_use_per
        self.if_use_beta = if_use_beta
        self.if_use_rep = if_use_rep

        self.rep_loss_weight = rep_loss_weight
        self.beta_loss_weight = beta_loss_weight

        self.transition_shape = transition_shape

        self.gamma = gamma
        self.reward_scale = reward_scale
        self.repeat_times = repeat_times
        self.batch_size = batch_size

        self.clip_grad_norm = clip_grad_norm
        self.soft_update_tau = soft_update_tau
        self.state_value_tau = state_value_tau

        self.last_state = None

        # 用于监控的指标存储
        self.last_q_values = None
        self.last_td_error = None
        self.last_policy_entropy = None

        self.rep = NET.build(rep_net).to(self.device)
        self.act = self.act_target = NET.build(act_net).to(self.device)
        self.cri = self.cri_target = NET.build(cri_net).to(self.device) if cri_net else self.act
        self.cri_target = deepcopy(self.cri)

        # build optimizer
        rep_optimizer = deepcopy(optimizer)
        rep_optimizer.update(dict(params=self.rep.parameters(), lr=self.rep_lr))
        self.rep_optimizer = OPTIMIZER.build(rep_optimizer)
        self.rep_optimizer.parameters = MethodType(get_optim_param, self.rep_optimizer)

        act_optimizer = deepcopy(optimizer)
        act_optimizer.update(dict(params=self.act.parameters(), lr=self.act_lr))
        self.act_optimizer = OPTIMIZER.build(act_optimizer)
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)

        cri_optimizer = deepcopy(optimizer)
        cri_optimizer.update(dict(params=self.cri.parameters(), lr=self.cri_lr))
        self.cri_optimizer = OPTIMIZER.build(cri_optimizer) if cri_net else self.act_optimizer
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        beta_optimizer = deepcopy(optimizer)
        beta_optimizer.update(dict(params=self.act.parameters(), lr=self.beta_lr))
        self.beta_optimizer = OPTIMIZER.build(beta_optimizer)
        self.beta_optimizer.parameters = MethodType(get_optim_param, self.beta_optimizer)

        alpha_optimizer = deepcopy(optimizer)
        self.alpha_log = torch.tensor((-1,),
                                      dtype=torch.float32,
                                      requires_grad=True,
                                      device=self.device)
        alpha_optimizer.update(dict(params=(self.alpha_log,), lr=alpha_optimizer["lr"]))
        self.alpha_optimizer = OPTIMIZER.build(alpha_optimizer)
        self.alpha_optimizer.parameters = MethodType(get_optim_param, self.alpha_optimizer)

        # build scheduler
        scheduler.update(dict(optimizer=self.rep_optimizer))
        self.rep_scheduler = SCHEDULER.build(scheduler)
        scheduler.update(dict(optimizer=self.act_optimizer))
        self.act_scheduler = SCHEDULER.build(scheduler)
        scheduler.update(dict(optimizer=self.cri_optimizer))
        self.cri_scheduler = SCHEDULER.build(scheduler)
        scheduler.update(dict(optimizer=self.beta_optimizer))
        self.beta_scheduler = SCHEDULER.build(scheduler)
        scheduler.update(dict(optimizer=self.alpha_optimizer))
        self.alpha_scheduler = SCHEDULER.build(scheduler)

        # get wrapper
        self.get_action = get_action_wrapper(self.act.get_action, method=action_wrapper_method, T = T)
        self.forward_action = forward_action_wrapper(self.act.forward, method=action_wrapper_method, T = T)
        self.get_action_logprob = get_action_logprob_wrapper(self.act.get_action_logprob, method=action_wrapper_method, T = T)

        self.if_use_per = if_use_per
        if self.if_use_per:
            criterion.update(dict(reduction = "none"))
            self.get_obj_critic = self.get_obj_critic_per
        else:
            criterion.update(dict(reduction="mean"))
            self.get_obj_critic = self.get_obj_critic_raw
        self.criterion = CRITERION.build(criterion)

        self.target_entropy = transition_shape["action"]["shape"][-1] # action dim

        self.global_step = 0

    def get_state_dict(self):
        print("get state dict")
        state_dict = {
            "alpha_log": self.alpha_log,
            "rep": self.rep.state_dict(),
            "act": self.act.state_dict(),
            "cri": self.cri.state_dict(),
            "act_target": self.act_target.state_dict(),
            "cri_target": self.cri_target.state_dict(),
            "rep_optimizer": self.rep_optimizer.state_dict(),
            "act_optimizer": self.act_optimizer.state_dict(),
            "cri_optimizer": self.cri_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "beta_optimizer": self.beta_optimizer.state_dict(),
            "rep_scheduler": self.rep_scheduler.state_dict(),
            "act_scheduler": self.act_scheduler.state_dict(),
            "cri_scheduler": self.cri_scheduler.state_dict(),
            "alpha_scheduler": self.alpha_scheduler.state_dict(),
            "beta_scheduler": self.beta_scheduler.state_dict(),
        }
        print("get state dict success")
        return state_dict

    def set_state_dict(self, state_dict):
        print("set state dict")
#        self.alpha_log = state_dict["alpha_log"]
        self.act.load_state_dict(state_dict["act"])
        self.cri.load_state_dict(state_dict["cri"])
        self.rep.load_state_dict(state_dict["rep"])
        self.act_target.load_state_dict(state_dict["act_target"])
        self.cri_target.load_state_dict(state_dict["cri_target"])
        self.rep_optimizer.load_state_dict(state_dict["rep_optimizer"])
        self.act_optimizer.load_state_dict(state_dict["act_optimizer"])
        self.cri_optimizer.load_state_dict(state_dict["cri_optimizer"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        self.beta_optimizer.load_state_dict(state_dict["beta_optimizer"])
        self.rep_scheduler.load_state_dict(state_dict["rep_scheduler"])
        self.act_scheduler.load_state_dict(state_dict["act_scheduler"])
        self.cri_scheduler.load_state_dict(state_dict["cri_scheduler"])
        self.alpha_scheduler.load_state_dict(state_dict["alpha_scheduler"])
        self.beta_scheduler.load_state_dict(state_dict["beta_scheduler"])
        print("set state dict success")

    def explore_env(self, env, horizon_len: int) -> Tuple[Tensor, ...]:

        states = build_storage((horizon_len, * self.transition_shape["state"]["shape"]),
                               self.transition_shape["state"]["type"], self.device)
        actions = build_storage((horizon_len, *self.transition_shape["action"]["shape"]),
                               self.transition_shape["action"]["type"], self.device)
        masks = build_storage((horizon_len, *self.transition_shape["mask"]["shape"]),
                                self.transition_shape["mask"]["type"], self.device)
        ids_restores = build_storage((horizon_len, *self.transition_shape["ids_restore"]["shape"]),
                                self.transition_shape["ids_restore"]["type"], self.device)
        rewards = build_storage((horizon_len, *self.transition_shape["reward"]["shape"]),
                                self.transition_shape["reward"]["type"], self.device)
        dones = build_storage((horizon_len, *self.transition_shape["done"]["shape"]),
                                self.transition_shape["done"]["type"], self.device)
        next_states = build_storage((horizon_len, *self.transition_shape["next_state"]["shape"]),
                              self.transition_shape["next_state"]["type"], self.device)

        state = self.last_state

        for t in range(horizon_len):
            b, e, n, d, f = state.shape
            rep_state, mask, ids_restore = self.rep.forward_state(
                rearrange(state, "b e n d f -> (b e) n d f", b=b, e=e))

            action = self.get_action(rep_state)

            states[t] = state

            ary_action = action.detach().cpu().numpy()
            #next_state, reward, done, _ = env.step(ary_action)
            next_state, reward, terminated, truncated, info = env.step(ary_action)
            done = np.logical_or(terminated, truncated)  # 合成 old style 的 done 标志

            state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)  # next state
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0)
            done = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(0)

            actions[t] = action.unsqueeze(0)
            masks[t] = mask
            ids_restores[t] = ids_restore
            rewards[t] = reward
            dones[t] = done
            next_states[t] = state

        self.last_state = state  # state.shape == (1, state_dim) for a single env.

        rewards *= self.reward_scale
        dones = dones.type(torch.float32)
        return states, actions, masks, ids_restores, rewards, dones, next_states

    def update_rep_net(self, states: Tensor, masks: Tensor, ids_restores: Tensor, lr_scheduler=None, step=None):
        rep_loss, _, _ = self.rep(states, masks, ids_restores)
        self.optimizer_update(self.rep_optimizer, rep_loss, lr_scheduler=lr_scheduler, step=step)
        return rep_loss

    def optimizer_update(self, optimizer: torch.optim, objective: Tensor, lr_scheduler=None, step=None):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()
        lr_scheduler.step_update(step)


    def optimizer_update_amp(self, optimizer: torch.optim, objective: Tensor, lr_scheduler=None,
                             step=None):  # automatic mixed precision
        """minimize the optimization objective via update the network parameters

        amp: Automatic Mixed Precision

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        # amp_scale = torch.cuda.amp.GradScaler()  # write in __init__()
        #
        # optimizer.zero_grad()
        # amp_scale.scale(objective).backward()  # loss.backward()
        # amp_scale.unscale_(optimizer)  # amp
        #
        # # from torch.nn.utils import clip_grad_norm_
        # clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        # amp_scale.step(optimizer)  # optimizer.step()
        # amp_scale.update()  # optimizer.step()
        # lr_scheduler.step_update(step)

        optimizer.zero_grad()

        if torch.cuda.is_available():
            amp_scale = torch.cuda.amp.GradScaler()
            amp_scale.scale(objective).backward()  # loss.backward()
            amp_scale.unscale_(optimizer)  # amp
            clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
            amp_scale.step(optimizer)  # optimizer.step()
            amp_scale.update()  # optimizer.step()
        else:
            objective.backward()
            clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
            optimizer.step()

        lr_scheduler.step_update(step)

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def update_net(self, buffer: ReplayBuffer) -> dict:

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0
        alphas = 0.0
        rep_losses = 0.0
        beta_losses = 0.0
        gradient_norms = 0.0

        update_times = int(self.repeat_times)

        assert update_times >= 1
        for _ in tqdm(range(update_times),bar_format="update net batch " + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"):
            '''objective of critic (loss function of critic)'''
            obj_critic, state, mask, ids_restore, rep_states = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()

            # 收集价值函数监控指标
            with torch.no_grad():
                q1, q2 = self.cri.get_q1_q2(rep_states, self.get_action(rep_states))
                self.last_q_values = torch.min(q1, q2)

            self.optimizer_update(self.cri_optimizer, obj_critic, self.cri_scheduler, self.global_step)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, log_prob = self.get_action_logprob(rep_states)  # policy gradient
            obj_alpha = (self.alpha_log * (self.target_entropy - log_prob).detach()).mean()

            # 收集策略熵
            self.last_policy_entropy = -log_prob.mean()
            self.optimizer_update(self.alpha_optimizer, obj_alpha, self.alpha_scheduler, self.global_step)

            '''objective of actor'''
            alpha = self.alpha_log.exp().detach()
            alphas += alpha.item()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            q_value_pg = self.cri_target(rep_states, action_pg).mean()
            obj_actor = (q_value_pg - log_prob * alpha).mean()
            obj_actors += obj_actor.item()
            self.optimizer_update(self.act_optimizer, -obj_actor, self.act_scheduler, self.global_step)

            # 计算梯度范数
            total_norm = 0.0
            for param in self.act.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            if total_norm > 0:
                gradient_norms += (total_norm ** (1. / 2))

            '''objective of beta (loss function of beta)'''
            if self.if_use_beta:
                beta_loss = self.get_object_beta(rep_states, mask, ids_restore)
                self.optimizer_update(self.beta_optimizer, beta_loss, lr_scheduler=self.beta_scheduler, step=self.global_step)
                beta_losses += beta_loss.item()

            '''objective of rep net (loss function of reconstruction)'''
            if self.if_use_rep:
                rep_loss = self.update_rep_net(state, mask, ids_restore, lr_scheduler=self.rep_scheduler, step=self.global_step)
                rep_losses += rep_loss.item()

            self.global_step += 1

        act_lr = self.act_optimizer.param_groups[0]["lr"]
        cri_lr = self.cri_optimizer.param_groups[0]["lr"]
        alpha_lr = self.alpha_optimizer.param_groups[0]["lr"]

        stats = {
            "obj_critics":obj_critics / update_times,
            "obj_actors":obj_actors / update_times,
            "alphas":alphas / update_times,
            "act_lr":act_lr,
            "cri_lr":cri_lr,
            "alpha_lr":alpha_lr,
            "rep_losses": rep_losses / update_times,
            "beta_losses": beta_losses / update_times,
            "gradient_norms": gradient_norms / update_times,
        }

        return stats

    def get_q_values(self):
        """返回当前Q值用于监控"""
        return self.last_q_values

    def validate_net(self, environment):
        state, _ = environment.reset()

        num_envs = environment.num_envs

        infos = {
            "portfolio_rets": [[] for _ in range(num_envs)],
            "portfolio_values": [[] for _ in range(num_envs)],
            "date": [[] for _ in range(num_envs)],
            "portfolios":[],
        }

        while True:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            b, e, n, d, f = state.shape
            state = rearrange(state, "b e n d f -> (b e) n d f", b=b, e=e)
            rep_state, _, _ = self.rep.forward_state(state)

            action = self.forward_action(x=rep_state)

            ary_action = action.detach().cpu().numpy()

            infos["portfolios"].append(ary_action[0])

            state, reward, terminated, truncated, info = environment.step(ary_action)
            done = np.logical_or(terminated, truncated)# next_state
            #print("INFO keys:", info.keys())  # 调试用

            # 如果是常规 step，就会包含 'portfolio_ret' 这些键
            if 'portfolio_ret' in info:
                for i in range(num_envs):
                    infos["portfolio_rets"][i].append(info['portfolio_ret'][i])
                    infos["portfolio_values"][i].append(info['portfolio_value'][i])
                    infos["date"][i].append(info['date'][i])
            # 如果是最后一步的 info，只有 'final_info'
            elif 'final_info' in info:
                # final_info 通常是个 list of dict，和 portfolio_ret 一一对应
                for i in range(num_envs):
                    fin = info['final_info'][i]
                    # 有些环境里 final_info 里字段名和常规 info 略有不同，依据实际键名拿值
                    # 下面示例假设它也有 'portfolio_ret','portfolio_value','date'
                    infos["portfolio_rets"][i].append(fin.get('portfolio_ret', None))
                    infos["portfolio_values"][i].append(fin.get('portfolio_value', None))
                    infos["date"][i].append(fin.get('date', None))

            if np.sum(done) > 0:  # if any done
                break

        metrics = {}

        for i in range(num_envs):
            rets = infos["portfolio_rets"][i]  # list of floats
            dates = infos["date"][i]
            date_index = pd.to_datetime(dates)
            rets_ary = pd.Series(rets, index=date_index)

            arr = ARR(rets_ary)
            # calculate volatility
            vol = VOL(rets_ary)
            # calculate downside deviation
            dd = DD(rets_ary)
            # calculate maximum drawdown
            mdd = MDD(rets_ary)
            # calculate sharpe ratio
            sr = SR(rets_ary)
            # calculate calmar ratio
            cr = CR(rets_ary)
            # calculate sortino ratio
            sor = SOR(rets_ary)
            # ASR
            asr = ASR(rets_ary)
            # IR
            ir = IR(rets_ary)

            metrics.update({
                "ARR_env{}".format(i): arr,
                "ASR_env{}".format(i): asr,
                "SR_env{}".format(i): sr,
                "CR_env{}".format(i): cr,
                "MDD_env{}".format(i): mdd,
                "VOL_env{}".format(i): vol,
                "DD_env{}".format(i): dd,
                "SOR_env{}".format(i): sor,
                "IR_env{}".format(i): ir,
            })

        # 保存每日portfolio数据到CSV
        self._save_portfolio_data(infos)

        return metrics, infos

    def _save_portfolio_data(self, infos):
        """保存每日portfolio数据到CSV"""
        import os
        import glob

        # 查找当前正在运行的实验目录
        # 工作目录无论是项目根目录还是 experiments/，pattern "experiments/exp_*" 都能正确定位
        exp_dirs = glob.glob("experiments/exp_*")
        print("[portfolio] glob found exp_dirs: {}".format(exp_dirs))
        if exp_dirs:
            # 使用最新修改的实验目录
            exp_dir = max(exp_dirs, key=os.path.getmtime)
            logs_dir = os.path.join(exp_dir, "logs")
        else:
            # 如果没有找到exp_*目录，使用默认目录
            logs_dir = os.path.join("experiments", "default", "logs")

        os.makedirs(logs_dir, exist_ok=True)

        os.makedirs(logs_dir, exist_ok=True)

        # 构建数据
        portfolio_data = []

        num_envs = len(infos["portfolio_rets"])
        for env_idx in range(num_envs):
            rets = infos["portfolio_rets"][env_idx]
            values = infos["portfolio_values"][env_idx]
            dates = infos["date"][env_idx]

            # 获取权重数据
            weights_list = infos.get("portfolios", [])

            for step, (ret, value, date) in enumerate(zip(rets, values, dates)):
                # 获取对应的权重数据
                weights = weights_list[step] if step < len(weights_list) else None
                cash_ratio = weights[0] if weights is not None else None

                record = {
                    'env': env_idx,
                    'step': step,
                    'date': date,
                    'portfolio_return': ret,
                    'portfolio_value': value,
                    'cash_ratio': cash_ratio
                }

                # 添加股票权重
                if weights is not None:
                    for i, weight in enumerate(weights[1:], 1):
                        record[f'stock_{i}_weight'] = weight

                portfolio_data.append(record)

        # 保存到CSV
        if portfolio_data:
            df = pd.DataFrame(portfolio_data)
            csv_path = os.path.join(logs_dir, "portfolio_daily_data.csv")
            df.to_csv(csv_path, index=False)
            print(f"每日portfolio数据已保存到: {csv_path}")

    def get_object_beta(self, rep_states: Tensor, mask: Tensor, ids_restore: Tensor) -> Tensor:
        mask_bool = torch.concat(
            [torch.zeros((rep_states.shape[0], 1), dtype=torch.bool, device=self.device), mask], dim=1).bool()

        weight = self.forward_action(rep_states)

        beta_loss = (weight * mask_bool).sum(dim=1).mean()
        return beta_loss

    def get_obj_critic_raw(self, buffer, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            states, actions, masks, ids_restores, rewards, dones, next_states = buffer.sample(batch_size)  # next_ss: next states

            if states.device != self.device:
                states = states.to(self.device)
                actions = actions.to(self.device)
                masks = masks.to(self.device)
                ids_restores = ids_restores.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                next_states = next_states.to(self.device)

            rep_next_states, _, _ = self.rep.forward_state(next_states,
                                                     mask=masks,
                                                     ids_restore=ids_restores)
            next_as, next_logprobs = self.get_action_logprob(rep_next_states)

            next_qs = self.cri_target.get_q_min(rep_next_states, next_as)

            alpha = self.alpha_log.exp().detach()
            q_labels = rewards + (1.0 - dones) * self.gamma * (next_qs - next_logprobs * alpha)

            rep_states, _, _ = self.rep.forward_state(states,
                                                mask=masks,
                                                ids_restore=ids_restores)

        q1, q2 = self.cri.get_q1_q2(rep_states, actions)

        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)  # twin critics

        # 收集TD误差用于监控
        with torch.no_grad():
            td_error1 = self.criterion(q1, q_labels)
            td_error2 = self.criterion(q2, q_labels)
            self.last_td_error = (td_error1 + td_error2) / 2.0

        return obj_critic, states, masks, ids_restores, rep_states

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            states, actions, masks, ids_restores, rewards, dones, next_states, is_weights, is_indices = buffer.sample_for_per(batch_size)

            if states.device != self.device:
                states = states.to(self.device)
                actions = actions.to(self.device)
                masks = masks.to(self.device)
                ids_restores = ids_restores.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                next_states = next_states.to(self.device)
                is_weights = is_weights.to(self.device)
                is_indices = is_indices.to(self.device)

            rep_next_states, _, _ = self.rep.forward_state(next_states,
                                                     mask=masks,
                                                     ids_restore=ids_restores)
            next_as, next_logprobs = self.get_action_logprob(rep_next_states)

            next_qs = self.cri_target.get_q_min(rep_next_states,
                                                next_as)

            alpha = self.alpha_log.exp().detach()
            q_labels = rewards + (1 - dones) * self.gamma * (next_qs - next_logprobs * alpha)

            rep_states, _, _ = self.rep.forward_state(states,
                                                mask=masks,
                                                ids_restore=ids_restores)

        q1, q2 = self.cri.get_q1_q2(rep_states,
                                    actions)
        td_errors = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, states, masks, ids_restores, rep_states