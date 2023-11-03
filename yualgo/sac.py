import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR

import numpy as np


class SAC:
    """SAC算法"""
    def __init__(self,
                 device,
                 policy_net,
                 va_net,
                 actor_lr,
                 critic_lr,
                 alpha_lr,
                 target_entropy,
                 gamma,
                 tau,
                 adam_eps,
                 total_steps,
                 policyNet_path,
                 ):
        # 设置GPU设备
        self.device = device
        # 策略网络
        self.actor = policy_net.to(self.device)
        # 第一个Q网络
        self.critic_1 = va_net.to(self.device)
        # 第一个目标Q网络
        self.target_critic_1 = va_net.to(self.device)
        # 第二个Q网络
        self.critic_2 = va_net.to(self.device)
        # 第二个目标Q网络
        self.target_critic_2 = va_net.to(self.device)
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        # 优化器
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),
                                                 lr=actor_lr, eps=adam_eps)
        self.critic_1_optimizer = torch.optim.AdamW(self.critic_1.parameters(),
                                                    lr=critic_lr, eps=adam_eps)
        self.critic_2_optimizer = torch.optim.AdamW(self.critic_2.parameters(),
                                                    lr=critic_lr, eps=adam_eps)
        # 初始化学习率衰减策略
        self.actor_scheduler = LinearLR(optimizer=self.actor_optimizer,
                                        start_factor=1,
                                        end_factor=0,
                                        total_iters=total_steps)
        self.critic_1_scheduler = LinearLR(optimizer=self.critic_1_optimizer,
                                           start_factor=1,
                                           end_factor=0,
                                           total_iters=total_steps)
        self.critic_2_scheduler = LinearLR(optimizer=self.critic_2_optimizer,
                                           start_factor=1,
                                           end_factor=0,
                                           total_iters=total_steps)
        # 初始化用于混合精度训练的梯度放大器
        self.actor_scaler = torch.cuda.amp.GradScaler()
        self.critic_1_scaler = torch.cuda.amp.GradScaler()
        self.critic_2_scaler = torch.cuda.amp.GradScaler()
        # 设置SAC算法相关参数
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float).to(self.device)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.AdamW([self.log_alpha],
                                                     lr=alpha_lr, eps=adam_eps)
        self.log_alpha_scheduler = LinearLR(optimizer=self.log_alpha_optimizer,
                                            start_factor=1,
                                            end_factor=0,
                                            total_iters=total_steps)
        self.log_alpha_scaler = torch.cuda.amp.GradScaler()
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma  # 奖励衰减参数
        self.tau = tau  # 平滑更新参数
        # 设置模型保存路径
        self.policyNet_path = policyNet_path

    def step_lr(self):
        """学习率衰减一步"""
        self.actor_scheduler.step()
        self.critic_1_scheduler.step()
        self.critic_2_scheduler.step()
        self.log_alpha_scheduler.step()

    def get_lr(self):
        """获取当前学习率"""
        actor_lr = self.actor_scheduler.get_last_lr()
        critic_1_lr = self.critic_1_scheduler.get_last_lr()
        critic_2_lr = self.critic_2_scheduler.get_last_lr()
        alpha_lr = self.log_alpha_scheduler.get_last_lr()
        return actor_lr, critic_1_lr, critic_2_lr, alpha_lr

    def take_action(self, state, cover, step):
        """根据策略网络采样动作，一般用于训练"""
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)  # 增加一个batch维度
        cover = torch.tensor(np.array([cover]), dtype=torch.float).view(-1, 1).to(self.device)
        step = torch.tensor(np.array([step]), dtype=torch.float).view(-1, 1).to(self.device)
        with torch.cuda.amp.autocast():
            probs = self.actor(state, cover, step)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def calc_target(self, rewards, next_states, next_covers, next_steps, dones):
        """计算目标Q值,直接用策略网络的输出概率进行期望计算"""
        with torch.cuda.amp.autocast():
            next_probs = self.actor(next_states, next_covers, next_steps)
            next_log_probs = torch.log(next_probs + 1e-8)
            entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
            q1_value = self.target_critic_1(next_states, next_covers, next_steps)
            q2_value = self.target_critic_2(next_states, next_covers, next_steps)
            min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                                   dim=1,
                                   keepdim=True)
            next_value = min_qvalue + self.log_alpha.exp() * entropy
            td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        """平滑更新q值网络"""
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        """根据存入字典列表的数据更新策略网络和价值网络"""
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        covers = torch.tensor(transition_dict['covers']).view(-1, 1).to(self.device)
        steps = torch.tensor(transition_dict['steps']).view(-1, 1).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        next_covers = torch.tensor(transition_dict['next_covers'],
                                   dtype=torch.float).view(-1, 1).to(self.device)
        next_steps = torch.tensor(transition_dict['next_steps'],
                                  dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        with torch.cuda.amp.autocast():
            td_target = self.calc_target(rewards, next_states, next_covers, next_steps, dones)
            critic_1_q_values = self.critic_1(states, covers, steps).gather(1, actions)
            critic_1_loss = torch.mean(
                F.mse_loss(critic_1_q_values, td_target.detach()))
            critic_2_q_values = self.critic_2(states, covers, steps).gather(1, actions)
            critic_2_loss = torch.mean(
                F.mse_loss(critic_2_q_values, td_target.detach()))

        self.critic_1_optimizer.zero_grad()
        self.critic_1_scaler.scale(critic_1_loss).backward()
        self.critic_1_scaler.unscale_(self.critic_1_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 0.5)
        self.critic_1_scaler.step(self.critic_1_optimizer)
        self.critic_1_scaler.update()

        self.critic_2_optimizer.zero_grad()
        self.critic_2_scaler.scale(critic_2_loss).backward()
        self.critic_2_scaler.unscale_(self.critic_2_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 0.5)
        self.critic_2_scaler.step(self.critic_2_optimizer)
        self.critic_2_scaler.update()

        # 更新策略网络
        with torch.cuda.amp.autocast():
            probs = self.actor(states, covers, steps)
            log_probs = torch.log(probs + 1e-8)
            # 直接根据概率计算熵
            entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
            q1_value = self.critic_1(states, covers, steps)
            q2_value = self.critic_2(states, covers, steps)
            min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                                   dim=1,
                                   keepdim=True)  # 直接根据概率计算期望
            actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)

        self.actor_optimizer.zero_grad()
        self.actor_scaler.scale(actor_loss).backward()
        self.actor_scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_scaler.step(self.actor_optimizer)
        self.actor_scaler.update()

        # 更新alpha值
        with torch.cuda.amp.autocast():
            alpha_loss = torch.mean(
                (entropy - self.target_entropy).detach() * self.log_alpha.exp())

        self.log_alpha_optimizer.zero_grad()
        self.log_alpha_scaler.scale(alpha_loss).backward()
        self.log_alpha_scaler.unscale_(self.log_alpha_optimizer)
        self.log_alpha_scaler.step(self.log_alpha_optimizer)
        self.log_alpha_scaler.update()

        # 平滑更新两个q值网络
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def save_model(self):
        """保存训练好的策略网络模型和价值网络模型"""
        torch.save(self.actor.state_dict(), self.policyNet_path)

    def train(self):
        """将网络模型设置为训练模式（在训练前使用）"""
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()

    def eval(self):
        """将网络模型设置为验证模式（在验证或测试时使用）"""
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
