import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR

import numpy as np


class PPO:
    """PPO算法(截断方式)"""
    def __init__(self,
                 policy_net,
                 value_net,
                 actor_lr,
                 critic_lr,
                 lr_decay,
                 total_steps,
                 optimizer,
                 adam_eps,
                 lmbda,
                 agent_epochs,
                 batch_size,
                 eps,
                 entropy_coef,
                 gamma,
                 adv_norm,
                 amp,
                 device,
                 valueNet_path,
                 policyNet_path,
                 val_update,
                 ):
        # 设置GPU设备
        self.device = device
        # 初始化策略网络和价值网络
        self.actor = policy_net.to(self.device)
        self.critic = value_net.to(self.device)
        # 初始化网络优化函数
        if optimizer == 'AdamW':
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),
                                                     lr=actor_lr, eps=adam_eps)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(),
                                                      lr=critic_lr, eps=adam_eps)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                    lr=actor_lr, eps=adam_eps)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                     lr=critic_lr, eps=adam_eps)
        # 初始化学习率衰减策略
        self.lr_decay = lr_decay
        if self.lr_decay:
            self.actor_scheduler = LinearLR(optimizer=self.actor_optimizer,
                                            start_factor=1,
                                            end_factor=0,
                                            total_iters=total_steps)
            self.critic_scheduler = LinearLR(optimizer=self.critic_optimizer,
                                             start_factor=1,
                                             end_factor=0,
                                             total_iters=total_steps)
        # 初始化用于混合精度训练的梯度放大器
        self.amp = amp
        if self.amp:
            self.actor_scaler = torch.cuda.amp.GradScaler()
            self.critic_scaler = torch.cuda.amp.GradScaler()
        # 设置PPO算法相关参数
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.entropy_coef = entropy_coef  # 策略熵系数
        self.adv_norm = adv_norm  # 是否使用优势估计标准化
        # 设置训练相关参数
        self.agent_epochs = agent_epochs  # 一条序列的数据用来训练的轮数
        self.batch_size = batch_size  # 每次梯度下降的步数
        # 设置模型保存路径
        self.valueNet_path = valueNet_path
        self.policyNet_path = policyNet_path
        # 若启用验证后更新模式，需要再创建一套临时网络
        if val_update:
            self.temp_actor = policy_net.to(self.device)
            self.temp_critic = value_net.to(self.device)
            self.temp_actor.load_state_dict(self.actor.state_dict())
            self.temp_critic.load_state_dict(self.critic.state_dict())

    def get_lr(self):
        """获取当前学习率"""
        actor_lr = self.actor_scheduler.get_last_lr()
        critic_lr = self.critic_scheduler.get_last_lr()
        return actor_lr, critic_lr

    def real_to_temp(self):
        """将实际网络参数赋值给临时网络参数"""
        self.temp_actor.load_state_dict(self.actor.state_dict())
        self.temp_critic.load_state_dict(self.critic.state_dict())

    def temp_to_real(self):
        """将临时网络参数赋值回实际网络参数"""
        self.actor.load_state_dict(self.temp_actor.state_dict())
        self.critic.load_state_dict(self.temp_critic.state_dict())

    def compute_advantage(self, td_delta):
        """广义优势估计GAE"""
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        if self.adv_norm:  # 使用优势估计标准化
            data = np.array(advantage_list)
            mean = np.mean(data)
            std = np.std(data)
            z_score = np.nan_to_num((data - mean) / std)
            return torch.tensor(z_score, dtype=torch.float)
        else:
            return torch.tensor(np.array(advantage_list), dtype=torch.float)

    def take_action(self, state):
        """根据策略网络采样动作，一般用于训练"""
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)  # 增加一个batch维度
        if self.amp:  # 若采用混合精度训练
            with torch.cuda.amp.autocast():
                probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def take_certain_action(self, state):
        """根据策略网络确定动作，一般用于验证和测试"""
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)  # 增加一个batch维度
        if self.amp:  # 若采用混合精度训练
            with torch.cuda.amp.autocast():
                probs = self.actor(state)
        action = torch.argmax(probs)
        return action.item()

    def update(self, transition_dict):
        """根据存入字典列表的数据更新策略网络和价值网络"""
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)  # 时序差分
        advantage = self.compute_advantage(td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()  # 取出对应动作的概率值

        for _ in range(self.agent_epochs):
            if self.batch_size:
                # 若采用minibatch更新方式，则将数据切分为数个batch
                batch_step = np.arange(0, len(states), self.batch_size)
                indices = np.arange(len(states), dtype=np.int64)
                np.random.shuffle(indices)
                batches = [indices[i:i+self.batch_size] for i in batch_step]
                # 随机梯度下降
                for batch in batches:
                    states_batch = states[batch]
                    actions_batch = actions[batch]
                    old_log_probs_batch = old_log_probs[batch]
                    advantage_batch = advantage[batch]
                    td_target_batch = td_target[batch]
                    if self.amp:  # 若采用混合精度训练
                        with torch.cuda.amp.autocast():
                            probs = self.actor(states_batch)
                            entropy = -(torch.log(probs) * probs).sum(-1).mean()  # 计算策略熵
                            log_probs = torch.log(probs.gather(1, actions_batch))  # 取出对应动作的概率值
                            ratio = torch.exp(log_probs - old_log_probs_batch)  # 新旧动作概率的比值
                            surr1 = ratio * advantage_batch
                            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage_batch  # 截断
                            actor_loss = torch.mean(-torch.min(surr1, surr2)) - entropy * self.entropy_coef  # PPO损失函数
                            critic_loss = torch.mean(F.mse_loss(self.critic(states_batch), td_target_batch.detach()))

                        self.actor_optimizer.zero_grad()
                        self.actor_scaler.scale(actor_loss).backward()  # 梯度缩放
                        self.actor_scaler.unscale_(self.actor_optimizer)  # 梯度反向缩放
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # 梯度裁剪
                        self.actor_scaler.step(self.actor_optimizer)
                        self.actor_scaler.update()

                        self.critic_optimizer.zero_grad()
                        self.critic_scaler.scale(critic_loss).backward()
                        self.critic_scaler.unscale_(self.critic_optimizer)
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                        self.critic_scaler.step(self.critic_optimizer)
                        self.critic_scaler.update()
                    else:
                        probs = self.actor(states_batch)
                        entropy = -(torch.log(probs) * probs).sum(-1).mean()  # 计算策略熵
                        log_probs = torch.log(probs.gather(1, actions_batch))  # 取出对应动作的概率值
                        ratio = torch.exp(log_probs - old_log_probs_batch)  # 新旧动作概率的比值
                        surr1 = ratio * advantage_batch
                        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage_batch  # 截断
                        actor_loss = torch.mean(-torch.min(surr1, surr2)) - entropy * self.entropy_coef  # PPO损失函数
                        critic_loss = torch.mean(F.mse_loss(self.critic(states_batch), td_target_batch.detach()))

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                        self.actor_optimizer.step()

                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                        self.critic_optimizer.step()
            else:
                # 若不采用minibatch更新方式，则整个序列作为一个batch一起更新
                if self.amp:  # 若采用混合精度训练
                    with torch.cuda.amp.autocast():
                        probs = self.actor(states)
                        entropy = -(torch.log(probs) * probs).sum(-1).mean()  # 计算策略熵
                        log_probs = torch.log(probs.gather(1, actions))  # 取出对应动作的概率值
                        ratio = torch.exp(log_probs - old_log_probs)  # 新旧动作概率的比值
                        surr1 = ratio * advantage
                        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
                        actor_loss = torch.mean(-torch.min(surr1, surr2)) - entropy * self.entropy_coef  # PPO损失函数
                        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

                    self.actor_optimizer.zero_grad()
                    self.actor_scaler.scale(actor_loss).backward()
                    self.actor_scaler.unscale_(self.actor_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.actor_scaler.step(self.actor_optimizer)
                    self.actor_scaler.update()

                    self.critic_optimizer.zero_grad()
                    self.critic_scaler.scale(critic_loss).backward()
                    self.critic_scaler.unscale_(self.critic_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_scaler.step(self.critic_optimizer)
                    self.critic_scaler.update()
                else:
                    probs = self.actor(states)
                    entropy = -(torch.log(probs) * probs).sum(-1).mean()  # 计算策略熵
                    log_probs = torch.log(probs.gather(1, actions))  # 取出对应动作的概率值
                    ratio = torch.exp(log_probs - old_log_probs)  # 新旧动作概率的比值
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
                    actor_loss = torch.mean(-torch.min(surr1, surr2)) - entropy * self.entropy_coef  # PPO损失函数
                    critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_optimizer.step()
            # 若采用学习率衰减，则更新学习率
            if self.lr_decay:
                self.actor_scheduler.step()
                self.critic_scheduler.step()

    def save_model(self):
        """保存训练好的策略网络模型和价值网络模型"""
        torch.save(self.actor.state_dict(), self.policyNet_path)
        torch.save(self.critic.state_dict(), self.valueNet_path)

    def train(self):
        """将网络模型设置为训练模式（在训练前使用）"""
        self.actor.train()
        self.critic.train()

    def eval(self):
        """将网络模型设置为验证模式（在验证或测试时使用）"""
        self.actor.eval()
        self.critic.eval()


class PPOPredict:
    """PPO算法(截断方式)预测专用"""
    def __init__(self,
                 policy_net,
                 amp,
                 device,
                 ):
        self.device = device
        self.actor = policy_net.to(self.device)
        self.amp = amp

    def take_action(self, state):
        """根据策略网络采样动作，一般用于训练"""
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)  # 增加一个batch维度
        if self.amp:  # 若采用混合精度训练
            with torch.cuda.amp.autocast():
                probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def take_certain_action(self, state):
        """根据策略网络确定动作，一般用于验证和测试"""
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)  # 增加一个batch维度
        if self.amp:  # 若采用混合精度训练
            with torch.cuda.amp.autocast():
                probs = self.actor(state)
        action = torch.argmax(probs)
        return action.item()

    def eval(self):
        """将网络模型设置为验证模式（在验证或测试时使用）"""
        self.actor.eval()
