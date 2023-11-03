import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR

import numpy as np


class D3QN:
    """D3QN算法(Double DQN + Dueling DQN)"""
    def __init__(self,
                 device,
                 va_net,
                 learning_rate,
                 total_steps,
                 adam_eps,
                 gamma,
                 epsilon,
                 target_update,
                 VANet_path,
                 ):
        # 设置GPU设备
        self.device = device
        # 初始化q值网络
        self.q_net = va_net.to(device)
        self.target_q_net = va_net.to(device)
        self.optimizer = torch.optim.AdamW(self.q_net.parameters(),
                                           lr=learning_rate, eps=adam_eps)
        # 初始化学习率衰减策略
        self.scheduler = LinearLR(optimizer=self.optimizer,
                                  start_factor=1,
                                  end_factor=0,
                                  total_iters=total_steps)
        # 初始化用于混合精度训练的梯度放大器
        self.scaler = torch.cuda.amp.GradScaler()
        # 设置D3QN算法相关参数
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        # 设置模型保存路径
        self.VANet_path = VANet_path

    def step_lr(self):
        """学习率衰减一步"""
        self.scheduler.step()

    def get_lr(self):
        """获取当前学习率"""
        lr = self.scheduler.get_last_lr()
        return lr

    def take_action(self, state, cover, step):
        """根据q值网络确定动作，并使用贪婪方式"""
        if np.random.random() < self.epsilon:
            action = np.random.randint(6)  # 一共6个动作
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            cover = torch.tensor(np.array([cover]), dtype=torch.float).view(-1, 1).to(self.device)
            step = torch.tensor(np.array([step]), dtype=torch.float).view(-1, 1).to(self.device)
            with torch.cuda.amp.autocast():
                action = self.q_net(state, cover, step).argmax().item()
        return action

    def update(self, transition_dict):
        """根据存入字典列表的数据更新q值网络"""
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

        with torch.cuda.amp.autocast():
            q_values = self.q_net(states, covers, steps).gather(1, actions)
            max_action = self.q_net(next_states, next_covers,
                                    next_steps).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states, next_covers,
                                                  next_steps).gather(1, max_action)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
            dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        self.scaler.scale(dqn_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # 更新一定次数后同步两个q值网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

    def save_model(self):
        """保存训练好的q值网络模型"""
        torch.save(self.q_net.state_dict(), self.VANet_path)

    def train(self):
        """将网络模型设置为训练模式（在训练前使用）"""
        self.q_net.train()

    def eval(self):
        """将网络模型设置为验证模式（在验证或测试时使用）"""
        self.q_net.eval()
