import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging
from datetime import datetime

from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    EnsureTyped,
)
from monai.data import CacheDataset, DataLoader

from yunet import ValueNet, PolicyNet


class PPO:
    """PPO算法(截断方式)"""
    def __init__(self,
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
        self.actor = PolicyNet().to(self.device)
        self.critic = ValueNet().to(self.device)
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
            self.temp_actor = PolicyNet().to(self.device)
            self.temp_critic = ValueNet().to(self.device)
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
        """根据策略网络采样动作，用于训练"""
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)  # 增加一个batch维度
        if self.amp:  # 若采用混合精度训练
            with torch.cuda.amp.autocast():
                probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def take_certain_action(self, state):
        """根据策略网络确定动作，用于验证和测试"""
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
        rewards = torch.tensor(transition_dict['rewards'],
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
                            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)  # 截断
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
                        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)  # 截断
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
                        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)  # 截断
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
                    surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)  # 截断
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


class CTEnv:
    """定义CT图像环境"""
    def __init__(self,
                 image,
                 mask,
                 state_size,
                 step_max,
                 step_limit_max,
                 state_mode,
                 reward_mode,
                 out_mode,
                 out_reward_mode,
                 ):
        self.image = image  # 初始化image
        self.mask = mask.int()  # 初始化mask
        self.state_size = state_size  # 初始化状态图大小
        self.step_max = step_max  # 限制最大步数
        self.step_limit_max = step_limit_max  # 限制无新标注的探索步数
        self.state_mode = state_mode  # 设置状态返回模式
        self.reward_mode = reward_mode  # 设置奖励函数模式
        self.out_mode = out_mode  # 设置边界外是否停止
        self.out_reward_mode = out_reward_mode  # 设置边界外奖励函数

        self.step_n = 0  # 初始步数
        self.step_limit_n = 0  # 初始无新标注步数

        # 根据状态图大小对原图像和标注图像进行padding
        self.pad_length = int((self.state_size[0] - 1) / 2)
        self.pad_width = int((self.state_size[1] - 1) / 2)
        self.pad_depth = int((self.state_size[2] - 1) / 2)
        pad_size = (self.pad_depth, self.pad_depth,
                    self.pad_width, self.pad_width,
                    self.pad_length, self.pad_length)
        self.image_padding = F.pad(self.image, pad_size, 'constant', 0)
        self.mask_padding = F.pad(self.mask, pad_size, 'constant', 0)

        # 创建与标注大小相同的预测图
        self.pred_padding = torch.zeros_like(self.mask_padding).int()

        # 初始化起点(注意：起点坐标为padding后的坐标)
        spots = torch.nonzero(self.mask_padding)
        i = np.random.randint(len(spots))
        self.ori_spot = spots[i]  # 从标注中随机选取一点作为起点
        self.spot = self.ori_spot.tolist()  # 初始化智能体位置坐标

    def reset(self):
        """回归初始状态（预测图只包含随机起点的状态）并返回初始状态值"""
        self.step_n = 1  # 步数置1
        self.step_limit_n = 0  # 无新标注步数置0
        self.spot = self.ori_spot.tolist()  # 初始化智能体位置坐标
        self.pred_padding = torch.zeros_like(self.mask_padding).int()  # 初始化预测图像
        if self.state_mode == 'post':
            next_state = self.spot_to_state()  # post模式先返回状态后标注
        self.pred_padding[tuple(self.spot)] = 1
        if self.state_mode == 'pre':
            next_state = self.spot_to_state()  # pre模式先标注后返回状态
        return next_state  # 返回下一个状态图

    def spot_to_state(self):
        """通过spot和给定的state图像大小计算返回相应state图像"""
        # 计算当前spot对应于padding后图像的状态图边界
        l_side = self.spot[0] - self.pad_length  # 状态图左边界
        r_side = self.spot[0] + self.pad_length + 1  # 状态图右边界
        u_side = self.spot[1] - self.pad_width  # 状态图上边界
        d_side = self.spot[1] + self.pad_width + 1  # 状态图下边界
        f_side = self.spot[2] - self.pad_depth  # 状态图前边界
        b_side = self.spot[2] + self.pad_depth + 1  # 状态图后边界
        # 分别获取原图像的状态图和预测图像的状态图并融合
        image_state = self.image_padding[l_side:r_side, u_side:d_side, f_side:b_side]
        pred_state = self.pred_padding[l_side:r_side, u_side:d_side, f_side:b_side]
        state = torch.stack((image_state, pred_state), dim=0)
        return np.array(state.cpu())  # 将状态图转换为numpy格式

    def step(self, action):
        """智能体完成一个动作，并返回下一个状态、奖励和完成情况"""
        change = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        self.spot = [x + y for x, y in zip(self.spot, change[action])]  # 将动作叠加到位置
        self.step_n += 1  # 步数累积
        done = False
        # 如果超出边界
        if self.spot_is_out():
            if self.out_mode:
                done = True
            next_state = self.spot_to_state()  # 超出边界时状态图不移动
            # 根据不同模式计算超出边界时的reward
            if self.out_reward_mode == 'step':
                reward = -self.step_n  # 超出边界时奖励为当前动作数的负值
            elif self.out_reward_mode == 'small':
                reward = -1  # 超出边界时奖励为-1
            elif self.out_reward_mode == 'large':
                reward = -100  # 超出边界时奖励为-100
        # 未超出边界时
        else:
            if self.state_mode == 'post':
                next_state = self.spot_to_state()  # post模式先返回状态后标注
            # 根据不同模式计算reward
            if self.reward_mode == 'const':
                # 计算reward
                reward = self.spot_to_const_reward()
                # 在预测图像中记录标注路径
                self.pred_padding[tuple(self.spot)] = 1
            elif self.reward_mode == 'dice_inc':
                # 计算上一个状态的dice值
                dice_old = self.compute_dice()
                # 在预测图像中记录标注路径
                self.pred_padding[tuple(self.spot)] = 1
                # 计算下个状态的dice值及增量
                dice_new = self.compute_dice()
                dice_inc = dice_new - dice_old
                reward = dice_inc * 100  # 奖励为dice值增量的100倍
            if self.state_mode == 'pre':
                next_state = self.spot_to_state()  # pre模式先标注后返回状态
        # 判断是否达到步数限制条件
        if reward <= 0:
            self.step_limit_n += 1  # 无新标注步数累积
        else:
            self.step_limit_n = 0  # 清空无新标注步数
        if self.step_limit_n >= self.step_limit_max:
            done = True
        if self.step_n >= self.step_max:
            done = True
        return next_state, reward, done

    def spot_is_out(self):
        """判断当前spot是否超出padding前的实际边界，超出边界的点赋值为边界值"""
        length = self.pred_padding.shape[0]
        width = self.pred_padding.shape[1]
        depth = self.pred_padding.shape[2]
        if self.spot[0] < self.pad_length:
            self.spot[0] = self.pad_length
            return True
        elif self.spot[0] > length - self.pad_length - 1:
            self.spot[0] = length - self.pad_length - 1
            return True
        elif self.spot[1] < self.pad_width:
            self.spot[1] = self.pad_width
            return True
        elif self.spot[1] > width - self.pad_width - 1:
            self.spot[1] = width - self.pad_width - 1
            return True
        elif self.spot[2] < self.pad_depth:
            self.spot[2] = self.pad_depth
            return True
        elif self.spot[2] > depth - self.pad_depth - 1:
            self.spot[2] = depth - self.pad_depth - 1
            return True
        else:
            return False  # 未超出边界则返回False

    def compute_dice(self):
        """返回当前预测图像和标注图像dice值"""
        return (2 * (self.pred_padding & self.mask_padding).sum() /
                (self.pred_padding.sum() + self.mask_padding.sum())).item()

    def spot_to_const_reward(self):
        """reward模式为const时，通过当前spot得到对应的reward"""
        if self.pred_padding[tuple(self.spot)] == 1:  # 重复时奖励为0
            reward = 0
        elif self.mask_padding[tuple(self.spot)] == 1:  # 不重复且涂对时奖励为1
            reward = 1
        else:
            reward = -1  # 未涂对时奖励为-1
        return reward


def train(train_files,
          val_files,
          agent,
          state_size,
          device,
          epochs,
          num_workers,
          step_max,
          step_limit_max,
          num_episodes,
          state_mode,
          reward_mode,
          out_mode,
          out_reward_mode,
          val_update,
          ):
    """训练"""

    # 定义图像前处理规则
    transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),   # 载入图像
            ScaleIntensityRanged(keys=["image"], a_min=-135, a_max=215, b_min=0.0, b_max=1.0, clip=True),  # 设置窗宽窗位
            EnsureTyped(keys=["image", "mask"])   # 转换为tensor
        ]
    )

    # 创建数据集并加载数据
    train_ds = CacheDataset(data=train_files, transform=transforms, cache_rate=1.0, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=num_workers)
    n_train = len(train_loader)
    val_ds = CacheDataset(data=val_files, transform=transforms, cache_rate=1.0, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)
    n_val = len(val_loader)

    # 设置训练轮次
    max_epochs = epochs

    # 记录最佳评价指标
    best_dice = 0

    # 记录得到最佳评价指标的轮次
    best_dice_epoch = -1

    # 开始训练
    for epoch in range(max_epochs):
        agent.train()  # 网络设置为训练模式
        epoch_return = 0  # 记录一轮训练平均回报
        epoch_length = 0  # 记录一轮训练平均序列长度
        epoch_dice = 0  # 记录一轮训练平均dice
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{max_epochs}', unit='batch') as pbar:
            for batch_data in train_loader:
                batch_return = 0  # 记录一个batch训练平均回报
                batch_length = 0  # 记录一个batch训练平均序列长度
                batch_dice = 0  # 记录一个batch训练平均dice
                images, masks = (
                    batch_data["image"].to(device),
                    batch_data["mask"].to(device),
                )
                env = CTEnv(images[0],
                            masks[0],
                            state_size,
                            step_max,
                            step_limit_max,
                            state_mode,
                            reward_mode,
                            out_mode,
                            out_reward_mode)  # 初始化环境
                for _ in range(num_episodes):
                    episode_return = 0  # 记录一个序列的总回报
                    episode_length = 0  # 记录一个序列的总长度
                    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                    state = env.reset()  # 每个序列训练前先进行初始化操作（设置随机起点）
                    done = False
                    while not done:
                        action = agent.take_action(state)
                        next_state, reward, done = env.step(action)
                        transition_dict['states'].append(state)
                        transition_dict['actions'].append(action)
                        transition_dict['next_states'].append(next_state)
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                        state = next_state
                        episode_return += reward
                        episode_length += 1
                    batch_return += episode_return
                    batch_length += episode_length
                    batch_dice += env.compute_dice()
                    agent.update(transition_dict)  # 根据训练出的一条序列进行网络模型的更新
                pbar.update(1)   # 进度条更新
                batch_return /= num_episodes
                batch_length /= num_episodes
                batch_dice /= num_episodes
                pbar.set_postfix(**{'return': batch_return, 'length': batch_length, 'dice': batch_dice})  # 设置进度条后缀
                epoch_return += batch_return
                epoch_length += batch_length
                epoch_dice += batch_dice
        epoch_return /= n_train  # 计算一个epoch的平均回报
        epoch_length /= n_train  # 计算一个epoch的平均序列长度
        epoch_dice /= n_train  # 计算一个epoch的平均dice
        actor_lr, critic_lr = agent.get_lr()

        # 记录训练集log
        logging.info(f'''
        epoch {epoch + 1}
        return: {epoch_return} length: {epoch_length} dice: {epoch_dice}
        actor_lr: {actor_lr} critic_lr: {critic_lr}''')
        print(f"epoch {epoch + 1} return: {epoch_return} length: {epoch_length} dice: {epoch_dice}")

        # 验证及保存模型
        agent.eval()  # 网络设置为验证模式
        val_return = 0  # 记录验证集平均回报
        val_length = 0  # 记录验证集平均序列长度
        val_dice = 0  # 记录验证集平均dice
        with torch.no_grad():
            for val_data in val_loader:
                images, masks = (
                    val_data["image"].to(device),
                    val_data["mask"].to(device),
                )
                env = CTEnv(images[0],
                            masks[0],
                            state_size,
                            step_max,
                            step_limit_max,
                            state_mode,
                            reward_mode,
                            out_mode,
                            out_reward_mode)  # 初始化环境
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_certain_action(state)
                    next_state, reward, done = env.step(action)
                    state = next_state
                    val_return += reward
                    val_length += 1
                val_dice += env.compute_dice()
            val_return /= n_val  # 计算验证集平均回报
            val_length /= n_val  # 计算验证集平均序列长度
            val_dice /= n_val  # 计算验证集平均dice
            # 保存模型
            if val_update:  # 如果需要验证后才更新参数
                if val_dice > best_dice:
                    best_dice = val_dice
                    best_dice_epoch = epoch + 1
                    agent.save_model()
                    agent.real_to_temp()
                else:
                    agent.temp_to_real()
            else:
                if val_dice > best_dice:
                    best_dice = val_dice
                    best_dice_epoch = epoch + 1
                    agent.save_model()

        # 记录验证集log
        logging.info(f'''
        val epoch {epoch + 1}
        dice: {val_dice} length: {val_length} return: {val_return}
        best dice {best_dice} at epoch {best_dice_epoch}''')
        print(f"val epoch {epoch + 1} dice: {val_dice} / best dice {best_dice} at epoch {best_dice_epoch}")


if __name__ == '__main__':
    """固定随机种子"""
    set_determinism(seed=0)

    """数据json路径"""
    json_path = '/workspace/data/rl/json/rl6.json'

    """训练编号"""
    id = '0'

    """设置GPU"""
    GPU_id = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}\n')

    """获取当前日期"""
    date_time = datetime.now().strftime("%m%d")

    """设置模型保存路径"""
    valueNet_path = f'/workspace/data/rl/model/ppo_value{date_time}{id}.pth'
    policyNet_path = f'/workspace/data/rl/model/ppo_policy{date_time}{id}.pth'

    """设置log保存路径"""
    log_path = f'/workspace/data/rl/log/ppo{date_time}{id}.log'

    """读取json获取所有图像文件名"""
    train_images = []
    train_masks = []
    df = pd.read_json(json_path)
    for index, row in tqdm(df.iterrows()):
        if row['dataset'] == 'train':
            train_images.append(row['image_path'])
            train_masks.append(row['mask_path'])

    """制作文件名字典（以便后面使用字典制作数据集）"""
    data_dicts = [{"image": image_name, "mask": mask_name}
                  for image_name, mask_name in zip(train_images, train_masks)]

    """划分得到训练集字典和验证集字典"""
    val_len = round(len(df) / 10)
    train_files, val_files = data_dicts[:-val_len], data_dicts[-val_len:]

    """设置日志写入规则"""
    logging.basicConfig(level=logging.INFO,
                        filename=log_path,
                        filemode='a',
                        format='%(levelname)s: %(message)s',
                        force=True)  # 写入info级别以上的日志

    """参数设置"""
    actor_lr = 1e-5  # 策略函数初始学习率
    critic_lr = 1e-4  # 价值函数初始学习率
    lr_decay = True  # 是否使用学习率衰减（线性衰减）
    optimizer = 'AdamW'  # 优化器选用，可选：Adam, AdamW
    adam_eps = 1e-5  # adam优化器限制值
    gamma = 0.98  # 价值估计倍率
    lmbda = 0.95  # 优势估计倍率
    adv_norm = True  # 是否进行优势估计标准化
    amp = True  # 是否使用混合精度训练和推断加速
    agent_epochs = 10  # 每个序列梯度下降迭代次数
    batch_size = 0  # 若使用minibatch方式进行更新，则需设置为非0值
    eps = 0.2  # ppo算法限制值
    entropy_coef = 0.01  # 策略熵系数，可设置为0
    state_size = [21, 21, 9]  # 状态图大小
    epochs = 500  # 总循环次数
    num_workers = 0  # 数据加载线程数
    step_max = 5000  # 序列最大长度
    step_limit_max = 500  # 序列无正面奖励限制长度
    num_episodes = 1  # 每条序列训练次数
    state_mode = 'pre'  # 状态模式，可选：pre(先标注再返回状态), post(返回状态后再标注)
    reward_mode = 'dice_inc'  # 奖励模式，可选：dice_inc, const
    out_mode = True  # 出边界是否停止，True则停止
    out_reward_mode = 'small'  # 出边界奖励模式，可选：small, large, step
    total_steps = epochs * len(train_files) * num_episodes * agent_epochs  # 计算梯度下降迭代总步数，后续进行学习率衰减使用
    val_update = False  # 是否经过验证集后才真正更新网络参数

    """记录参数信息"""
    logging.info(f'''
    json_path = {json_path}
    GPU_id = {GPU_id}
    actor_lr = {actor_lr}  critic_lr = {critic_lr}
    lr_decay = {lr_decay}
    optimizer = {optimizer}  adam_eps = {adam_eps}
    agent_epochs = {agent_epochs}
    batch_size = {batch_size}
    adv_norm = {adv_norm}  entropy_coef = {entropy_coef}
    amp = {amp}
    step_max = {step_max}  step_limit_max = {step_limit_max}
    num_episodes = {num_episodes}
    state_mode = {state_mode}
    reward_mode = {reward_mode}
    out_mode = {out_mode}  out_reward_mode = {out_reward_mode}
    val_update = {val_update}''')

    """初始化agent"""
    agent = PPO(actor_lr=actor_lr,
                critic_lr=critic_lr,
                lr_decay=lr_decay,
                total_steps=total_steps,
                optimizer=optimizer,
                adam_eps=adam_eps,
                lmbda=lmbda,
                agent_epochs=agent_epochs,
                batch_size=batch_size,
                eps=eps,
                entropy_coef=entropy_coef,
                gamma=gamma,
                adv_norm=adv_norm,
                amp=amp,
                device=device,
                valueNet_path=valueNet_path,
                policyNet_path=policyNet_path,
                val_update=val_update,
                )

    """训练"""
    train(train_files=train_files,
          val_files=val_files,
          agent=agent,
          state_size=state_size,
          device=device,
          epochs=epochs,
          num_workers=num_workers,
          step_max=step_max,
          step_limit_max=step_limit_max,
          num_episodes=num_episodes,
          state_mode=state_mode,
          reward_mode=reward_mode,
          out_mode=out_mode,
          out_reward_mode=out_reward_mode,
          val_update=val_update,
          )
