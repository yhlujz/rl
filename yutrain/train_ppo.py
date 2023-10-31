import torch
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
    ThresholdIntensityd,
    NormalizeIntensityd,
    Orientationd,
)
from monai.data import CacheDataset, DataLoader

from yualgo.ppo import PPO
from yuenv.ct_env import CTEnv


class RunningMeanStd:
    """动态计算mean和std"""
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    """状态标准化"""
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    """奖励标准化"""
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


def train(train_files,
          val_files,
          agent,
          state_channel,
          state_size,
          norm_method,
          state_norm,
          reward_norm,
          gamma,
          epochs,
          num_workers,
          step_max,
          step_limit_max,
          num_episodes,
          state_mode,
          reward_mode,
          out_mode,
          out_reward_mode,
          train_certain,
          val_certain,
          val_update,
          train_spot_type,
          val_spot_type,
          device,
          ):
    """训练"""

    # 定义图像前处理规则
    if norm_method == 'min_max':
        transforms = Compose(
            [
                LoadImaged(keys=["image", "mask", "pred"]),   # 载入图像
                LoadImaged(keys=["prob"],
                           reader="NumpyReader",
                           npz_keys='probabilities'),  # 载入预测概率图
                Orientationd(keys=["prob"], axcodes="SAR"),
                ScaleIntensityRanged(keys=["image"], a_min=-135, a_max=215,
                                     b_min=0, b_max=1,
                                     clip=True),  # 归一化
                EnsureTyped(keys=["image", "mask", "pred", "prob"])  # 转换为tensor
            ]
        )
    elif norm_method == 'norm':
        transforms = Compose(
            [
                LoadImaged(keys=["image", "mask", "pred"]),   # 载入图像
                LoadImaged(keys=["prob"],
                           reader="NumpyReader",
                           npz_keys='probabilities'),  # 载入预测概率图
                Orientationd(keys=["prob"], axcodes="SAR"),
                ThresholdIntensityd(keys=["image"], threshold=72, above=True, cval=72),
                ThresholdIntensityd(keys=["image"], threshold=397, above=False, cval=397),
                NormalizeIntensityd(keys=["image"],
                                    subtrahend=226.2353057861328,
                                    divisor=62.27060317993164),
                EnsureTyped(keys=["image", "mask", "pred", "prob"])  # 转换为tensor
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

    # 初始化状态和奖励的标准化方法
    state_normalizer = Normalization(state_size)
    reward_normalizer = RewardScaling(1, gamma)

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
                images, masks, preds, probs = (
                    batch_data["image"].to(device),
                    batch_data["mask"].to(device),
                    batch_data["pred"].to(device),
                    batch_data["prob"].to(device),
                )
                env = CTEnv(images[0],
                            masks[0],
                            preds[0],
                            probs[0][1],
                            state_channel,
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
                    transition_dict = {'states': [],
                                       'actions': [],
                                       'next_states': [],
                                       'rewards': [],
                                       'dones': []}
                    state = env.reset(spot_type=train_spot_type)  # 每个序列训练前先进行初始化操作
                    reward_normalizer.reset()  # 每个序列初始化奖励标准化方法
                    done = False
                    while not done:
                        if state_norm:  # 采用状态标准化
                            state = state_normalizer(state)
                        if train_certain:  # 采用确定性训练策略
                            action = agent.take_certain_action(state)
                        else:
                            action = agent.take_action(state)
                        next_state, reward, done = env.step(action)
                        if reward_norm:  # 采用奖励标准化
                            reward = reward_normalizer(reward)
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
                    batch_dice += env.dice
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
                images, masks, preds, probs = (
                    val_data["image"].to(device),
                    val_data["mask"].to(device),
                    val_data["pred"].to(device),
                    val_data["prob"].to(device),
                )
                env = CTEnv(images[0],
                            masks[0],
                            preds[0],
                            probs[0][1],
                            state_channel,
                            state_size,
                            step_max,
                            step_limit_max,
                            state_mode,
                            reward_mode,
                            out_mode,
                            out_reward_mode)  # 初始化环境
                state = env.reset(spot_type=val_spot_type)
                done = False
                while not done:
                    if state_norm:  # 验证时采用状态标准化
                        state = state_normalizer(state, update=False)
                    if val_certain:  # 采用确定性验证策略
                        action = agent.take_certain_action(state)
                    else:
                        action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    state = next_state
                    val_return += reward
                    val_length += 1
                val_dice += env.dice
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
    json_path = '/workspace/data/rl/json/rl6_new.json'

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
    train_preds = []
    train_probs = []
    df = pd.read_json(json_path)
    for _, row in tqdm(df.iterrows()):
        if row['dataset'] == 'train':
            train_images.append(row['image_path'])
            train_masks.append(row['mask_path'])
            train_preds.append(row['pred_path'])
            train_probs.append(row['prob_path'])

    """制作文件名字典（以便后面使用字典制作数据集）"""
    data_dicts = [{"image": image_name, "mask": mask_name,
                   "pred": pred_name, "prob": prob_name}
                  for image_name, mask_name, pred_name, prob_name
                  in zip(train_images, train_masks, train_preds, train_probs)]

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
    state_norm = False  # 是否使用状态标准化
    reward_norm = False  # 是否使用奖励标准化
    amp = True  # 是否使用混合精度训练和推断加速
    agent_epochs = 10  # 每个序列梯度下降迭代次数
    batch_size = 0  # 若使用minibatch方式进行更新，则需设置为非0值
    eps = 0.2  # ppo算法限制值
    entropy_coef = 0.01  # 策略熵系数，可设置为0

    state_channel = 3  # 状态图通道数，可选2，3
    state_size = [21, 21, 9]  # 状态图大小
    norm_method = 'norm'  # 归一化方法，可选：min_max, norm
    epochs = 500  # 总循环次数
    num_workers = 0  # 数据加载线程数
    step_max = 5000  # 序列最大长度
    step_limit_max = 500  # 限制无新标注的探索步数
    num_episodes = 1  # 每条序列训练次数
    state_mode = 'pre'  # 状态模式，可选：pre(先标注再返回状态), post(返回状态后再标注)
    reward_mode = 'dice_inc_const'  # 奖励模式，可选：dice_inc, const, dice_inc_const
    out_mode = False  # 出边界是否停止，True则停止
    out_reward_mode = 'small'  # 出边界奖励模式，可选：small, large, step，0
    total_steps = epochs * len(train_files) * num_episodes * agent_epochs  # 计算梯度下降迭代总步数，后续进行学习率衰减使用
    train_certain = False  # 是否在训练时采用确定性策略，False代表采用随机采样策略
    val_certain = False  # 是否在验证时采用确定性策略，False代表采用随机采样策略
    val_update = False  # 是否经过验证集后才真正更新网络参数
    train_spot_type = 'ori_spot'  # 设置训练起点类型，可选ori_spot，prob_spot
    val_spot_type = 'prob_spot'  # 设置验证起点类型

    """网络选择"""
    net_name = 'pvnet'  # 可选：pvnet, pvnet2, resnet
    if net_name == 'pvnet':
        from yunet import ValueNet as ValueNet, PolicyNet as PolicyNet
    elif net_name == 'pvnet2':
        from yunet import ValueNet2 as ValueNet, PolicyNet2 as PolicyNet
    elif net_name == 'resnet':
        from yunet import ValueResNet as ValueNet, PolicyResNet as PolicyNet
    else:
        print('error: the net is not exist!')

    policy_net = PolicyNet(state_channel).to(device)
    value_net = ValueNet(state_channel).to(device)

    """记录参数信息"""
    logging.info(f'''
    net_name = {net_name}
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
    state_channel = {state_channel}  state_size = {state_size}
    norm_method = {norm_method}
    state_mode = {state_mode}  state_norm = {state_norm}
    reward_mode = {reward_mode}  reward_norm = {reward_norm}
    out_mode = {out_mode}  out_reward_mode = {out_reward_mode}
    train_certain = {train_certain}
    val_certain = {val_certain}  val_update = {val_update}
    train_spot_type = {train_spot_type}  val_spot_type = {val_spot_type}''')

    """初始化agent"""
    agent = PPO(policy_net=policy_net,
                value_net=value_net,
                actor_lr=actor_lr,
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
          state_channel=state_channel,
          state_size=state_size,
          norm_method=norm_method,
          state_norm=state_norm,
          reward_norm=reward_norm,
          gamma=gamma,
          epochs=epochs,
          num_workers=num_workers,
          step_max=step_max,
          step_limit_max=step_limit_max,
          num_episodes=num_episodes,
          state_mode=state_mode,
          reward_mode=reward_mode,
          out_mode=out_mode,
          out_reward_mode=out_reward_mode,
          train_certain=train_certain,
          val_certain=val_certain,
          val_update=val_update,
          train_spot_type=train_spot_type,
          val_spot_type=val_spot_type,
          device=device,
          )
