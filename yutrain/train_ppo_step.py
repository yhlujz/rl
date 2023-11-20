import torch
from tqdm import tqdm
import logging

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    ThresholdIntensityd,
    NormalizeIntensityd,
    Orientationd,
)
from monai.data import CacheDataset, DataLoader


def train_ppo_step(train_files,
                   val_files,
                   agent,
                   env,
                   state_channel,
                   state_size,
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
                   train_spot_type,
                   val_spot_type,
                   device,
                   ):
    """训练"""

    # 定义图像前处理规则
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
                env = env(images[0],
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
                                       'covers': [],
                                       'steps': [],
                                       'actions': [],
                                       'next_states': [],
                                       'next_covers': [],
                                       'next_steps': [],
                                       'rewards': [],
                                       'dones': []}
                    state, cover, step = env.reset(spot_type=train_spot_type)  # 每个序列训练前先进行初始化操作
                    done = False
                    while not done:
                        if train_certain:  # 采用确定性训练策略
                            action = agent.take_certain_action(state, cover, step)
                        else:
                            action = agent.take_action(state, cover, step)
                        next_state, next_cover, next_step, reward, done = env.step(action)
                        transition_dict['states'].append(state)
                        transition_dict['covers'].append(cover)
                        transition_dict['steps'].append(step)
                        transition_dict['actions'].append(action)
                        transition_dict['next_states'].append(next_state)
                        transition_dict['next_covers'].append(next_cover)
                        transition_dict['next_steps'].append(next_step)
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                        state = next_state
                        cover = next_cover
                        step = next_step
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
                env = env(images[0],
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
                state, cover, step = env.reset(spot_type=val_spot_type)
                done = False
                while not done:
                    if val_certain:  # 采用确定性验证策略
                        action = agent.take_certain_action(state, cover, step)
                    else:
                        action = agent.take_action(state, cover, step)
                    next_state, next_cover, next_step, reward, done = env.step(action)
                    state = next_state
                    cover = next_cover
                    step = next_step
                    val_return += reward
                    val_length += 1
                val_dice += env.dice
            val_return /= n_val  # 计算验证集平均回报
            val_length /= n_val  # 计算验证集平均序列长度
            val_dice /= n_val  # 计算验证集平均dice
            # 保存模型
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
