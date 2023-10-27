import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging
from datetime import datetime
import collections
import random

from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    ThresholdIntensityd,
    NormalizeIntensityd,
    Orientationd,
)
from monai.data import CacheDataset, DataLoader

from yualg.d3qn import D3QN
from yuenv.ct_env import CTEnv


class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据，数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, np.array(reward), np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


def train(train_files,
          val_files,
          agent,
          buffer_size,
          minimal_size,
          batch_size,
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

    # 初始化回放池
    replay_buffer = ReplayBuffer(buffer_size)

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
                    state = env.reset(spot_type=train_spot_type)  # 每个序列训练前先进行初始化操作
                    done = False
                    while not done:
                        action = agent.take_action(state)
                        next_state, reward, done = env.step(action)
                        replay_buffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward
                        episode_length += 1
                        # 当buffer数据的数量超过一定值后，才进行Q网络训练
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'dones': b_d
                            }
                            agent.update(transition_dict)  # 每个迭代更新一次
                    agent.step_lr()
                    batch_return += episode_return
                    batch_length += episode_length
                    batch_dice += env.dice
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
        lr = agent.get_lr()

        # 记录训练集log
        logging.info(f'''
        epoch {epoch + 1}
        return: {epoch_return} length: {epoch_length} dice: {epoch_dice}
        lr: {lr}''')
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
    """网络选择"""
    net_name = 'vanet'
    if net_name == 'vanet':
        from yunet import VANet
    else:
        print('error: the net is not exist!')

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
    VANet_path = f'/workspace/data/rl/model/d3qn_va{date_time}{id}.pth'

    """设置log保存路径"""
    log_path = f'/workspace/data/rl/log/d3qn{date_time}{id}.log'

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
    learning_rate = 1e-4  # 初始学习率
    adam_eps = 1e-5  # adam优化器限制值
    gamma = 0.98  # 价值估计倍率
    epsilon = 0.01  # 贪婪因子
    target_update = 50  # 多少次迭代后同步两个网络参数

    buffer_size = 125000  # 回放池大小
    minimal_size = 25000  # 回放池最小大小
    batch_size = 1600  # 每次迭代的batch大小
    state_size = [21, 21, 9]  # 状态图大小
    epochs = 100  # 总循环次数
    num_workers = 0  # 数据加载线程数
    step_max = 5000  # 序列最大长度
    step_limit_max = 500  # 序列重复标注最大长度
    num_episodes = 1  # 每条序列训练次数
    state_mode = 'pre'  # 状态模式，可选：pre(先标注再返回状态), post(返回状态后再标注)
    reward_mode = 'dice_inc_const'  # 奖励模式，可选：dice_inc, const, dice_inc_const
    out_mode = False  # 出边界是否停止，True则停止
    out_reward_mode = 'small'  # 出边界奖励模式，可选：small, large, step，0
    total_steps = epochs * len(train_files) * num_episodes  # 计算梯度下降迭代总步数，后续进行学习率衰减使用
    train_spot_type = 'ori_spot'  # 设置训练起点类型，可选ori_spot，prob_spot
    val_spot_type = 'prob_spot'  # 设置验证起点类型

    """记录参数信息"""
    logging.info(f'''
    net_name = {net_name}
    json_path = {json_path}
    GPU_id = {GPU_id}
    lr = {learning_rate}
    adam_eps = {adam_eps}
    epsilon = {epsilon}
    target_update = {target_update}
    buffer_size = {buffer_size}  minimal_size = {minimal_size}
    batch_size = {batch_size}
    step_max = {step_max}  step_limit_max = {step_limit_max}
    num_episodes = {num_episodes}
    state_size = {state_size}
    state_mode = {state_mode}
    reward_mode = {reward_mode}
    out_mode = {out_mode}  out_reward_mode = {out_reward_mode}
    train_spot_type = {train_spot_type}  val_spot_type = {val_spot_type}''')

    """初始化agent"""
    agent = D3QN(device=device,
                 va_net=VANet(),
                 learning_rate=learning_rate,
                 total_steps=total_steps,
                 adam_eps=adam_eps,
                 gamma=gamma,
                 epsilon=epsilon,
                 target_update=target_update,
                 VANet_path=VANet_path,
                 )

    """训练"""
    train(train_files=train_files,
          val_files=val_files,
          agent=agent,
          buffer_size=buffer_size,
          minimal_size=minimal_size,
          batch_size=batch_size,
          state_size=state_size,
          epochs=epochs,
          num_workers=num_workers,
          step_max=step_max,
          step_limit_max=step_limit_max,
          num_episodes=num_episodes,
          state_mode=state_mode,
          reward_mode=reward_mode,
          out_mode=out_mode,
          out_reward_mode=out_reward_mode,
          train_spot_type=train_spot_type,
          val_spot_type=val_spot_type,
          device=device,
          )
