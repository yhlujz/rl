import torch
import pandas as pd
from tqdm import tqdm
import os
import logging
from datetime import datetime
from monai.utils import set_determinism

# 导入环境
from yuenv import CTEnvStep, CTEnv

# 导入算法
from yualgo import PPO, PPOStep, D3QN, SAC

# 导入网络
from yunet import (
    PolicyNet,
    PolicyNetBig,
    PolicyNetLight,
    PolicyNet2,
    PolicyResNet,
    PolicyNetStep,
    PolicyNetStep2,
    PolicyNetStepBig,
    PolicyNetStepGelu,
    ValueNet,
    ValueNetBig,
    ValueNetLight,
    ValueNet2,
    ValueResNet,
    ValueNetStep,
    ValueNetStep2,
    ValueNetStepBig,
    ValueNetStepGelu,
    QNet,
    QNet2,
    VANet,
    VANet2,
    VANetRelu,
)

# 导入训练流程
from yutrain import train_ppo, train_d3qn, train_ppo_step, train_sac


def divide_dataset(json_path):
    """划分数据集"""

    # 读取json获取所有图像文件名
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

    # 制作文件名字典（以便后面使用字典制作数据集）
    data_dicts = [{"image": image_name, "mask": mask_name,
                   "pred": pred_name, "prob": prob_name}
                  for image_name, mask_name, pred_name, prob_name
                  in zip(train_images, train_masks, train_preds, train_probs)]

    # 划分得到训练集字典和验证集字典
    val_len = round(len(df) / 10)
    train_files, val_files = data_dicts[:-val_len], data_dicts[-val_len:]

    return train_files, val_files


if __name__ == '__main__':

    """必要参数设置"""
    # 训练编号
    id = '0'

    # 设置GPU
    GPU_id = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}\n')

    # 获取当前日期
    date_time = datetime.now().strftime("%m%d")

    # 固定随机种子
    set_determinism(seed=0)

    # 数据json路径
    json_path = '/workspace/data/rl/json/rl6_new.json'
    # 划分数据集
    train_files, val_files = divide_dataset(json_path)

    # 算法选择，可选ppo,ppo_step,sac,d3qn
    algo = 'ppo_step'

    # 环境选择，可选CTEnv,CTEnvStep
    Env = CTEnvStep

    # 设置模型保存路径
    valueNet_path = f'/workspace/data/rl/model/{algo}_value{date_time}{id}.pth'
    policyNet_path = f'/workspace/data/rl/model/{algo}_policy{date_time}{id}.pth'
    VANet_path = f'/workspace/data/rl/model/{algo}_va{date_time}{id}.pth'

    # 设置log保存路径
    log_path = f'/workspace/data/rl/log/{algo}{date_time}{id}.log'

    # 设置日志写入规则
    logging.basicConfig(level=logging.INFO,
                        filename=log_path,
                        filemode='a',
                        format='%(levelname)s: %(message)s',
                        force=True)  # 强制写入info级别以上的日志

    """公用参数设置"""
    lr_decay = [1, 0]  # 学习率衰减（线性衰减）范围
    optimizer = 'AdamW'  # 优化器选用，可选：Adam, AdamW
    adam_eps = 1e-5  # adam优化器限制值
    gamma = 0.98  # 价值估计倍率

    amp = True  # 是否使用混合精度训练和推断加速
    comp = True  # 是否使用编译加速

    action_num = 6  # 动作个数，可选6，7(增加回到起点)，26，27(增加回到起点)
    state_num = [0, 2, 3]  # 状态图包含的图像，0image，1preded，2prob，3pred
    state_channel = len(state_num)  # 状态图通道数
    state_size = [21, 21, 9]  # 状态图大小
    norm_method = 'norm'  # 状态图归一化方法，可选：min_max, norm
    state_mode = 'pre'  # 状态模式，可选：pre(先标注再返回状态), post(返回状态后再标注)
    reward_mode = 'dice_inc_const'  # 奖励模式，可选：dice_inc, const, dice_inc_const
    out_reward_mode = 'small'  # 出边界奖励模式，可选：small, large, 0
    train_spot_type = 'edge_spot'  # 设置训练起点类型，可选random_spot，max_prob_spot，edge_spot
    val_spot_type = 'edge_spot'  # 设置验证起点类型，可选random_spot，max_prob_spot，edge_spot

    epochs = 100  # 总循环次数
    num_workers = 0  # 数据加载线程数
    step_max = 5000  # 序列最大长度
    num_episodes = 1  # 每张图训练序列数
    agent_epochs = 10  # 每个序列梯度下降迭代次数
    total_steps = (epochs * len(train_files) * num_episodes *
                   agent_epochs)  # 计算梯度下降迭代总步数，后续进行学习率衰减使用

    train_certain = False  # 是否在训练时采用确定性策略，False代表采用随机采样策略
    val_certain = False  # 是否在验证时采用确定性策略，False代表采用随机采样策略

    net_name = ['PolicyNetStep2', 'ValueNetStep2']  # 网络选择，需要根据不同强化学习算法选择一个或两个网络
    OI = True  # 是否使用正交初始化

    # 策略网络
    if 'PolicyNet' in net_name:
        policy_net = PolicyNet(action_num, state_channel, OI).to(device)
    if 'PolicyNetBig' in net_name:
        policy_net = PolicyNetBig(action_num, state_channel, OI).to(device)
    if 'PolicyNetLight' in net_name:
        policy_net = PolicyNetLight(action_num, state_channel, OI).to(device)
    if 'PolicyNet2' in net_name:
        policy_net = PolicyNet2(action_num, state_channel, OI).to(device)
    if 'PolicyResNet' in net_name:
        policy_net = PolicyResNet(action_num, state_channel, OI).to(device)
    if 'PolicyNetStep' in net_name:
        policy_net = PolicyNetStep(action_num, state_channel, OI).to(device)
    if 'PolicyNetStep2' in net_name:
        policy_net = PolicyNetStep2(action_num, state_channel, OI).to(device)
    if 'PolicyNetStepBig' in net_name:
        policy_net = PolicyNetStepBig(action_num, state_channel, OI).to(device)
    if 'PolicyNetStepGelu' in net_name:
        policy_net = PolicyNetStepGelu(action_num, state_channel, OI).to(device)

    # 价值网络
    if 'ValueNet' in net_name:
        value_net = ValueNet(state_channel, OI).to(device)
    if 'ValueNetBig' in net_name:
        value_net = ValueNetBig(state_channel, OI).to(device)
    if 'ValueNetLight' in net_name:
        value_net = ValueNetLight(state_channel, OI).to(device)
    if 'ValueNet2' in net_name:
        value_net = ValueNet2(state_channel, OI).to(device)
    if 'ValueResNet' in net_name:
        value_net = ValueResNet(state_channel, OI).to(device)
    if 'ValueNetStep' in net_name:
        value_net = ValueNetStep(state_channel, OI).to(device)
    if 'ValueNetStep2' in net_name:
        value_net = ValueNetStep2(state_channel, OI).to(device)
    if 'ValueNetStepBig' in net_name:
        value_net = ValueNetStepBig(state_channel, OI).to(device)
    if 'ValueNetStepGelu' in net_name:
        value_net = ValueNetStepGelu(state_channel, OI).to(device)

    # q值网络
    if 'QNet' in net_name:
        q_net = QNet(action_num, state_channel, OI).to(device)
    if 'QNet2' in net_name:
        q_net = QNet2(action_num, state_channel, OI).to(device)
    if 'VANet' in net_name:
        q_net = VANet(action_num, state_channel, OI).to(device)
    if 'VANet2' in net_name:
        q_net = VANet2(action_num, state_channel, OI).to(device)
    if 'VANetRelu' in net_name:
        q_net = VANetRelu(action_num, state_channel, OI).to(device)

    """特定参数设置"""
    # PPO算法
    if algo == "ppo" or algo == "ppo_step":
        actor_lr = 1e-4  # 策略函数初始学习率
        critic_lr = 1e-3  # 价值函数初始学习率
        lmbda = 0.95  # 优势估计倍率
        adv_norm = True  # 是否进行优势估计标准化
        eps = 0.2  # ppo算法限制值
        entropy_coef = 0.01  # 策略熵系数，可设置为0

    # D3QN算法
    if algo == "d3qn":
        learning_rate = 1e-4  # 初始学习率
        epsilon = 0.01  # 贪婪因子
        target_update = 50  # 多少次迭代后同步两个网络参数
        buffer_size = 500000  # 回放池大小
        minimal_size = 100000  # 回放池最小大小
        batch_size = 5000  # 每次迭代的batch大小

    # SAC算法
    if algo == "sac":
        actor_lr = 1e-4  # 策略函数初始学习率
        critic_lr = 1e-3  # 价值函数初始学习率
        alpha_lr = 1e-4  # 熵初始学习率
        tau = 0.005  # 软更新参数
        target_entropy = -1  # 目标熵
        buffer_size = 500000  # 回放池大小
        minimal_size = 100000  # 回放池最小大小
        batch_size = 5000  # 每次迭代的batch大小

    """记录参数信息"""
    # PPO算法
    if algo == 'ppo_step' or algo == 'ppo':
        logging.info(f'''
        环境参数：
        json_path = {json_path}
        state_num = {state_num}  state_size = {state_size}
        norm_method = {norm_method}
        step_max = {step_max}  action_num = {action_num}
        state_mode = {state_mode}
        reward_mode = {reward_mode}
        out_reward_mode = {out_reward_mode}
        train_spot_type = {train_spot_type}  val_spot_type = {val_spot_type}
        算法参数：
        net_name = {net_name}  OI = {OI}
        gamma = {gamma}
        lmbda = {lmbda}  eps = {eps}
        adv_norm = {adv_norm}  entropy_coef = {entropy_coef}
        train_certain = {train_certain}  val_certain = {val_certain}
        训练参数：
        actor_lr = {actor_lr}  critic_lr = {critic_lr}  lr_decay = {lr_decay}
        optimizer = {optimizer}  adam_eps = {adam_eps}
        epochs = {epochs}  num_episodes = {num_episodes}  agent_epochs = {agent_epochs}
        amp = {amp}  comp = {comp}''')

    # D3QN算法
    if algo == 'd3qn':
        logging.info(f'''
        环境参数：
        json_path = {json_path}
        state_num = {state_num}  state_size = {state_size}
        norm_method = {norm_method}
        step_max = {step_max}  action_num = {action_num}
        state_mode = {state_mode}
        reward_mode = {reward_mode}
        out_reward_mode = {out_reward_mode}
        train_spot_type = {train_spot_type}  val_spot_type = {val_spot_type}
        算法参数：
        net_name = {net_name}  OI = {OI}
        gamma = {gamma}
        epsilon = {epsilon}
        target_update = {target_update}
        buffer_size = {buffer_size}  minimal_size = {minimal_size}
        batch_size = {batch_size}
        train_certain = {train_certain}  val_certain = {val_certain}
        训练参数：
        lr = {learning_rate}  lr_decay = {lr_decay}
        optimizer = {optimizer}  adam_eps = {adam_eps}
        epochs = {epochs}  num_episodes = {num_episodes}  agent_epochs = {agent_epochs}
        amp = {amp}  comp = {comp}''')

    # SAC算法
    if algo == 'sac':
        logging.info(f'''
        环境参数：
        json_path = {json_path}
        state_num = {state_num}  state_size = {state_size}
        norm_method = {norm_method}
        step_max = {step_max}  action_num = {action_num}
        state_mode = {state_mode}
        reward_mode = {reward_mode}
        out_reward_mode = {out_reward_mode}
        train_spot_type = {train_spot_type}  val_spot_type = {val_spot_type}
        算法参数：
        net_name = {net_name}  OI = {OI}
        gamma = {gamma}
        tau = {tau}
        target_entropy = {target_entropy}
        buffer_size = {buffer_size}  minimal_size = {minimal_size}
        batch_size = {batch_size}
        train_certain = {train_certain}  val_certain = {val_certain}
        训练参数：
        actor_lr = {actor_lr}  critic_lr = {critic_lr}  alpha_lr = {alpha_lr}
        optimizer = {optimizer}  adam_eps = {adam_eps}
        epochs = {epochs}  num_episodes = {num_episodes}  agent_epochs = {agent_epochs}
        amp = {amp}  comp = {comp}''')

    """训练"""
    # PPO算法
    if algo == 'ppo':
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
                    eps=eps,
                    entropy_coef=entropy_coef,
                    gamma=gamma,
                    adv_norm=adv_norm,
                    amp=amp,
                    comp=comp,
                    device=device,
                    valueNet_path=valueNet_path,
                    policyNet_path=policyNet_path,
                    )
        train_ppo(train_files=train_files,
                  val_files=val_files,
                  agent=agent,
                  Env=Env,
                  action_num=action_num,
                  state_num=state_num,
                  state_size=state_size,
                  norm_method=norm_method,
                  epochs=epochs,
                  num_workers=num_workers,
                  step_max=step_max,
                  num_episodes=num_episodes,
                  state_mode=state_mode,
                  reward_mode=reward_mode,
                  out_reward_mode=out_reward_mode,
                  train_certain=train_certain,
                  val_certain=val_certain,
                  train_spot_type=train_spot_type,
                  val_spot_type=val_spot_type,
                  device=device,
                  )

    if algo == 'ppo_step':
        agent = PPOStep(policy_net=policy_net,
                        value_net=value_net,
                        actor_lr=actor_lr,
                        critic_lr=critic_lr,
                        lr_decay=lr_decay,
                        total_steps=total_steps,
                        optimizer=optimizer,
                        adam_eps=adam_eps,
                        lmbda=lmbda,
                        agent_epochs=agent_epochs,
                        eps=eps,
                        entropy_coef=entropy_coef,
                        gamma=gamma,
                        adv_norm=adv_norm,
                        amp=amp,
                        comp=comp,
                        device=device,
                        valueNet_path=valueNet_path,
                        policyNet_path=policyNet_path,
                        )
        train_ppo_step(train_files=train_files,
                       val_files=val_files,
                       agent=agent,
                       Env=Env,
                       action_num=action_num,
                       state_num=state_num,
                       state_size=state_size,
                       norm_method=norm_method,
                       epochs=epochs,
                       num_workers=num_workers,
                       step_max=step_max,
                       num_episodes=num_episodes,
                       state_mode=state_mode,
                       reward_mode=reward_mode,
                       out_reward_mode=out_reward_mode,
                       train_certain=train_certain,
                       val_certain=val_certain,
                       train_spot_type=train_spot_type,
                       val_spot_type=val_spot_type,
                       device=device,
                       )

    # D3QN算法
    if algo == 'd3qn':
        agent = D3QN(device=device,
                     q_net=q_net,
                     learning_rate=learning_rate,
                     total_steps=total_steps,
                     adam_eps=adam_eps,
                     gamma=gamma,
                     epsilon=epsilon,
                     target_update=target_update,
                     VANet_path=VANet_path,
                     )
        train_d3qn(train_files=train_files,
                   val_files=val_files,
                   agent=agent,
                   Env=Env,
                   buffer_size=buffer_size,
                   minimal_size=minimal_size,
                   batch_size=batch_size,
                   agent_epochs=agent_epochs,
                   action_num=action_num,
                   state_num=state_num,
                   state_size=state_size,
                   epochs=epochs,
                   num_workers=num_workers,
                   step_max=step_max,
                   num_episodes=num_episodes,
                   state_mode=state_mode,
                   reward_mode=reward_mode,
                   out_reward_mode=out_reward_mode,
                   train_spot_type=train_spot_type,
                   val_spot_type=val_spot_type,
                   device=device,
                   )

    # SAC算法
    if algo == 'sac':
        agent = SAC(device=device,
                    policy_net=policy_net,
                    q_net=q_net,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    alpha_lr=alpha_lr,
                    target_entropy=target_entropy,
                    gamma=gamma,
                    tau=tau,
                    adam_eps=adam_eps,
                    total_steps=total_steps,
                    policyNet_path=policyNet_path,
                    )
        train_sac(train_files=train_files,
                  val_files=val_files,
                  agent=agent,
                  Env=Env,
                  buffer_size=buffer_size,
                  minimal_size=minimal_size,
                  batch_size=batch_size,
                  agent_epochs=agent_epochs,
                  action_num=action_num,
                  state_num=state_num,
                  state_size=state_size,
                  epochs=epochs,
                  num_workers=num_workers,
                  step_max=step_max,
                  num_episodes=num_episodes,
                  state_mode=state_mode,
                  reward_mode=reward_mode,
                  out_reward_mode=out_reward_mode,
                  train_spot_type=train_spot_type,
                  val_spot_type=val_spot_type,
                  device=device,
                  )
