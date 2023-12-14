import torch
import pandas as pd
from tqdm import tqdm
import os
from monai.utils import set_determinism

# 导入环境
from yuenv import CTEnvStep, CTEnv

# 导入算法
from yualgo import PPOPredict, PPOStepPredict

# 导入网络
from yunet import (
    PolicyNet,
    PolicyNetLight,
    PolicyNet2,
    PolicyResNet,
    PolicyNetStep,
    PolicyNetStep2,
    PolicyNetStepGelu,
)

# 导入预测流程
from yupred import pred_ppo, pred_ppo_step


def load_dataset(json_path):
    """加载测试集"""

    # 读取json获取所有图像文件名
    test_images = []
    test_masks = []
    test_preds = []
    test_probs = []
    df = pd.read_json(json_path)
    for _, row in tqdm(df.iterrows()):
        if row['dataset'] == 'test':
            test_images.append(row['image_path'])
            test_masks.append(row['mask_path'])
            test_preds.append(row['pred_path'])
            test_probs.append(row['prob_path'])

    # 制作文件名字典（以便后面使用字典制作数据集）
    data_dicts = [{"image": image_name, "mask": mask_name,
                   "pred": pred_name, "prob": prob_name}
                  for image_name, mask_name, pred_name, prob_name
                  in zip(test_images, test_masks, test_preds, test_probs)]

    return data_dicts


if __name__ == '__main__':

    """必要参数设置"""
    # 设置GPU
    GPU_id = '7'
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}\n')

    # 固定随机种子
    set_determinism(seed=0)

    # 数据json路径
    json_path = '/workspace/data/rl/json/rl6_new.json'
    # 加载测试集
    test_files = load_dataset(json_path)

    # 算法选择，可选ppo,ppo_step,sac,d3qn
    algo = 'ppo_step'

    # 环境选择，可选CTEnv,CTEnvStep
    Env = CTEnv

    # 模型编号
    id = '11095'

    # 预测数据保存路径
    output_path = f'/workspace/data/rl/output6/{algo}{id}'

    """公用参数设置"""
    amp = True  # 是否使用混合精度训练和推断加速
    comp = True  # 是否使用编译加速

    action_num = 6  # 动作个数，可选6，7(增加回到起点)，26，27(增加回到起点)
    state_num = [0, 1, 2]  # 状态图包含的图像，0image，1pred，2prob，3preded
    state_channel = len(state_num)  # 状态图通道数
    state_size = [21, 21, 9]  # 状态图大小
    norm_method = 'norm'  # 归一化方法，可选：min_max, norm
    state_mode = 'pre'  # 状态模式，可选：pre(先标注再返回状态), post(返回状态后再标注)
    reward_mode = 'dice_inc_const'  # 奖励模式，可选：dice_inc, const, dice_inc_const
    out_reward_mode = 'small'  # 出边界奖励模式，可选：small, large, step，0
    val_spot_type = 'edge_spot'  # 设置验证起点类型，可选random_spot，max_prob_spot，edge_spot

    num_workers = 0  # 数据加载线程数
    step_max = 5000  # 序列最大长度

    val_certain = False  # 是否在验证时采用确定性策略，False代表采用随机采样策略

    net_name = ['PolicyNetStep']  # 网络选择，需要根据不同强化学习算法选择一个或两个网络
    OI = True  # 是否使用正交初始化

    if 'PolicyNet' in net_name:
        policy_net = PolicyNet(action_num, state_channel, OI).to(device)
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
    if 'PolicyNetStepGelu' in net_name:
        policy_net = PolicyNetStepGelu(action_num, state_channel, OI).to(device)

    """特定参数设置"""
    # PPO算法模型加载
    if algo == "ppo" or algo == "ppo_step":
        policyNet_path = f'/workspace/data/rl/model/{algo}_policy{id}.pth'
        policy_net.load_state_dict(torch.load(policyNet_path, map_location=device))

    """预测"""
    if algo == 'ppo':
        agent = PPOPredict(
            policy_net=policy_net,
            amp=amp,
            comp=comp,
            device=device,
        )
        pred_ppo(
            test_files=test_files,
            agent=agent,
            Env=Env,
            action_num=action_num,
            state_num=state_num,
            state_size=state_size,
            norm_method=norm_method,
            num_workers=num_workers,
            step_max=step_max,
            state_mode=state_mode,
            reward_mode=reward_mode,
            out_reward_mode=out_reward_mode,
            val_certain=val_certain,
            val_spot_type=val_spot_type,
            device=device,
            output_path=output_path,
        )

    if algo == 'ppo_step':
        agent = PPOStepPredict(
            policy_net=policy_net,
            amp=amp,
            comp=comp,
            device=device,
        )
        pred_ppo_step(
            test_files=test_files,
            agent=agent,
            Env=Env,
            action_num=action_num,
            state_num=state_num,
            state_size=state_size,
            norm_method=norm_method,
            num_workers=num_workers,
            step_max=step_max,
            state_mode=state_mode,
            reward_mode=reward_mode,
            out_reward_mode=out_reward_mode,
            val_certain=val_certain,
            val_spot_type=val_spot_type,
            device=device,
            output_path=output_path,
        )
