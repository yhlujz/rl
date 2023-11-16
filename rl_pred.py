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
    PolicyNet2,
    PolicyResNet,
    PolicyNetStep,
    VANet,
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

    # 预测数据保存路径
    output_path = '/workspace/data/rl/output6'

    """公用参数设置"""
    amp = True  # 是否使用混合精度训练和推断加速

    state_channel = 3  # 状态图通道数，可选2，3
    state_size = [21, 21, 9]  # 状态图大小
    norm_method = 'norm'  # 归一化方法，可选：min_max, norm

    num_workers = 0  # 数据加载线程数
    step_max = 5000  # 序列最大长度
    step_limit_max = 5000  # 限制无新标注的探索步数

    state_mode = 'pre'  # 状态模式，可选：pre(先标注再返回状态), post(返回状态后再标注)
    reward_mode = 'dice_inc_const'  # 奖励模式，可选：dice_inc, const, dice_inc_const
    out_mode = False  # 出边界是否停止，True则停止
    out_reward_mode = 'small'  # 出边界奖励模式，可选：small, large, step，0

    val_certain = False  # 是否在验证时采用确定性策略，False代表采用随机采样策略
    val_spot_type = 'max_prob_spot'  # 设置验证起点类型，可选random_spot，max_prob_spot

    # 网络选择，需要根据不同强化学习算法选择一个或两个网络
    net_name = ['PolicyNetStep']
    if 'PolicyNet' in net_name:
        policy_net = PolicyNet(state_channel).to(device)
    if 'PolicyNet2' in net_name:
        policy_net = PolicyNet2(state_channel).to(device)
    if 'PolicyResNet' in net_name:
        policy_net = PolicyResNet(state_channel).to(device)
    if 'PolicyNetStep' in net_name:
        policy_net = PolicyNetStep(state_channel).to(device)
    if 'VANet' in net_name:
        q_net = VANet(state_channel).to(device)

    """特定参数设置"""
    # 模型加载路径
    policyNet_path = '/workspace/data/rl/model/ppo_step_policy11095.pth'
    # 模型加载
    policy_net.load_state_dict(torch.load(policyNet_path, map_location=device))

    """预测"""
    if algo == 'ppo':
        agent = PPOPredict(
            policy_net=policy_net,
            amp=amp,
            device=device,
        )
        pred_ppo(
            test_files=test_files,
            agent=agent,
            CTEnv=CTEnv,
            state_size=state_size,
            norm_method=norm_method,
            num_workers=num_workers,
            step_max=step_max,
            step_limit_max=step_limit_max,
            state_mode=state_mode,
            reward_mode=reward_mode,
            out_mode=out_mode,
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
            device=device,
        )
        pred_ppo_step(
            test_files=test_files,
            agent=agent,
            CTEnv=CTEnvStep,
            state_channel=state_channel,
            state_size=state_size,
            norm_method=norm_method,
            num_workers=num_workers,
            step_max=step_max,
            step_limit_max=step_limit_max,
            state_mode=state_mode,
            reward_mode=reward_mode,
            out_mode=out_mode,
            out_reward_mode=out_reward_mode,
            val_certain=val_certain,
            val_spot_type=val_spot_type,
            device=device,
            output_path=output_path,
        )
