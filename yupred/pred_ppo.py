import torch
import pandas as pd
from tqdm import tqdm
import os

from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    EnsureTyped,
    ThresholdIntensityd,
    NormalizeIntensityd,
    Orientationd,
    Invertd,
    SaveImaged,
)
from monai.data import CacheDataset, DataLoader, decollate_batch

from yualg.ppo import PPOPredict
from yuenv.ct_env import CTEnv


def predict(test_files,
            agent,
            state_size,
            norm_method,
            num_workers,
            step_max,
            step_limit_max,
            state_mode,
            reward_mode,
            out_mode,
            out_reward_mode,
            val_certain,
            val_spot_type,
            device,
            output_path,
            ):
    """预测"""

    agent.eval()  # 网络设置为验证模式

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

    # 定义测试集数据保存规则
    post_transforms = Compose([
        EnsureTyped(keys="output"),
        Invertd(
            keys="output",
            transform=transforms,
            orig_keys="image",
            meta_keys="output_meta_dict",
            orig_meta_keys="image_meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        SaveImaged(
            keys="output",
            meta_keys="output_meta_dict",
            output_dir=output_path,
            separate_folder=False,
            output_postfix="pred",
            resample=False,
        )
    ])

    # 创建数据集并加载数据
    test_ds = CacheDataset(data=test_files, transform=transforms, cache_rate=1.0, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=num_workers)

    # 开始预测
    with torch.no_grad():
        for test_data in test_loader:
            images, masks, preds, probs = (
                test_data["image"].to(device),
                test_data["mask"].to(device),
                test_data["pred"].to(device),
                test_data["prob"].to(device),
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
                if val_certain:  # 采用确定性验证策略
                    action = agent.take_certain_action(state)
                else:
                    action = agent.take_action(state)
                next_state, _, done = env.step(action)
                state = next_state
            test_data["output"] = env.to_pred().unsqueeze(0)
            _ = [post_transforms(i) for i in decollate_batch(test_data)]


if __name__ == '__main__':
    """固定随机种子"""
    set_determinism(seed=0)

    """设置GPU"""
    GPU_id = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}\n')

    """数据json路径"""
    json_path = '/workspace/data/rl/json/rl6_new.json'

    """预测数据保存路径"""
    output_path = '/workspace/data/rl/output6'

    """读取json获取所有图像文件名"""
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

    """制作文件名字典（以便后面使用字典制作数据集）"""
    data_dicts = [{"image": image_name, "mask": mask_name,
                   "pred": pred_name, "prob": prob_name}
                  for image_name, mask_name, pred_name, prob_name
                  in zip(test_images, test_masks, test_preds, test_probs)]

    """参数设置"""
    amp = True  # 是否使用混合精度训练和推断加速
    state_channel = 3  # 状态图通道数，可选2，3
    state_size = [21, 21, 9]  # 状态图大小
    norm_method = 'norm'  # 归一化方法，可选：min_max, norm
    num_workers = 0  # 数据加载线程数
    step_max = 5000  # 序列最大长度
    step_limit_max = 500  # 序列重复标注最大长度
    state_mode = 'pre'  # 状态模式，可选：pre(先标注再返回状态), post(返回状态后再标注)
    reward_mode = 'dice_inc_const'  # 奖励模式，可选：dice_inc, const, dice_inc_const
    out_mode = False  # 出边界是否停止，True则停止
    out_reward_mode = 'small'  # 出边界奖励模式，可选：small, large, step，0
    val_certain = False  # 是否在验证时采用确定性策略，False代表采用随机采样策略
    val_spot_type = 'prob_spot'  # 设置验证起点类型，可选ori_spot，prob_spot

    """网络加载"""
    from yunet import PolicyNet as PolicyNet
    policyNet_path = '/workspace/data/rl/model/ppo_policy09080.pth'
    policy_net = PolicyNet(state_channel).to(device)
    policy_net.load_state_dict(torch.load(policyNet_path, map_location=device))

    """初始化agent"""
    agent = PPOPredict(
        policy_net=policy_net,
        amp=amp,
        device=device,
    )

    """预测"""
    predict(
        test_files=data_dicts,
        agent=agent,
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
