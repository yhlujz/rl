import torch

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


def pred_ppo_step(test_files,
                  agent,
                  CTEnv,
                  state_channel,
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
                next_state, next_cover, next_step, _, done = env.step(action)
                state = next_state
                cover = next_cover
                step = next_step
            test_data["output"] = env.to_pred().unsqueeze(0)
            _ = [post_transforms(i) for i in decollate_batch(test_data)]