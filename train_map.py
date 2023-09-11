from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    SpatialPadd,
    ScaleIntensityRanged,
    EnsureTyped,
    EnsureType
)
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.networks.nets import UNet

import torch
import os
import pandas as pd

from tqdm import tqdm


def train(epochs,
          batch_size,
          learning_rate,
          num_workers,
          device,
          ):
    """训练"""

    # 模型保存路径
    model_path = "/workspace/data/rl/model/map_UNet.pth"

    # 数据json路径
    json_path = '/workspace/data/rl/json/rl6.json'

    # 读取json获取所有图像文件名
    train_images = []
    train_masks = []
    df = pd.read_json(json_path)
    for _, row in tqdm(df.iterrows()):
        if row['dataset'] == 'train':
            train_images.append(row['image_path'])
            train_masks.append(row['mask_path'])

    # 制作文件名字典（以便后面使用字典制作数据集）
    data_dicts = [{"image": image_name, "mask": mask_name}
                  for image_name, mask_name in zip(train_images, train_masks)]

    # 划分得到训练集字典和验证集字典
    val_len = round(len(df) / 10)
    train_files, val_files = data_dicts[:-val_len], data_dicts[-val_len:]

    # 载入模型以及初始化模型参数
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # 定义数据变换规则
    transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),  # 载入图像
            EnsureChannelFirstd(keys=["image", "mask"]),  # 添加通道维
            ScaleIntensityRanged(keys=["image"], a_min=-135, a_max=215, b_min=0.0, b_max=1.0, clip=True),  # 设置窗宽窗位
            SpatialPadd(keys=["image", "mask"], spatial_size=[128, 160, 64]),  # padding到统一大小
            EnsureTyped(keys=["image", "mask"])  # 转换为tensor
        ]
    )

    # 创建训练集
    train_ds = CacheDataset(data=train_files, transform=transforms, cache_rate=1.0, num_workers=num_workers)
    # 加载训练集数据
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    n_train = len(train_loader)

    # 创建验证集并加载数据
    val_ds = CacheDataset(data=val_files, transform=transforms, cache_rate=1.0, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    n_val = len(val_loader)

    # 打印网络训练信息
    print(f'''Starting training:
                Epochs:               {epochs}
                Batch size:           {batch_size}
                Learning rate:        {learning_rate}
                Training size:        {n_train}
                Validation size:      {n_val}
                Device:               {device.type}
                ''')

    # 设置损失，优化器，梯度缩放以及评价指标
    loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # 定义预测值和真实标签的变换（用于验证集计算评价指标）
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

    # 设置训练轮次
    max_epochs = epochs

    # 设置经过多少轮进行一次验证
    val_interval = 2

    # 记录最佳评价指标
    best_metric = -1

    # 记录得到最佳评价指标的轮次
    best_metric_epoch = -1

    # 开始训练
    for epoch in range(max_epochs):
        net.train()  # 网络设置为训练模式
        epoch_loss = 0
        step = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for batch_data in train_loader:
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["mask"].to(device),
                )
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = net(inputs)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                pbar.update(1)  # 进度条更新
                step += 1
                epoch_loss += loss.item()  # 记录每个epoch总损失
                pbar.set_postfix(**{'loss (batch)': loss.item()})  # 设置进度条后缀为每个batch的loss
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss}")

        # 验证部分
        if (epoch + 1) % val_interval == 0:
            net.eval()
            with torch.no_grad():
                for val_data in tqdm(val_loader, total=n_val, desc='Validation round', unit='batch'):
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["mask"].to(device),
                    )
                    with torch.cuda.amp.autocast():
                        val_outputs = net(val_inputs)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # 计算总平均指标和各分类平均指标
                metric = dice_metric.aggregate().item()
                # 累加器清零
                dice_metric.reset()

                # 比较记录最好的模型及轮次
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(net.state_dict(), model_path)
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric} \n"
                    f"best mean dice: {best_metric} at epoch: {best_metric_epoch}"
                )
    print(
        f"train completed, best_metric: {best_metric} at epoch: {best_metric_epoch}"
    )


if __name__ == '__main__':
    '''主函数'''

    # 固定随机种子
    set_determinism(seed=0)

    # GPU设置
    GPU_id = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}\n')

    # 参数设置
    epochs = 500
    batch_size = 2
    learning_rate = 1e-5
    num_workers = 0

    # 开始训练
    train(epochs=epochs,
          batch_size=batch_size,
          learning_rate=learning_rate,
          num_workers=num_workers,
          device=device,
          )
