import torch
import pandas as pd
from tqdm import tqdm
import os
from yunet import SpotNet

from monai.utils import set_determinism

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    EnsureTyped,
)
from monai.data import CacheDataset, DataLoader

import logging
from datetime import datetime


def train(net,
          device,
          epochs,
          batch_size,
          learning_rate,
          num_workers,
          model_path):
    """训练"""

    # 定义训练集数据变换规则
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),   #载入图像
            EnsureChannelFirstd(keys=["image"]),  #添加通道维
            ScaleIntensityRanged(keys=["image"], a_min=-135, a_max=215, b_min=0.0, b_max=1.0, clip=True),  #设置窗宽窗位
            EnsureTyped(keys=["image", "label"])     #转换为tensor
        ]
    )

    # 定义验证集数据变换规则
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(keys=["image"], a_min=-135, a_max=215, b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys=["image", "label"])
        ]
    )

    # 创建数据集并加载数据
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    n_train = len(train_loader)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    n_val = len(val_loader)

    # log记录
    logging.info(f'''Starting training:
                 Epochs:               {epochs}
                 Batch size:           {batch_size}
                 Learning rate:        {learning_rate}
                 Training size:        {n_train}
                 Validation size:      {n_val}
                 Device:               {device.type}
                 ''')
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
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    # 设置训练轮次
    max_epochs = epochs

    # 设置经过多少轮进行一次验证
    val_interval = 2

    # 记录最佳评价指标
    best_loss = 1000

    # 记录得到最佳评价指标的轮次
    best_loss_epoch = -1

    # 开始训练
    for epoch in range(max_epochs):
        net.train()   # 网络设置为训练模式
        epoch_loss = 0
        step = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for batch_data in train_loader:
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
                optimizer.zero_grad()
                loss = 0
                with torch.cuda.amp.autocast():
                    outputs = net(inputs)
                    for i in range(len(outputs)):
                        spots = torch.nonzero(labels[i])
                        loss += torch.min(torch.pairwise_distance(outputs[i], spots))
                    loss /= len(outputs)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                pbar.update(1)   # 进度条更新
                step += 1

                epoch_loss += loss.item()   # 记录每个epoch总损失

                pbar.set_postfix(**{'loss (batch)': loss.item()})      #设置进度条后缀为每个batch的loss

        epoch_loss /= step
        logging.info(f''''
                        epoch{epoch+1} loss: {epoch_loss}
                        ''')
        print(f"epoch {epoch + 1} average loss: {epoch_loss: .4f}")

        # 验证部分
        if (epoch + 1) % val_interval == 0:
            net.eval()
            epoch_loss = 0
            with torch.no_grad():
                for val_data in tqdm(val_loader, total=n_val, desc='Validation round', unit='batch'):
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    loss = 0
                    outputs = net(val_inputs)
                    for i in range(len(outputs)):
                        spots = torch.nonzero(val_labels[i])
                        loss += torch.min(torch.pairwise_distance(outputs[i], spots))
                    loss /= len(outputs)
                    epoch_loss += loss
                # 保存模型
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_loss_epoch = epoch + 1
                    torch.save(net.state_dict(), model_path)
                    print("saved new best model")
                # 记录loss及轮次
                logging.info(f'''
                             current epoch: {epoch + 1} current loss: {epoch_loss}
                             best loss: {best_loss} at epoch: {best_loss_epoch}
                             ''')
                print(f'''
                      current epoch: {epoch + 1} current loss: {epoch_loss}
                      best loss: {best_loss} at epoch: {best_loss_epoch}
                      ''')
    logging.info(f'''
                 train completed, best loss: {best_loss} at epoch: {best_loss_epoch}
                 ''')
    print(f'''
          train completed, best loss: {best_loss} at epoch: {best_loss_epoch}
          ''')


def predict(net,
            device,
            num_workers
            ):
    """预测"""

    # 网络设置为验证模式
    net.eval()

    # 定义测试集数据变换规则
    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),  # 载入图像
            EnsureChannelFirstd(keys=["image"]),  # 添加通道维
            ScaleIntensityRanged(keys=["image"], a_min=-135, a_max=215, b_min=0.0, b_max=1.0, clip=True),  # 设置窗宽窗位
            EnsureTyped(keys=["image", "label"])  # 转换为tensor
        ]
    )

    # 创建数据集并加载数据
    test_ds = CacheDataset(data=data_dicts, transform=test_transforms, cache_rate=1.0, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=num_workers)
    n_test = len(test_loader)

    # log记录
    logging.info(f'''Starting testing:
                 Testing size:         {n_test}
                 Device:               {device.type}
                 ''')
    # 打印网络测试信息
    print(f'''Starting testing:
          Testing size:         {n_test}
          Device:               {device.type}
          ''')

    # 开始预测
    with torch.no_grad():
        for test_data in tqdm(test_loader, total=n_test, desc='Test round', unit='it'):
            test_inputs, test_labels = (
                test_data["image"].to(device),
                test_data["label"].to(device),
            )
            outputs = net(test_inputs)
            position = torch.round(outputs[0]) # 四舍五入取整
            spots = torch.nonzero(test_labels[0])
            loss = torch.min(torch.pairwise_distance(position, spots))
            # 记录坐标和loss
            logging.info(f'''
                         filename: {test_data['image_meta_dict']['filename_or_obj']}
                         position: {position} loss: {loss}
                         ''')
        logging.info('''Test Completed!''')
        print('''Test Completed!''')


'''主函数'''
if __name__ == '__main__':

    """数据json路径"""
    json_path = 'D:/alec/dataset/rl/json/rl.json'

    """网络时间"""
    date_time = datetime.now().strftime("%Y%m%d")

    """网络路径"""
    model_path = f'/workspace/data/rl/model/spot{date_time}.pth'

    """log路径"""
    log_path = f'/workspace/data/rl/log/spot{date_time}.log'

    """获取所有图像文件名"""
    train_images = []
    train_labels = []
    df = pd.read_json(json_path)
    for index, row in tqdm(df.iterrows()):
        if row['dataset'] == 'train':
            train_images.append(row['image_path'])
            train_labels.append(row['mask_path'])

    """制作文件名字典（以便后面使用字典制作数据集）"""
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

    """划分训练集字典和验证集字典"""
    train_files, val_files = data_dicts[:-98], data_dicts[-98:]

    """固定随机种子"""
    set_determinism(seed=0)

    # 设置日志写入规则
    logging.basicConfig(level=logging.INFO,
                        filename=log_path,
                        filemode='a',
                        format='%(levelname)s: %(message)s',
                        force=True)  # 写入info级别以上的日志

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}\n')

    # 初始化网络参数
    net = SpotNet()
    net.to(device=device)

    # 训练
    train(net=net,
          device=device,
          epochs=1000,
          batch_size=8,
          learning_rate=1e-5,
          num_workers=0,
          model_path=model_path
          )

    # 预测
    net.load_state_dict(torch.load(model_path, map_location=device))
    print('Model loaded!')

    predict(net=net,
            device=device,
            num_workers=0
            )
