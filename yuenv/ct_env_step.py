import torch
import torch.nn.functional as F

import numpy as np


class CTEnvStep:
    """定义CT图像环境"""
    def __init__(self,
                 image,
                 mask,
                 preded,
                 prob,
                 state_channel,
                 state_size,
                 step_max,
                 step_limit_max,
                 state_mode,
                 reward_mode,
                 out_mode,
                 out_reward_mode,
                 ):
        self.image = image  # 初始化image
        self.mask = mask.int()  # 初始化mask
        self.preded = preded.int()  # 初始化已预测图像
        self.prob = prob  # 初始化概率图

        self.state_channel = state_channel  # 初始化状态图通道数
        self.state_size = state_size  # 初始化状态图大小
        self.step_max = step_max  # 限制最大步数
        self.step_limit_max = step_limit_max  # 限制无新标注的探索步数
        self.state_mode = state_mode  # 设置状态返回模式
        self.reward_mode = reward_mode  # 设置奖励函数模式
        self.out_mode = out_mode  # 设置边界外是否停止
        self.out_reward_mode = out_reward_mode  # 设置边界外奖励函数

        self.step_n = 0  # 初始步数
        self.step_limit_n = 0  # 初始无新标注步数
        self.dice = 0  # 初始化dice值
        self.cover = 0  # 初始化cover值

        # 根据状态图大小对原图像、标注图像、已预测图像和概率图进行padding
        self.pad_length = int((self.state_size[0] - 1) / 2)
        self.pad_width = int((self.state_size[1] - 1) / 2)
        self.pad_depth = int((self.state_size[2] - 1) / 2)
        pad_size = (self.pad_depth, self.pad_depth,
                    self.pad_width, self.pad_width,
                    self.pad_length, self.pad_length)
        self.image_padding = F.pad(self.image, pad_size, 'constant', 0)
        self.mask_padding = F.pad(self.mask, pad_size, 'constant', 0)
        self.preded_padding = F.pad(self.preded, pad_size, 'constant', 0)
        self.prob_padding = F.pad(self.prob, pad_size, 'constant', 0)

        # 创建与标注大小相同的预测图
        self.pred_padding = torch.zeros_like(self.mask_padding).int()

        # 根据mask随机初始化起点(注意：起点坐标为padding后的坐标)
        spots = torch.nonzero(self.mask_padding)
        self.mask_spots_n = len(spots)  # 记录mask中标注点的数量
        i = np.random.randint(self.mask_spots_n)
        self.random_spot = spots[i].tolist()  # 从标注中随机选取一点作为起点
        self.spot = self.random_spot  # 初始化智能体位置坐标

        # 根据已预测概率图初始化起点(使用概率最大点的坐标)
        self.max_prob_spot = torch.nonzero(
            self.prob_padding == torch.max(self.prob_padding))[0].tolist()

    def reset(self, spot_type):
        """回归初始状态（预测图只包含随机起点的状态）并返回初始状态值"""
        self.step_n = 1  # 步数置1
        self.step_limit_n = 0  # 无新标注步数置0
        if spot_type == 'random_spot':
            self.spot = self.random_spot  # 重新初始化智能体位置坐标
        elif spot_type == 'max_prob_spot':
            self.spot = self.max_prob_spot
        else:
            print('error: invalid spot_type!')
        self.pred_padding = torch.zeros_like(self.mask_padding).int()  # 初始化预测图像
        if self.state_mode == 'post':
            next_state = self.spot_to_state()  # post模式先返回状态后标注
        self.pred_padding[tuple(self.spot)] = 1
        if self.state_mode == 'pre':
            next_state = self.spot_to_state()  # pre模式先标注后返回状态
        self.dice = self.compute_dice()
        self.cover = self.pred_cover()
        return next_state, self.cover, (self.step_max - self.step_n) / self.step_max  # 返回下一个状态图、dice值和剩余步数

    def spot_to_state(self):
        """通过spot和给定的state图像大小计算返回相应state图像"""
        # 计算当前spot对应于padding后图像的状态图边界
        l_side = self.spot[0] - self.pad_length  # 状态图左边界
        r_side = self.spot[0] + self.pad_length + 1  # 状态图右边界
        u_side = self.spot[1] - self.pad_width  # 状态图上边界
        d_side = self.spot[1] + self.pad_width + 1  # 状态图下边界
        f_side = self.spot[2] - self.pad_depth  # 状态图前边界
        b_side = self.spot[2] + self.pad_depth + 1  # 状态图后边界
        # 取状态图并融合
        image_state = self.image_padding[l_side:r_side, u_side:d_side, f_side:b_side]
        pred_state = self.pred_padding[l_side:r_side, u_side:d_side, f_side:b_side]
        prob_state = self.prob_padding[l_side:r_side, u_side:d_side, f_side:b_side]
        if self.state_channel == 3:
            state = torch.stack((image_state, prob_state, pred_state), dim=0)
        elif self.state_channel == 2:
            state = torch.stack((image_state, pred_state), dim=0)
        else:
            print("state_channel is error!")
        return np.array(state.cpu())  # 将状态图转换为numpy格式

    def step(self, action):
        """智能体完成一个动作，并返回下一个状态、奖励和完成情况"""
        change = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        self.spot = [x + y for x, y in zip(self.spot, change[action])]  # 将动作叠加到位置
        self.step_n += 1  # 步数累积
        done = False  # 默认为未完成
        # 如果超出边界
        if self.spot_is_out():
            if self.out_mode:
                done = True
            next_state = self.spot_to_state()  # 超出边界时状态图不移动
            # 根据不同模式计算超出边界时的reward
            if self.out_reward_mode == 'small':
                reward = -1  # 超出边界时奖励为-1
            elif self.out_reward_mode == 'large':
                reward = -100  # 超出边界时奖励为-100
            elif self.out_reward_mode == '0':
                reward = 0  # 超出边界时奖励为0
        # 未超出边界时
        else:
            if self.state_mode == 'post':
                next_state = self.spot_to_state()  # post模式先返回状态后标注
            # 根据不同模式计算reward
            if self.reward_mode == 'const':
                # 计算reward
                reward = self.spot_to_const_reward()
            # 在预测图像中记录标注路径
            self.pred_padding[tuple(self.spot)] = 1
            dice_new = self.compute_dice()
            if self.reward_mode == 'dice_inc':
                # 计算dice值增量
                dice_inc = dice_new - self.dice
                reward = dice_inc * 100  # 奖励为dice值增量的100倍
            if self.reward_mode == 'dice_inc_const':
                dice_inc = dice_new - self.dice
                reward = dice_inc * self.mask_spots_n  # 奖励为dice值增量乘上标注点数量
            self.dice = dice_new
            if self.state_mode == 'pre':
                next_state = self.spot_to_state()  # pre模式先标注后返回状态
        cover_new = self.pred_cover()
        cover_inc = cover_new - self.cover
        self.cover = cover_new
        # 判断是否达到步数限制条件
        if cover_inc > 0:
            self.step_limit_n = 0  # 清空无新标注覆盖步数
        else:
            self.step_limit_n += 1  # 无新标注覆盖步数累积
        if self.step_limit_n >= self.step_limit_max:
            done = True
        if self.step_n >= self.step_max:
            done = True
        return next_state, reward, done, self.cover, (self.step_max - self.step_n) / self.step_max

    def spot_is_out(self):
        """判断当前spot是否超出padding前的实际边界，超出边界的点赋值为边界值"""
        length = self.pred_padding.shape[0]
        width = self.pred_padding.shape[1]
        depth = self.pred_padding.shape[2]
        if self.spot[0] < self.pad_length:
            self.spot[0] = self.pad_length
            return True
        elif self.spot[0] > length - self.pad_length - 1:
            self.spot[0] = length - self.pad_length - 1
            return True
        elif self.spot[1] < self.pad_width:
            self.spot[1] = self.pad_width
            return True
        elif self.spot[1] > width - self.pad_width - 1:
            self.spot[1] = width - self.pad_width - 1
            return True
        elif self.spot[2] < self.pad_depth:
            self.spot[2] = self.pad_depth
            return True
        elif self.spot[2] > depth - self.pad_depth - 1:
            self.spot[2] = depth - self.pad_depth - 1
            return True
        else:
            return False  # 未超出边界则返回False

    def compute_dice(self):
        """返回当前预测图像和标注图像dice值"""
        return (2 * (self.pred_padding & self.mask_padding).sum() /
                (self.pred_padding.sum() + self.mask_padding.sum())).item()

    def pred_cover(self):
        """计算当前预测图像对已预测图像的覆盖值dice"""
        return (2 * (self.pred_padding & self.preded_padding).sum() /
                (self.pred_padding.sum() + self.preded_padding.sum())).item()

    def spot_to_const_reward(self):
        """reward模式为const时，通过当前spot得到对应的reward"""
        if self.pred_padding[tuple(self.spot)] == 1:  # 重复时奖励为0
            reward = 0
        elif self.mask_padding[tuple(self.spot)] == 1:  # 不重复且涂对时奖励为1
            reward = 1
        else:
            reward = -1  # 未涂对时奖励为-1
        return reward

    def to_pred(self):
        """根据pred_padding输出预测结果pred"""
        pred = self.pred_padding[self.pad_length:-self.pad_length,
                                 self.pad_width:-self.pad_width,
                                 self.pad_depth:-self.pad_depth]
        return pred