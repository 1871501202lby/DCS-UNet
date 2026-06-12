
import torch
import torch.nn as nn

from sam2.modeling.ARConv import ARConv
from sam2.modeling.SCSA import SCSA
from sam2.modeling.Tripletattention import TripletAttention


class PFA2D(nn.Module):
    expansion = 4

    def __init__(self, inchannel, outchannel,pool_size):
        super(PFA2D, self).__init__()

        # 激活函数
        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)
        self.relu_3 = nn.ReLU(inplace=True)


        # 2D 卷积层代替 3D 卷积
        self.conv2d_1 = nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(inchannel)
        self.conv2d_2 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(outchannel)

        self.globalAvgPool = nn.AvgPool2d(pool_size, stride=1)  # 使用2D池化，保持空间特征
        self.globalMaxPool = nn.MaxPool2d(pool_size, stride=1)
        # self.globalAvgPool = nn.AdaptiveAvgPool2d(1)  # 变成自适应全局平均池化
        # self.globalMaxPool = nn.AdaptiveMaxPool2d(1)  # 变成自适应全局最大池化

        # 两个全连接层
        self.fc1 = nn.Linear(in_features=inchannel, out_features=outchannel)
        self.fc2 = nn.Linear(in_features=outchannel, out_features=inchannel)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        # self.triplet = TripletAttention(gate_channels=64)


    def forward(self, x ):
        residual = x  # 存储输入作为残差
        original_out = x
        out = x
        out1 = x

        # 对输入应用全局平均池化
        out = self.globalAvgPool(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu_1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.reshape(out.size(0), out.size(1), 1, 1)  # 变成 (batch_size, out_channels, 1, 1)
        out = out * original_out  # 与原始输入相乘

        # 对输入应用全局最大池
        out1 = self.globalMaxPool(out1)
        out1 = out1.reshape(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu_1(out1)
        out1 = self.dropout(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.reshape(out1.size(0), out1.size(1), 1, 1)
        out1 = out1 * original_out  # 与原始输入相乘

        # 将两个池化结果相加
        out += out1


        # 激活与卷积操作
        out = self.relu_2(out)
        out = self.conv2d_1(out)
        out = self.bn_1(out)
        out += residual
        out = self.relu_2(out)
        out = self.conv2d_2(out)
        out = self.bn_2(out)
        out = self.relu_3(out)
        return out

        # # #使用多尺度
        # residual = x
        # original_out = x
        # out3x3 = self.conv3x3(x)
        # out5x5 = self.conv5x5(x)
        # out7x7 = self.conv7x7(x)
        # out = torch.cat([out3x3, out5x5, out7x7], dim=1)
        # out = self.triplet(out)
        # avg_out = self.conv1x1(out)
        # max_out =self.conv1x1(out)
        # avg_out = self.globalAvgPool(avg_out)
        # avg_out = avg_out.view(avg_out.size(0), -1)
        # avg_out = self.fc1(avg_out)
        # avg_out = self.relu_1(avg_out)
        # avg_out = self.dropout(avg_out)
        # avg_out = self.fc2(avg_out)
        # avg_out = self.sigmoid(avg_out)
        # avg_out = avg_out.reshape(avg_out.size(0), avg_out.size(1), 1, 1)  # 变成 (batch_size, out_channels, 1, 1)
        # avg_out = avg_out * original_out  # 与原始输入相乘
        #
        # # 对输入应用全局最大池化
        # max_out = self.globalMaxPool(max_out)
        # max_out = max_out.reshape(max_out.size(0), -1)
        # max_out = self.fc1(max_out)
        # max_out = self.relu_1(max_out)
        # max_out = self.dropout(max_out)
        # max_out = self.fc2(max_out)
        # max_out = self.sigmoid(max_out)
        # max_out = max_out.reshape(max_out.size(0), max_out.size(1), 1, 1)
        # max_out = max_out * original_out  # 与原始输入相乘
        #
        #
        # avg_out += max_out
        # # 激活与卷积操作
        # avg_out = self.relu_2(avg_out)
        # avg_out = self.conv2d_1(avg_out)
        # avg_out = self.bn_1(avg_out)
        # avg_out += residual
        # avg_out = self.relu_2(avg_out)
        #
        # avg_out = self.conv2d_2(avg_out)
        # avg_out = self.bn_2(avg_out)
        # avg_out = self.relu_3(avg_out)
        #
        # return avg_out
if __name__ == '__main__':
    # 定义输入的张量，大小为 (1, 3, 352, 352)
    input_tensor = torch.randn(1,64,22,22)  # 随机生成一个输入张量

    # 创建 PFA2D 模型实例
    model = PFA2D(64,64,(22,22))

    # 将输入传递给模型
    output = model(input_tensor)

    # 打印输出的形状
    print(f"Output shape: {output.shape}")






import torch
import torch.nn as nn
from sam2.modeling.Tripletattention import TripletAttention


class PFA2D(nn.Module):
    expansion = 4

    def __init__(self, inchannel, outchannel,pool_size):
        super(PFA2D, self).__init__()

        # 激活函数
        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)
        self.relu_3 = nn.ReLU(inplace=True)

        # #多尺度卷积
        # self.conv3x3 = nn.Conv2d(inchannel, outchannel // 3, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv5x5 = nn.Conv2d(inchannel, outchannel // 3, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv7x7 = nn.Conv2d(inchannel, outchannel // 3, kernel_size=7, stride=1, padding=3, bias=False)

        # 2D 卷积层代替 3D 卷积
        self.conv2d_1 = nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(inchannel)

        self.conv2d_2 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(outchannel)

        self.globalAvgPool = nn.AvgPool2d(pool_size, stride=1)  # 使用2D池化，保持空间特征
        self.globalMaxPool = nn.MaxPool2d(pool_size, stride=1)


        # 两个全连接层
        self.fc1 = nn.Linear(in_features=inchannel, out_features=outchannel)
        self.fc2 = nn.Linear(in_features=outchannel, out_features=inchannel)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        # self.triplet1 = TripletAttention(gate_channels=64)
        # self.triplet2 = TripletAttention(gate_channels=64)
        # self.conv1x1 = nn.Conv2d(63, 64, kernel_size=1)
        self.fuse_conv = nn.Conv2d(128, 64, kernel_size=1)

    def forward(self, x):
        residual = x  # 存储输入作为残差
        original_out = x
        out = x
        out1 = x

        # 对输入应用全局平均池化
        out = self.globalAvgPool(out)
        # print(f"Shape after global avg pool: {out.shape}")
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu_1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.reshape(out.size(0), out.size(1), 1, 1)  # 变成 (batch_size, out_channels, 1, 1)
        out = out * original_out  # 与原始输入相乘

        # 对输入应用全局最大池化
        out1 = self.globalMaxPool(out1)
        out1 = out1.reshape(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu_1(out1)
        out1 = self.dropout(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.reshape(out1.size(0), out1.size(1), 1, 1)
        out1 = out1 * original_out  # 与原始输入相乘

        # 将两个池化结果相加
        out += out1


        # 激活与卷积操作
        out = self.relu_2(out)
        out = self.conv2d_1(out)
        out = self.bn_1(out)
        # out = self.triplet1(out)
        out += residual
        out = self.relu_2(out)

        out = self.conv2d_2(out)
        out = self.bn_2(out)
        # out = self.triplet2(out)
        out = self.relu_3(out)
        return out

if __name__ == '__main__':
    # 定义输入的张量，大小为 (1, 3, 352, 352)
    input_tensor = torch.randn(1,64,11,11)  # 随机生成一个输入张量

    # 创建 PFA2D 模型实例
    model = PFA2D(64,64,(11,11))

    # 将输入传递给模型
    output = model(input_tensor)

    # 打印输出的形状
    print(f"Output shape: {output.shape}")










