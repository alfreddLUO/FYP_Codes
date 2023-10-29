from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 如果没有传入norm_layer，默认使用BN
        if activation_layer is None:
            activation_layer = nn.ReLU6  # 如果没有传入activation_layer，默认使用ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer(inplace=True))


# SE模块，即注意力机制模块
class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        # 第1个FC层输出结点个数为输入特征矩阵channel的1/4，这里设置squeeze_factor: int = 4
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)  # 计算第1个FC层的节点个数
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)  # 直接用卷积Conv2d作为全连接层，FC1
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)  # 因为FC2输出channel与输入的channel相同，因此输出参数设置为input_c

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # 自适应平均池化，output_size=(1,1)这样可以将每个channel平均池化到1×1大小
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x  # 将每个权重数据与原来的特征矩阵相乘


# 这里的InvertedResidualConfig对应的是MobileNetV3中的每一个bneck结构的参数配置
class InvertedResidualConfig:
    def __init__(self,
                 input_c: int,  # 输入特征矩阵的channel
                 kernel: int,  # DW卷积所对应的卷积核大小
                 expanded_c: int,  # exp size是第1层1×1卷积层所使用的卷积核个数
                 out_c: int,  # 最后1层1×1卷积层所使用的卷积核大小
                 use_se: bool,  # 是否使用SE模块
                 activation: str,  # 激活函数，RE对应relu，HS对应h-swish
                 stride: int,  # DW卷积对应的步距
                 width_multi: float  # 对应V2中的α参数，用来调节每一个卷积层所使用channel的倍率因子
                 ):
        self.input_c = self.adjust_channels(input_c, width_multi)  # 用adjust_channels得到调节后的输入特征矩阵channel
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)  # 用adjust_channels得到调节后的expand channel
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # whether using h-swish activation
        # 如果activation == "HS"，则use_hs=True，使用h-swish激活函数。如果使用RE，则use_hs=False。
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        # 初始化函数中出入cnf文件，即上面的InvertedResidualConfig
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:  # 参数表中，步距只有1和2两种情况。如果不等于1或2，则为非法的。
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)  # 判断是否有short cut分支

        layers: List[nn.Module] = []  # 创建1个空列表layers
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU  # 判断使用哪个激活函数

        # expand，第1个1×1卷积层
        if cnf.expanded_c != cnf.input_c:  # 对应表格第2行，exp size=input c时，没有第1层1×1卷积层
            layers.append(ConvBNActivation(cnf.input_c,
                                           cnf.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        # depthwise,DW卷积
        layers.append(ConvBNActivation(cnf.expanded_c,  # 输入特征矩阵channel为上1层输出特征矩阵的channel
                                       cnf.expanded_c,  # DW卷积input c = output c
                                       kernel_size=cnf.kernel,
                                       stride=cnf.stride,
                                       groups=cnf.expanded_c,  # DW卷积针对每一个channel单独处理，groups数和channel数保持一致
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))

        if cnf.use_se:  # 接下来判断当前层结构是否使用SE注意力机制
            layers.append(SqueezeExcitation(cnf.expanded_c))  # SE模块只需要传入1个参数，即input channel，这里是exoanded_c(通过DW输出的c)

        # project，最后1个1×1的卷积层
        layers.append(ConvBNActivation(cnf.expanded_c,  # 无论是否使用SE模块，最后一层卷积的input_c都等于DW卷积后的output_c
                                       cnf.out_c,  # 输出特征矩阵的channel为配置文件中给定的#out
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Identity))  # 最后1层卷积的激活函数为线性激活，即没有做任何处理

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x  # 如果使用了short cut连接，主分支block输出与原始x相加

        return result


class MobileNetV3(nn.Module):
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],  # 对应一些列bneck参数的列表
                 last_channel: int,  # 对应参数表中倒数第2个卷积层(也是FC层)输出结点的个数
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,  # block对应上面定义的更新倒残差结构，默认设置为None
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:  # 如果没有传入bneck参数，会报错
            raise ValueError("The inverted_residual_setting should not be empty.")
        # 下面进行数据检查，如果传入参数不是列表或列表中的参数不是InvertedResidualConfig的参数时，也会报错
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:  # 将block默认设置为InvertedResidual
            block = InvertedResidual

        if norm_layer is None:  # 将norm_layer默认设置为BN
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
            # partial为BatchNorm2d方法默认传入参数eps=0.001,momentum=0.01

        layers: List[nn.Module] = []  # 创建一个空列表layers

        # building first layer
        firstconv_output_c = inverted_residual_setting[0].input_c  # 获取第1个卷积层输出的channel，它对应着第1个bneck的input channel
        # 定义第1个卷积层，对应参数列表第1行
        layers.append(ConvBNActivation(3,  # 使用rgb图像，故输入channel为3
                                       firstconv_output_c,  # 输出channel为下面第2行对应的第1个bneck的input_c
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,  # 将上面定义好的BN结构赋给norm_layer
                                       activation_layer=nn.Hardswish))  # 第1层使用h-swish函数
        # building inverted residual blocks
        for cnf in inverted_residual_setting:  # 遍历每1个bneck结构，将配置文件和norm_layer传给block，并将block添加到layers中
            layers.append(block(cnf, norm_layer))

        # 构建最后几个层结构，包括卷积、池化和全连接层
        # building last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_c  # 获取最后1个bneck结构的output_c,它是下一个卷积层的input_c
        lastconv_output_c = 6 * lastconv_input_c  # 根据参数列表中倒数第4行数据，该层卷积层的out_c=6*input_c
        # 定义参数列表倒数第4行的卷积层，将其添加到layers[]中
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)  # 将以上所有的层结构(第1行到倒数第3行)传入作为特征提取网络
        # 接下来定义分类网络，主要包括平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        # 构建最后啷个全连接层
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        # 输入c等于上面计算所得，输出last_channel为初始化中传入参数
                                        nn.Hardswish(inplace=True),  # 使用HS激活函数
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))  # 输入节点个数为上1个FC层输出的节点个数，输出节点个数为分类类别个数

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)  # 经过平均池化后，高和宽都变成1×1了
        x = torch.flatten(x, 1)  # 不在需要高和宽维度，展平成一维
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v3_large(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0  # 设置1个超参数α
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)  # 给InvertedResidualConfig传入默认超参数α=1
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)
    # 给adjust_channels方法传入超参数

    reduce_divider = 2 if reduced_tail else 1
    # pytorch官方给定的参数，改变最后3个bneck的channel数目，默认不使用。如果想进一步减少网络参数，可以设置其为True

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4，如果将reduced_tail设置为ture，这里就会进一步调整网络
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5，默认情况下，该值等于1280，即倒数第2行全连接层的节点个数

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.
    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)

