from torch import nn
from timm.models import efficientnet

from my_config import Config
from src.model_lib.MiniFASNet import MiniFASNetV2SE
from src.model_lib.MultiFTNet import FTGenerator


class Baseline(nn.Module):
    def __init__(self, conf: Config):
        super().__init__()
        self.model = MiniFASNetV2SE(embedding_size=conf.embedding_size,
                                    conv6_kernel=conf.kernel_size,
                                    num_classes=conf.num_classes,
                                    img_channel=conf.input_channel)
        self.ft_generator = FTGenerator(in_channels=128)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2_dw(x)
        x = self.model.conv_23(x)
        x = self.model.conv_3(x)
        x = self.model.conv_34(x)
        x = self.model.conv_4(x)
        x1 = self.model.conv_45(x)
        x1 = self.model.conv_5(x1)
        x1 = self.model.conv_6_sep(x1)
        x1 = self.model.conv_6_dw(x1)
        x1 = self.model.conv_6_flatten(x1)
        x1 = self.model.linear(x1)
        x1 = self.model.bn(x1)
        x1 = self.model.drop(x1)
        cls = self.model.prob(x1)

        if self.training:
            ft = self.ft_generator(x)
            return cls, ft
        else:
            return cls


class FASNet(Baseline):
    def __init__(self, conf: Config):
        super().__init__(conf)
        self.model = efficientnet.efficientnet_lite0(pretrained=conf.imagenet_pretrained)
        self.ft_generator = FTGenerator(in_channels=40)
        self.cls_head = nn.Linear(in_features=1280, out_features=conf.num_classes, bias=True)

    def forward(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.blocks[0](x)
        x = self.model.blocks[1](x)
        x = self.model.blocks[2](x)
        x1 = self.model.blocks[3](x)
        x1 = self.model.blocks[4](x1)
        x1 = self.model.blocks[5](x1)
        x1 = self.model.blocks[6](x1)
        x1 = self.model.conv_head(x1)
        x1 = self.model.bn2(x1)
        x1 = self.model.global_pool(x1)
        cls = self.cls_head(x1)

        if self.training:
            ft = self.ft_generator(x)
            return cls, ft
        else:
            return cls


if __name__ == '__main__':
    import torch
    from my_config import get_default_config

    im = torch.rand((1, 3, 384, 384), dtype=torch.float32)
    model = Baseline(get_default_config())
    model.eval()
    print(model(im))
