"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.

Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding)
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

import torch
from torch import nn
import torchinfo
from typing import List, Optional

DEFAULT_ARCHITECTURE = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # https://tinyurl.com/ap22f8nf on set track_running_stats to False
        self.batchnorm = nn.BatchNorm2d(
            out_channels, track_running_stats=False
        )
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.leakyrelu(self.batchnorm(self.conv(x)))


DEFAULT_ARCHITECTURE = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # https://tinyurl.com/ap22f8nf on set track_running_stats to False
        self.batchnorm = nn.BatchNorm2d(
            out_channels, track_running_stats=False
        )
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1Darknet(nn.Module):
    def __init__(
        self,
        architecture: Optional[List] = None,
        in_channels: int = 3,
        grid_size: int = 7,
        num_bboxes_per_grid: int = 2,
        num_classes: int = 20,
        init_weights: bool = False,
    ):
        super().__init__()

        self.architecture = architecture
        self.in_channels = in_channels
        self.S = grid_size
        self.B = num_bboxes_per_grid
        self.C = num_classes

        if self.architecture is None:
            self.architecture = DEFAULT_ARCHITECTURE

        # backbone is darknet
        self.backbone = self._create_conv_layers(self.architecture)
        self.head = self._create_fcs(self.S, self.B, self.C)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, tuple):
                layers += [
                    CNNBlock(
                        in_channels,
                        x[1],
                        kernel_size=x[0],
                        stride=x[2],
                        padding=x[3],
                    )
                ]
                in_channels = x[1]

            # max pooling
            elif isinstance(x, str) and x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif isinstance(x, list):
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    @staticmethod
    def _create_fcs(S: int, B: int, C: int) -> torch.nn.Sequential:
        """Create the fully connected layers.

        Note:
        In original paper this should be
            nn.Linear(1024*S*S, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S*S*(B*5+C))
        """

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)),
            # nn.Sigmoid(),  # This is not in the original implementation but added to avoid loss explosion - 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        )


if __name__ == "__main__":
    # 自定义输入张量，验证网络可以正常跑通，并计算loss，调试用
    batch_size = 16
    image_size = 448
    in_channels = 3
    S = 7
    B = 2
    C = 20

    x = torch.zeros(batch_size, in_channels, image_size, image_size)
    y_trues = torch.zeros(batch_size, S, S, B * 5 + C)
    yolov1 = Yolov1Darknet(
        in_channels=in_channels,
        grid_size=S,
        num_bboxes_per_grid=B,
        num_classes=C,
    )
    y_preds = yolov1(x)
    assert (
        y_preds.shape
        == (batch_size, 7 * 7 * (20 + 2 * 5))
        == (batch_size, S * S * (C + B * 5))
    )
    print(y_preds.shape)

    torchinfo.summary(
        yolov1, input_size=(batch_size, in_channels, image_size, image_size)
    )
