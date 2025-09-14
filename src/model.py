"""
Implementation of Yolo (v1) architecture with slight modification with added BatchNorm.
"""


from typing import List

import torch
import torchinfo
from torch import nn


class CNNBlock(nn.Module):
    """Creates CNNBlock similar to YOLOv1 Darknet architecture

    Note:
        1. On top of `nn.Conv2d` we add `nn.BatchNorm2d` and `nn.LeakyReLU`.
        2. We set `track_running_stats=False` in `nn.BatchNorm2d` because we want
           to avoid updating running mean and variance during training.
           ref: https://tinyurl.com/ap22f8nf
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        """Initialize CNNBlock.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            **kwargs (Dict[Any]): Keyword arguments for `nn.Conv2d` such as `kernel_size`,
                     `stride` and `padding`.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(
            num_features=out_channels, track_running_stats=False
        )
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1Darknet(nn.Module):
    def __init__(
        self,
        architecture: List,
        in_channels: int = 3,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        init_weights: bool = False,
    ) -> None:
        """Initialize Yolov1Darknet.

        Note:
            1. `self.backbone` is the backbone of Darknet.
            2. `self.head` is the head of Darknet.
            3. Currently the head is hardcoded to have 1024 neurons and if you change
               the image size from the default 448, then you will have to change the
               neurons in the head.

        Args:
            architecture (List): The architecture of Darknet. See config.py for more details.
            in_channels (int): The in_channels. Defaults to 3 as we expect RGB images.
            S (int): Grid Size. Defaults to 7.
            B (int): Number of Bounding Boxes to predict. Defaults to 2.
            C (int): Number of Classes. Defaults to 20.
            init_weights (bool): Whether to init weights. Defaults to False.

        Reference:
            Aladdin's repo: https://github.com/aladdinpersson/Machine-Learning-Collection
        """
        super().__init__()

        self.architecture = architecture
        self.in_channels = in_channels
        self.S = S
        self.B = B
        self.C = C

        # backbone is darknet
        self.backbone = self._create_darknet_backbone()
        self.head = self._create_darknet_head()

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for Conv2d, BatchNorm2d, and Linear layers."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.backbone(x)
        x = self.head(torch.flatten(x, start_dim=1))
        x = x.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # if self.squash_type == "flatten":
        #     x = torch.flatten(x, start_dim=1)
        # elif self.squash_type == "3D":
        #     x = x.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # elif self.squash_type == "2D":
        #     x = x.reshape(-1, self.S * self.S, self.C + self.B * 5)
        return x

    def _create_darknet_backbone(self) -> nn.Sequential:
        """Create Darknet backbone."""
        layers = []
        in_channels = self.in_channels

        for layer_config in self.architecture:
            # convolutional layer
            if isinstance(layer_config, tuple):
                out_channels, kernel_size, stride, padding = layer_config
                layers += [
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                ]
                # update next layer's in_channels to be current layer's out_channels
                in_channels = layer_config[0]

            # max pooling
            elif isinstance(layer_config, str) and layer_config == "M":
                # hardcode maxpooling layer
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif isinstance(layer_config, list):
                conv1 = layer_config[0]
                conv2 = layer_config[1]
                num_repeats = layer_config[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            out_channels=conv1[0],
                            kernel_size=conv1[1],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            in_channels=conv1[0],
                            out_channels=conv2[0],
                            kernel_size=conv2[1],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[0]

        return nn.Sequential(*layers)

    def _create_darknet_head(self) -> nn.Sequential:
        """Create the fully connected layers of Darknet head.

        Note:
            1. In original paper this should be
                nn.Sequential(
                    nn.Linear(1024*S*S, 4096),
                    nn.LeakyReLU(0.1),
                    nn.Linear(4096, S*S*(B*5+C))
                    )
            2. You can add `nn.Sigmoid` to the last layer to stabilize training
               and avoid exploding gradients with high loss since sigmoid will
               force your values to be between 0 and 1. Remember if you do not put
               this your predictions can be unbounded and contain negative numbers even.
        """

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5)),
            # nn.Sigmoid(),
        )


if __name__ == "__main__":
    batch_size = 4
    image_size = 448
    in_channels = 3
    S = 7
    B = 2
    C = 20

    DARKNET_ARCHITECTURE = [
        (64, 7, 2, 3),
        "M",
        (192, 3, 1, 1),
        "M",
        (128, 1, 1, 0),
        (256, 3, 1, 1),
        (256, 1, 1, 0),
        (512, 3, 1, 1),
        "M",
        [(256, 1, 1, 0), (512, 3, 1, 1), 4],
        (512, 1, 1, 0),
        (1024, 3, 1, 1),
        "M",
        [(512, 1, 1, 0), (1024, 3, 1, 1), 2],
        (1024, 3, 1, 1),
        (1024, 3, 2, 1),
        (1024, 3, 1, 1),
        (1024, 3, 1, 1),
    ]

    x = torch.zeros(batch_size, in_channels, image_size, image_size)
    y_trues = torch.zeros(batch_size, S, S, B * 5 + C)

    yolov1 = Yolov1Darknet(
        architecture=DARKNET_ARCHITECTURE,
        in_channels=in_channels,
        S=S,
        B=B,
        C=C,
    )

    y_preds = yolov1(x)

    print(f"x.shape: {x.shape}")
    print(f"y_trues.shape: {y_trues.shape}")
    print(f"y_preds.shape: {y_preds.shape}")
    print(f"yolov1 last layer: {yolov1.head[-1]}")

    torchinfo.summary(
        yolov1, input_size=(batch_size, in_channels, image_size, image_size)
    )
