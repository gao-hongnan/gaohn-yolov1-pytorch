from dataclasses import asdict, dataclass, field
from typing import List, Dict


@dataclass(frozen=False, init=True)
class GlobalConfig:
    pass


@dataclass(frozen=False, init=True)
class ModelConfig:
    """Information about architecture configuration of model.

    Note:
        - Tuple is structured by (out_channels, kernel_size, stride, padding);
            - (64, 7, 2, 3) means 64 output channels, kernel size of 7, stride of 2, padding of 3
              for that particular layer.
            - A note is that the in_channels of the first layer is 3 (RGB) and subsequent layers'
              in_channels is the out_channels of the previous layer.
        - "M" means maxpooling with kernel size of 2 and stride of 2.
        - List is structured by (conv1, conv2, num_repeats);
            - conv1 and conv2 are tuples structured by (out_channels, kernel_size, stride, padding);
            - num_repeats is the number of times to repeat the conv1 and conv2 layers.
            - [(256, 1, 1, 0), (512, 3, 1, 1), 4] means 4 repeats of the following layers:
                - 256 output channels, kernel size of 1, stride of 1, padding of 0
                - 512 output channels, kernel size of 3, stride of 1, padding of 1
    """

    architecture: List = field(
        default_factory=lambda: [
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
    )


@dataclass(frozen=False, init=True)
class ClassMap:
    classes: List[str] = field(
        default_factory=lambda: [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
    )
    classes_map: Dict[str, int] = field(init=False)

    def __post_init__(self):
        self.classes_map = {v: k for v, k in enumerate(self.classes)}
