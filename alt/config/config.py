from dataclasses import asdict, dataclass, field
from typing import List, Dict


@dataclass(frozen=False, init=True)
class GlobalConfig:
    pass


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
