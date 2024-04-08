from torchvision import transforms
from PIL import Image

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import ToPILImage

from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.data.utils import read_image
from anomalib.deploy import ExportType, OpenVINOInferencer
from anomalib.engine import Engine
from anomalib.models import Padim
import os
from pathlib import Path

from anomalib.data import Folder

# Create the datamodule
datamodule = Folder(
    name="hazelnut_toy",
    root="anomalib/datasets/hazelnut_toy",
    normal_dir="good",
    abnormal_dir="crack",
    task="classification",
    image_size=(512,512)
  )

# Setup the datamodule
datamodule.setup()

i, train_data = next(enumerate(datamodule.train_dataloader()))
print(train_data.keys())
# dict_keys(['image_path', 'label', 'image'])

i, val_data = next(enumerate(datamodule.val_dataloader()))
print(val_data.keys())
# dict_keys(['image_path', 'label', 'image'])

i, test_data = next(enumerate(datamodule.test_dataloader()))
print(test_data.keys())
# dict_keys(['image_path', 'label', 'image'])