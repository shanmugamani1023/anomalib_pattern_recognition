from typing import Any
from torchvision import transforms
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import ToPILImage
import cv2
from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.data.utils import read_image
from anomalib.deploy import ExportType, OpenVINOInferencer
from anomalib.engine import Engine
from anomalib.models import Padim
import os
from pathlib import Path

image_path = "datasets/hazelnut_toy/good/01.jpg"
image = read_image(path=image_path)
output_path = "D:/Shravtek/neutech_airfillter/anamolib/anomalib/results/Padim/hazelnut_toy/latest"

print(output_path)
openvino_model_path = output_path+"/weights"+"/openvino"+"/model.bin"
metadata = output_path+"/weights"+"/openvino"+"/metadata.json"
# print(openvino_model_path.exists(), metadata.exists())
inferencer = OpenVINOInferencer(
    path=openvino_model_path,  # Path to the OpenVINO IR model.
    metadata=metadata,  # Path to the metadata file.
    device="CPU",  # We would like to run it on an Intel CPU.
)
predictions = inferencer.predict(image=image_path)
print(predictions.pred_score, predictions.pred_label)
cv2.imshow("seg_image",predictions.segmentations)

cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()