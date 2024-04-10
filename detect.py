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

image_path = "D:/Shravtek/neutech_airfillter/anamolib/anomalib/datasets/bottle/good/008.png"
image = read_image(path=image_path)
output_path = "D:/Shravtek/neutech_airfillter/anamolib/Models/bottle/epoch_3_works_good"

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

print(type(predictions.pred_label))
if predictions.pred_label=="ABNORMAL":
    print("ABNORMAL")
    cv2.putText(image,str("NOK"), (100, 100), cv2.FONT_HERSHEY_SIMPLEX,2,(0, 0, 255), 3)
else:
    print("NORMAL")
    cv2.putText(image,str("OK"), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0), 3)

cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()