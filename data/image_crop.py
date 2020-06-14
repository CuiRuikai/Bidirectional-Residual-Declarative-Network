import os
import sys
import pandas as pd

sys.path.append("../")

from faced.detector import FaceDetector
from utils.preprocess_tools import mkdir
from utils.preprocess_tools import crop_image

face_detector = FaceDetector()
path_to_source = "dataset/raw_data/"
path_to_target = "dataset/dataset1/"

folders = os.listdir(path_to_source)
mkdir(path_to_target)
meta = {'no': [], 'filename': [], 'prob': [], 'emotion': []}

no = 0
for folder in folders:
    mkdir(path_to_target + folder)
    file = os.listdir(path_to_source + folder)
    for img_name in file:
        origin = path_to_source + folder + '/' + img_name
        target = path_to_target + folder + '/' + img_name
        crop_image(face_detector, origin, target, img_name, meta, no, folder, width=128)
        no += 1
    print("Finish " + folder)

df = pd.DataFrame(meta)
df.to_csv(path_to_target + 'meta.csv')
print("done")
