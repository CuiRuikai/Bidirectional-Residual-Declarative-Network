import sys

sys.path.append("../")

from utils.preprocess_tools import image_filp
from utils.preprocess_tools import image_ressize

path = 'dataset/dataset1/train'

image_filp(path)
print("---------Flip Finish---------")
image_ressize(path)
print("---------Resize Finish---------")
