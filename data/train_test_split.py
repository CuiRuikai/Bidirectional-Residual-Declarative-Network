import sys

sys.path.append("../")
from utils.preprocess_tools import train_test_split

source_path = 'dataset/raw_data/'
target_path = 'dataset/dataset1/'
ratio = 0.25

train_test_split(source_path, target_path, ratio)
print("---------Train/Test Split Finish---------")
