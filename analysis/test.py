import argparse
import pandas as pd

import torch.utils.data
import torchvision.transforms as transforms
from sklearn.metrics import classification_report

import sys

sys.path.append("../")
from utils.eval_tools import *
from utils.training_tools import load_model
from utils.analysis_tools import *
from utils.ImageFolderWithPaths import ImageFolderWithPaths


torch.manual_seed(2809)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training (Modified)')

# please change the default path to the checkpoint path
data_dir = "../data/dataset/dataset1/val" # the path refers to the path of test set or val set

parser.add_argument('--checkpoint_path', default='checkpoint.pth.tar', type=str, metavar='N',
                    help='path of checkpoint')
parser.add_argument('--projection-type', dest='projection_type', type=str, default='LInfS',
                    help="Euclidean projection type {L1S, L1B, L2S, L2B, LInfS, LInfB, ''}")
parser.add_argument('--radius', default=1.0, type=float,
                    help='Lp-sphere or ball radius')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total')

args = parser.parse_args()

model = load_model(args)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

dataloader = torch.utils.data.DataLoader(
    ImageFolderWithPaths(data_dir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=256, shuffle=False, pin_memory=True)



shows_some_images_with_path(dataloader, 4)

(outputs, targets, paths)= model_predict_with_path(dataloader, model, topk=1)

print(accuracy(outputs, targets, topk=(1, 3)))

y_true = targets.numpy()
maxk = max((1, 0))
_, pred = outputs.topk(maxk, 1, True, True)
print(classification_report(y_true, pred, target_names=classes, digits=4))

torch.save(outputs, 'outputs.pt')
torch.save(targets, 'targets.pt')
pd.DataFrame(paths).to_csv('paths.csv')