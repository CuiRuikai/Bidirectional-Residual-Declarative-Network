# tools for BDNN training

import torch

import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

import sys

sys.path.append(".")
from utils.model_modify import set_fc_output_class
from utils.model_modify import add_projection


def load_model(args):
    checkpoint = torch.load(args.checkpoint_path)
    print('model info')
    print('epoch: ', checkpoint['epoch'])
    print('best_acc1: ', checkpoint['best_acc1'])

    model = models.__dict__[checkpoint['arch']]()
    model.fc = set_fc_output_class(model)  # change the output class to 7
    model.fc = add_projection(args.projection_type, args.radius, model)
    model.load_state_dict(checkpoint['state_dict'])
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    return model


def to_one_hot(y, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims.
    author；justheuristic
    url: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24
    """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def get_fc_inputs(pre_model, images, projection_type):
    with torch.no_grad():
        # switch to evaluate mode
        x = pre_model.conv1(images)
        x = pre_model.bn1(x)
        x = F.relu(x)
        x = pre_model.maxpool(x)

        x = pre_model.layer1(x)
        x = pre_model.layer2(x)
        x = pre_model.layer3(x)
        x = pre_model.layer4(x)

        x = pre_model.avgpool(x)
        x = torch.flatten(x, 1)
        if projection_type in ('L1S', 'L1B', 'L2S', 'L2B', 'LInfS', 'LInfB'):
            x = pre_model.fc[0](x)
            x = pre_model.fc[1](x)
        return x
