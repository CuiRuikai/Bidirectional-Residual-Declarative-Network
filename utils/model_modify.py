# tools for modify the Pytorch resnet structure

import torch.nn as nn

import sys

sys.path.append("../")
import ddn.pytorch.projections as projections


def set_fc_output_class(model):
    fc_features = model.fc.in_features
    fc = nn.Linear(fc_features, 7)
    return fc


def add_projection(projection_type, radius, model):
    # Insert BN layer and Lp-sphere or Lp-ball projection layer before FC layer
    if projection_type == 'L1S':
        method = projections.L1Sphere
    elif projection_type == 'L1B':
        method = projections.L1Ball
    elif projection_type == 'L2S':
        method = projections.L2Sphere
    elif projection_type == 'L2B':
        method = projections.L2Ball
    elif projection_type == 'LInfS':
        method = projections.LInfSphere
    elif projection_type == 'LInfB':
        method = projections.LInfBall
    else:
        method = None
    if method:
        print('Prepending FC layer with a BN layer and an {} projection layer with radius {}'.format(method.__name__,
                                                                                                     radius))
        batchnorm = nn.BatchNorm1d(model.fc.in_features, eps=1e-05, momentum=0.1,
                                   affine=False)  # without learnable parameters
        projection = projections.EuclideanProjection(method=method, radius=radius)
        return nn.Sequential(batchnorm, projection, model.fc)
    else:
        return model.fc
