# This is the implementation of BDNN
# But this class is not used in our Framework
# As we applied a simpler approach, which is use the Sequential Class of Pytorch


import torch
import torch.nn as nn

# # Three-Layer BDNN
# class BDNN(object):
#     """
#     Class for BDNN
#     paper； https://ieeexplore.ieee.org/abstract/document/487348
#     """
#
#     def __init__(self, args):
#         self.forward_net = nn.Sequential(nn.Linear(args.inputs, args.hiddens, bias=True),
#                                          nn.Linear(args.hiddens, args.outputs, bias=True))
#         self.backward_net = nn.Sequential(nn.Linear(args.outputs, args.hiddens, bias=True),
#                                           nn.Linear(args.hiddens, args.inputs, bias=True))
#
#         self.criterion1 = nn.CrossEntropyLoss()
#         self.optimizer1 = torch.optim.SGD(self.forward_net.parameters(), lr=args.lr, momentum=args.momentum,
#                                           weight_decay=args.weight_decay)
#
#         self.criterion2 = nn.MSELoss()
#         self.optimizer2 = torch.optim.SGD(self.backward_net.parameters(), lr=args.lr, momentum=args.momentum,
#                                           weight_decay=args.weight_decay)
#         if args.gpu is not None:
#             torch.cuda.set_device(args.gpu)
#             self.forward_net = self.forward_net.cuda(args.gpu)
#             self.backward_net = self.backward_net.cuda(args.gpu)
#
#     def weights_backward_to_forward(self):
#         self.forward_net[0].weight.data=self.backward_net[1].weight.T.data
#         self.forward_net[1].weight.data=self.backward_net[0].weight.T.data
#
#     def weights_forward_to_backward(self):
#         self.backward_net[0].weight.data=self.forward_net[1].weight.T.data
#         self.backward_net[1].weight.data=self.forward_net[0].weight.T.data

# Two-Layer BDNN

class BDNN(object):
    """
    Class for BDNN
    paper； https://ieeexplore.ieee.org/abstract/document/487348
    """

    def __init__(self, args):
        self.forward_net = nn.Sequential(nn.Linear(args.inputs, args.outputs, bias=True))
        self.backward_net = nn.Sequential(nn.Linear(args.outputs, args.inputs, bias=True))

        self.criterion1 = nn.CrossEntropyLoss()
        self.optimizer1 = torch.optim.SGD(self.forward_net.parameters(), lr=args.lr, momentum=args.momentum,
                                          weight_decay=args.weight_decay)

        self.criterion2 = nn.MSELoss()
        self.optimizer2 = torch.optim.SGD(self.backward_net.parameters(), lr=args.lr/10, momentum=args.momentum,
                                          weight_decay=args.weight_decay)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            self.forward_net = self.forward_net.cuda(args.gpu)
            self.backward_net = self.backward_net.cuda(args.gpu)

    def weights_backward_to_forward(self):
        self.forward_net[0].weight.data = self.backward_net[0].weight.T.data

    def weights_forward_to_backward(self):
        self.backward_net[0].weight.data = self.forward_net[0].weight.T.data
    def assign_backward_weights(self,model,args):
        if args.projection_type in ('L1S', 'L1B', 'L2S', 'L2B', 'LInfS', 'LInfB'):
            self.backward_net[0].weight.data=model.fc[2].weight.T.data
        else:
            self.backward_net[0].weight.data = model.fc.weight.T.data
