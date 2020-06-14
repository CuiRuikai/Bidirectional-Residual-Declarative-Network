# This tools are modified from
# https://github.com/anucvml/ddn/blob/master/apps/classification/image/main.py
# developed by
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Stephen Gould <stephen.gould@anu.edu.au>

import shutil
import time
import numpy as np

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import sys

sys.path.append(".")
from utils.training_tools import get_fc_inputs


def validate_bdnn(val_loader, model, pre_model, record, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top3],
        prefix='Test: ')

    # switch to evaluate mode
    model.forward_net.eval()

    with torch.no_grad():
        outputs = torch.empty(0)
        targets = torch.empty(0, dtype=torch.long)
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            inputs = get_fc_inputs(pre_model, images, args.projection_type)
            output = model.forward_net(inputs)
            loss = model.criterion1(output, target)

            # measure accuracy and record loss
            acc1, acc3 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top3.update(acc3[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            outputs = torch.cat((outputs, output.cpu()), dim=0)
            targets = torch.cat((targets, target.cpu()), dim=0)

        map, aps = mean_average_precision(outputs, targets)

        # TODO: this should also be done with the ProgressMeter
        print('* loss {loss:.4f} Acc@1 {top1.avg:.4f} Acc@3 {top3.avg:.4f} mAP {map:.4f}'
              .format(loss=loss.item(), top1=top1, top3=top3, map=map))
        record['loss'].append(losses.avg)
        record['acc@1'].append(float(top1.avg))
        record['acc@3'].append(float(top3.avg))
        record['mAP'].append(map)

    return top1.avg


def validate(val_loader, model, criterion, epoch, record, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top3],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        outputs = torch.empty(0)
        targets = torch.empty(0, dtype=torch.long)
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc3 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top3.update(acc3[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            outputs = torch.cat((outputs, output.cpu()), dim=0)
            targets = torch.cat((targets, target.cpu()), dim=0)

        map, aps = mean_average_precision(outputs, targets)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.4f} Acc@3 {top3.avg:.4f} mAP {map:.4f}'
              .format(top1=top1, top3=top3, map=map))
        record['loss'].append(losses.avg)
        record['acc@1'].append(float(top1.avg))
        record['acc@3'].append(float(top3.avg))
        record['mAP'].append(map)

        if args.writer:
            args.writer.add_scalar('loss_val', losses.avg, global_step=epoch)
            args.writer.add_scalar('top1_val', top1.avg, global_step=epoch)
            args.writer.add_scalar('top3_val', top3.avg, global_step=epoch)
            args.writer.add_scalar('map_val', map, global_step=epoch)

    return top1.avg


def save_checkpoint(state, is_best, dir='', filename='checkpoint.pth.tar'):
    torch.save(state, dir + filename)
    if is_best:
        shutil.copyfile(dir + filename, dir + 'model_best.pth.tar')


def save_checkpoint_with_bdnn(pre_model, model, epoch, best_acc1, is_best, projection_type, dir=''):
    # assign weights and bias
    if projection_type in ('L1S', 'L1B', 'L2S', 'L2B', 'LInfS', 'LInfB'):
        pre_model.fc[2].weight.data = model.forward_net[0].weight.data
        pre_model.fc[2].bias.data = model.forward_net[0].bias.data
    else:
        pre_model.fc.weight.data = model.forward_net[0].weight.data
        pre_model.fc.bias.data = model.forward_net[0].bias.data
    # save
    torch.save({
        'arch': 'resnet18',
        'epoch': epoch,
        'state_dict': pre_model.state_dict(),
        'best_acc1': best_acc1,
    }, dir + 'bddnet.pth.tar')
    if is_best:
        shutil.copyfile(dir + 'bddnet.pth.tar', dir + 'bddnet_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_average_precision(output, target):
    with torch.no_grad():
        num_samples = output.size()[0]
        num_classes = output.size()[1]
        # Convert to numpy
        output = output.numpy()
        target = target.numpy()
        # Sort by confidence
        sorted_ind = np.argsort(-output, axis=0)
        aps = []
        for n in range(6):
            npos = (target == n).sum()
            tp = np.zeros(num_samples)
            fp = np.zeros(num_samples)
            for i in range(num_samples):
                if target[sorted_ind[i, n]] == n:
                    tp[i] = 1.
                else:
                    fp[i] = 1.
            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            # rec = tp / float(npos)
            rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = voc_ap(rec, prec, use_07_metric=False)
            aps += [ap]
    return np.mean(aps), aps


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
