# tools for analyze models

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

classes = ('angry', 'disgust', 'fear', 'happy', 'natural', 'sad', 'surprise')


def imshow(img):
    img[0, :, :] = img[0, :, :] * 0.229 + 0.485
    img[1, :, :] = img[1, :, :] * 0.224 + 0.456
    img[2, :, :] = img[2, :, :] * 0.225 + 0.406
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def shows_some_images(dataloder, n):
    # get some random training images
    dataiter = iter(dataloder)
    images, labels = dataiter.next()
    images = images[:n, :, :, :]
    labels = labels[:n]
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(n)))
    return [classes[labels[j]] for j in range(n)]

def shows_some_images_with_path(dataloder, n):
    # get some random training images
    dataiter = iter(dataloder)
    images, labels, path = dataiter.next()
    images = images[:n, :, :, :]
    labels = labels[:n]
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(n)))
    return [classes[labels[j]] for j in range(n)]



def model_predict(data_loader, model, topk=1):
    model.eval()  # switch to evaluate mode
    with torch.no_grad():
        outputs = torch.empty(0)
        targets = torch.empty(0, dtype=torch.long)
        for i, (images, target) in enumerate(data_loader):
            # compute output
            output = model(images)

            outputs = torch.cat((outputs, output.cpu()), dim=0)
            targets = torch.cat((targets, target.cpu()), dim=0)

    return outputs, targets

def model_predict_with_path(data_loader, model, topk=1):
    model.eval()  # switch to evaluate mode
    with torch.no_grad():
        outputs = torch.empty(0)
        targets = torch.empty(0, dtype=torch.long)
        for i, (images, target, path) in enumerate(data_loader):
            # compute output
            output = model(images)

            outputs = torch.cat((outputs, output.cpu()), dim=0)
            targets = torch.cat((targets, target.cpu()), dim=0)

    return outputs, targets, path