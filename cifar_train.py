import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from efficientnet import EfficientNet
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from utils import Progbar
import time


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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
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
            res.append(correct_k.mul_(1.0 / batch_size))
    return res


if __name__ == '__main__':
    root = '/home/palm/PycharmProjects/data'
    batch_size = 64
    lr_exp = 0.96
    init_lr = 1e-1
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(15),
         transforms.RandomAffine(10),
         transforms.RandomVerticalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                             download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    model = EfficientNet.from_name('efficientnet-b0', first_stride=False)
    model._fc = nn.Linear(model._fc.in_features, 10)
    model = nn.DataParallel(model)
    model.cuda()
    # checkpoint = torch.load('/home/root1/PycharmProjects/ptt/checkpoint/69/box-can_0_0.8873.torch')
    # model.load_state_dict(checkpoint['net'])
    # del checkpoint
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                momentum=0.9,
                                weight_decay=1e-5)

    lr_schudule = ExponentialLR(optimizer, lr_exp)


    def train(train_loader, model, criterion, optimizer):
        progress = Progbar(len(train_loader))
        model.train()
        for i, (images, target) in enumerate(train_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            suffix = [('loss', loss.item()), ('acc', acc1[0].cpu().numpy())]
            progress.update(i + 1, suffix)


    def validate(val_loader, model, criterion, epoch=0):
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = Progbar(len(val_loader))

        # switch to evaluate mode
        model.eval()
        predicted = np.array([], dtype='float32')
        targets = np.array([], dtype='float32')
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images = images.cuda()
                target = target.cuda()

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                predicted = np.append(predicted, np.argmax(output.cpu().numpy(), axis=1))
                targets = np.append(targets, target.cpu().numpy())
                acc1 = accuracy(output, target, topk=(1,))
                top1.update(acc1[0], images.size(0))

                suffix = [('loss', loss.item()), ('acc', acc1[0].cpu().numpy())]
                progress.update(i + 1, suffix)
        np.save(f'opt/predict_{epoch}.npy', predicted)
        np.save(f'opt/target_{epoch}.npy', targets)
        return top1.avg


    def cf(val_loader):
        progress = Progbar(len(val_loader))
        model.eval()
        predicted = np.array([], dtype='float32')
        targets = np.array([], dtype='float32')
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images = images.cuda()
                target = target.cuda()

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                predicted = np.append(predicted, np.argmax(output, axis=1))
                targets = np.append(targets, target)
                acc1 = accuracy(output, target, topk=(1,))

                suffix = [('loss', loss.item()), ('acc', acc1[0].cpu().numpy())]
                progress.update(i + 1, suffix)
        cfm = confusion_matrix(targets, predicted)
        return cfm


    save_folder = len(os.listdir('checkpoint'))
    os.mkdir(os.path.join('checkpoint', str(save_folder)))
    c = [31, 32, 33, 34, 35, 36, 37]
    for i in range(40):
        print(f'\033[{c[int(str(time.time())[-1]) % 7]}m', 'Epoch:', i + 1)
        train(trainloader, model, criterion, optimizer)
        torch.save(model.state_dict(), f'checkpoint/temp.torch')
        acc = validate(testloader, model, criterion, i + 1)
        dct = {'net': model.state_dict(),
               'opt': optimizer.state_dict(),
               'acc': acc,
               'init_lr': init_lr,
               'lr_exp': lr_exp,
               'batch_size': batch_size
               }
        torch.save(dct, f'checkpoint/{save_folder}/final_crop_{i}_{float(acc.cpu().numpy()):.4f}.torch')
        lr_schudule.step(i + 1)
