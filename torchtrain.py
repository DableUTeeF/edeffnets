import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from efficientnet import EfficientNet
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import time
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from utils import Progbar
from PIL import Image
import time
torch.manual_seed(0)
np.random.seed(0)


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
    root = '/media/space/imagenet/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/'
    batch_size = 384
    train_transform = transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
    val_transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

    trainset = torchvision.datasets.ImageFolder(root + 'train', transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=32)

    testset = torchvision.datasets.ImageFolder(root + 'val', transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8)
    model = EfficientNet.from_name('efficientnet-b0')
    model = nn.DataParallel(model)
    model.cuda()
    # checkpoint = torch.load('/home/root1/PycharmProjects/ptt/checkpoint/69/box-can_0_0.8873.torch')
    # model.load_state_dict(checkpoint['net'])
    # del checkpoint
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), 1e-2,
                                momentum=0.9,
                                weight_decay=1e-5)

    lr_schudule = ExponentialLR(optimizer, 0.97)


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
                predicted = np.append(predicted, np.argmax(output, axis=1))
                targets = np.append(targets, target)
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
                end = time.time()

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
        acc, f1 = validate(testloader, model, criterion, i+1)
        dct = {'net': model.state_dict(),
               'opt': optimizer.state_dict(),
               'acc': acc,
               'f1': f1
               }
        torch.save(dct, f'checkpoint/{save_folder}/final_crop_{i}_{f1:.4f}.torch')
        lr_schudule.step(i+1)
