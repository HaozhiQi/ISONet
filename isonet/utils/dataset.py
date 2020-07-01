import os
import torch
import torchvision
import torchvision.transforms as transforms
from isonet.utils.config import C


def construct_dataset():
    transform = {
        'cifar_train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'cifar_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'ilsvrc2012_train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        'ilsvrc2012_test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    }

    if C.DATASET.NAME == 'CIFAR10':
        train_set = torchvision.datasets.CIFAR10(root=C.DATASET.ROOT, train=True, transform=transform['cifar_train'])
        val_set = torchvision.datasets.CIFAR10(root=C.DATASET.ROOT, train=False, transform=transform['cifar_test'])
    elif C.DATASET.NAME == 'CIFAR100':
        train_set = torchvision.datasets.CIFAR100(root=C.DATASET.ROOT, train=True, transform=transform['cifar_train'])
        val_set = torchvision.datasets.CIFAR100(root=C.DATASET.ROOT, train=False, transform=transform['cifar_test'])
    elif C.DATASET.NAME == 'ILSVRC2012':
        train_dir = os.path.join(C.DATASET.ROOT, 'ILSVRC2012', 'train')
        val_dir = os.path.join(C.DATASET.ROOT, 'ILSVRC2012', 'val')
        train_set = torchvision.datasets.ImageFolder(train_dir, transform['ilsvrc2012_train'])
        val_set = torchvision.datasets.ImageFolder(val_dir, transform['ilsvrc2012_test'])
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=C.SOLVER.TRAIN_BATCH_SIZE,
                                               shuffle=True, num_workers=C.DATASET.NUM_WORKERS,
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=C.SOLVER.TEST_BATCH_SIZE,
                                             shuffle=False, num_workers=C.DATASET.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader
