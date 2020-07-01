import os
import torch
import warnings
import argparse
from isonet.utils.misc import tprint, pprint
from isonet.utils.config import C
from isonet.models import *
from torchvision import datasets
from torchvision import transforms


def arg_parse():
    parser = argparse.ArgumentParser(description='Trains an ImageNet Classifier')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--gpus', type=str)
    parser.add_argument('--ckpt', default='', type=str)
    args = parser.parse_args()
    return args


def test(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            tprint(f'{batch_idx} / {len(test_loader)}: {100 * correct / total:.2f}')

    return correct / total


def main():
    args = arg_parse()
    # disable imagenet dataset jpeg warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    # ---- setup GPUs ----
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    assert torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    # ---- setup configs ----
    C.merge_from_file(args.cfg)
    C.SOLVER.TRAIN_BATCH_SIZE *= num_gpus
    C.SOLVER.TEST_BATCH_SIZE *= num_gpus
    C.SOLVER.BASE_LR *= num_gpus
    C.freeze()

    # Load datasets
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_dir = os.path.join(C.DATASET.ROOT, 'ILSVRC2012', 'val')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_dir, test_transform),
        batch_size=C.SOLVER.TEST_BATCH_SIZE,
        shuffle=False,
    )

    net = ISONet()
    net.to(torch.device('cuda'))
    net = torch.nn.DataParallel(
        net, device_ids=list(range(args.gpus.count(',') + 1))
    )
    cp = torch.load(args.ckpt)
    if 'net' in cp:
        net.load_state_dict(cp['net'])
    else:
        net.load_state_dict(cp)

    test_acc1 = test(net, val_loader)
    pprint(f'Top-1 Accuracy for {args.ckpt}: {100 * test_acc1:.2f}')


if __name__ == '__main__':
    main()
