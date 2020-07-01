import os
import argparse
import warnings

import torch.backends.cudnn as cudnn

from isonet.utils.config import C

import isonet.utils.dataset as du
import isonet.utils.optim as ou
import isonet.utils.logger as lu

from isonet.models import *
from isonet.trainer import Trainer


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--output', default='default', type=str)
    parser.add_argument('--gpus', type=str)
    parser.add_argument('--resume', default='', type=str)
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    # disable imagenet dataset jpeg warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    # ---- setup GPUs ----
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    assert torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    cudnn.benchmark = True
    # ---- setup configs ----
    C.merge_from_file(args.cfg)
    C.SOLVER.TRAIN_BATCH_SIZE *= num_gpus
    C.SOLVER.TEST_BATCH_SIZE *= num_gpus
    C.SOLVER.BASE_LR *= num_gpus
    C.freeze()
    # ---- setup logger and output ----
    output_dir = os.path.join(C.OUTPUT_DIR, C.DATASET.NAME, args.output)
    os.makedirs(output_dir, exist_ok=True)
    logger = lu.construct_logger('isonet', output_dir)
    logger.info('Using {} GPUs'.format(num_gpus))
    logger.info(C.dump())
    # ---- setup dataset ----
    train_loader, val_loader = du.construct_dataset()

    net = ISONet()
    net.to(torch.device('cuda'))
    net = torch.nn.DataParallel(
        net, device_ids=list(range(args.gpus.count(',') + 1))
    )
    optim = ou.construct_optim(net, num_gpus)

    trainer = Trainer(
        torch.device('cuda'),
        train_loader,
        val_loader,
        net,
        optim,
        logger,
        output_dir,
    )

    if args.resume:
        cp = torch.load(args.resume)
        trainer.model.load_state_dict(cp['net'])
        trainer.optim.load_state_dict(cp['optim'])
        trainer.epochs = cp['epoch']
        trainer.train_acc = cp['train_accuracy']
        trainer.val_acc = cp['test_accuracy']

    trainer.train()


if __name__ == '__main__':
    main()
