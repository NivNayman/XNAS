import os
import sys
import time
import numpy as np
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_imagenet import NetworkImageNet as Network
import parameters as params
from utils import infer, data_transforms_imagenet_valid, count_parameters_in_MB

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='../datasets/imagenet/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--model_path', type=str, default='trained_models/imagenet.path.tar',
                    help='path of pretrained model')
parser.add_argument('--arch', type=str, default='XNAS', help='which architecture to use')
parser.add_argument('--calc_flops', action='store_true', default=False, help='calc_flops')

# Network design
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--init_channels', type=int, default=46, help='init_channels')

args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    logging.info(genotype)

    dataset = params.datasets['ImageNet']

    network_params = {'C': args.init_channels,
                      'num_classes': dataset.num_classes,
                      'layers': args.layers,
                      'genotype': genotype,
                      }
    model = Network(**network_params)

    if args.calc_flops:
        from thop import profile, clever_format
        input = torch.randn(1, dataset.num_channels, dataset.hw[0], dataset.hw[1])
        flops, num_params = profile(model, inputs=(input, ))
        flops, num_params = clever_format([flops, num_params], "%.2f")

    utils.load(model, args.model_path)

    model = model.cuda()

    val_transform = data_transforms_imagenet_valid()
    validdir = os.path.join(args.data, 'val')
    valid_data = dset.ImageFolder(validdir, val_transform)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    with torch.no_grad():
        ts = time.time()
        val_acc, samples = infer(valid_queue, model, args.report_freq)
        te = time.time()
        infer_time_ms = (te - ts) / samples * 1000

    if args.calc_flops:
        logging.info('Validation Accuracy: %.2f%% | Number of parameters: %s | Inference time: %2.2fms | Flops: %s',
                     val_acc, num_params, infer_time_ms, flops)
    else:
        logging.info('Validation Accuracy: %.2f%% | Inference time: %2.2fms', val_acc, infer_time_ms)


if __name__ == '__main__':
    main()
