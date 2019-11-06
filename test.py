import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import Network
import parameters as params
from utils import infer, data_transforms_cifar10, count_parameters_in_MB

parser = argparse.ArgumentParser("cifar10")
parser.add_argument('--dset_name', type=str, default='CIFAR10', help='data set name')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--model_path', type=str, default='trained_models/xnas_small_cifar10.t7',
                    help='path of pretrained model')
parser.add_argument('--arch', type=str, default='XNAS', help='which architecture to use')
parser.add_argument('--calc_flops', action='store_true', default=False, help='calc_flops')

# Model Design
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--num_reductions', type=int, default=2, help='Number of reduction cells')
parser.add_argument('--reduction_location_mode', type=str, default='uniform_start', help='reduction cells allocation.')
parser.add_argument('--do_SE', action='store_true', default=False)

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

    dataset = params.datasets[args.dset_name]
    network_params = {'C': args.init_channels,
                      'num_classes': dataset.num_classes,
                      'layers': args.layers,
                      'num_reductions': args.num_reductions,
                      'reduction_location_mode': args.reduction_location_mode,
                      'genotype': genotype,
                      'stem_multiplier': dataset.num_channels,
                      'do_SE': args.do_SE}
    model = Network(**network_params)

    logging.info("Loading model parameters from %s", args.model_path)
    utils.load(model, args.model_path)

    flops, num_params = None, None
    if args.calc_flops:
        from thop import profile, clever_format
        input = torch.randn(1, dataset.num_channels, dataset.hw[0], dataset.hw[1])
        flops, num_params = profile(model, inputs=(input, ))
        flops, num_params = clever_format([flops, num_params], "%.2f")

    model = model.cuda()

    test_transform = data_transforms_cifar10()
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    with torch.no_grad():
        test_acc, infer_time = infer(test_queue, model, args.report_freq)

    if args.calc_flops:
        logging.info('Test Accuracy: %.2f%% | Number of parameters: %s | Inference time: %2.2fms | Flops: %s',
                     test_acc, num_params, infer_time * 1000, flops)
    else:
        logging.info('Test Accuracy: %.2f%% | Inference time: %2.2fms', test_acc, infer_time * 1000)

if __name__ == '__main__':
    main()

