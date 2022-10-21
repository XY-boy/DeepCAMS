from __future__ import print_function
import argparse

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from model import ABPN_v5, VDSR
import torchvision.transforms as transforms
from collections import OrderedDict
import logging
import numpy as np
from os.path import join
import time
import math
from dataset import is_image_file
import utils_logger
from PIL import Image, ImageOps
from os import listdir
from prepare_images import *
import torch.utils.data as utils
from torch.autograd import Variable
import scipy.io as scio
from torch.nn import functional as F


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=3, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=120, help='0 to use original frame size')
parser.add_argument('--stride', type=int, default=120, help='0 to use original patch size')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--image_dataset', type=str, default='/media/amax/XY/SR_PM2.5/Original_CAMS/2009-mat/cams/')
parser.add_argument('--model_type', type=str, default='ABPN')
parser.add_argument('--image_output', default='/media/amax/XY/SR_PM2.5/S-SR-Ori/2009/', help='Location to save checkpoint models')
parser.add_argument('--model', default='Model_PM2.5/VAE_epoch_300.pth', help='sr pretrained base model')

opt = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('===> Building model ', opt.model_type)
model = ABPN_v5(input_dim=1, dim=32)
# model = VDSR()
model = model.to(device)

model_name = os.path.join(opt.model)
if os.path.exists(model_name):
    pretrained_dict = torch.load(model_name, map_location=lambda storage, loc: storage)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Pre-trained SR model is loaded.')


img_splitter = ImageSplitter(opt.patch_size, opt.upscale_factor, opt.stride)

def eval():
    # utils_logger.logger_info('AIM-track', log_path='AIM-track.log')
    # logger = logging.getLogger('AIM-track')
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False

    # number of parameters
    # number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    # logger.info('Params number: {}'.format(number_parameters))

    LR_filename = opt.image_dataset
    LR_image = [join(LR_filename, x) for x in listdir(LR_filename) if is_image_file(x)]
    SR_image = [join(opt.image_output, x) for x in listdir(LR_filename) if is_image_file(x)]
    print(LR_image)

    # record PSNR, runtime
    # test_results = OrderedDict()
    # test_results['runtime'] = []
    #
    # logger.info(opt.image_dataset)
    # logger.info(opt.image_output)
    idx = 0


    for i in range(LR_image.__len__()):

        idx += 1
        img_name, ext = os.path.splitext(LR_image[i])
        # logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        # LR = Image.open(LR_image[i]).convert('RGB')
        LR = scio.loadmat(LR_image[i])['data']*1e9
        LR[LR > 800] = 800
        LR[LR < 0] = 0
        print('--------')
        print('LR dir: {}'.format(LR_image[i]))
        LR = torch.Tensor(LR.astype(float)).mul_(1.0) / 800
        LR = LR.unsqueeze(0).unsqueeze(0).cuda()
        LR = torch.where(torch.isnan(LR), torch.full_like(LR, 0), LR)
        # print(LR.size())
        # LR_90 = LR.transpose(Image.ROTATE_90)
        # LR_180 = LR.transpose(Image.ROTATE_180)
        # LR_270 = LR.transpose(Image.ROTATE_270)
        # LR_f = LR.transpose(Image.FLIP_LEFT_RIGHT)
        # LR_90f = LR_90.transpose(Image.FLIP_LEFT_RIGHT)
        # LR_180f = LR_180.transpose(Image.FLIP_LEFT_RIGHT)
        # LR_270f = LR_270.transpose(Image.FLIP_LEFT_RIGHT)

        with torch.no_grad():

            pred = chop_forward(LR,model)

            # pred = model(LR)
            # pred = pred.cpu().data[0].numpy().astype(np.float32).squeeze(0)
            print(pred.shape)

            # pred_90 = chop_forward(LR_90, model)
            # pred_180 = chop_forward(LR_180, model)
            # pred_270 = chop_forward(LR_270, model)
            # pred_f = chop_forward(LR_f, model)
            # pred_90f = chop_forward(LR_90f, model)
            # pred_180f = chop_forward(LR_180f, model)
            # pred_270f = chop_forward(LR_270f, model)


        # pred_90 = np.rot90(pred_90, 3)
        # pred_180 = np.rot90(pred_180, 2)
        # pred_270 = np.rot90(pred_270, 1)
        # pred_f = np.fliplr(pred_f)
        # pred_90f = np.rot90(np.fliplr(pred_90f), 3)
        # pred_180f = np.rot90(np.fliplr(pred_180f), 2)
        # pred_270f = np.rot90(np.fliplr(pred_270f), 1)
        # prediction = (pred + pred_90 + pred_180 + pred_270 + pred_f + pred_90f + pred_180f + pred_270f) * 255.0 / 8.0
        # prediction = prediction.clip(0, 255)
        print('SR dir:{}'.format(SR_image[i]))
        print('--------')
        scio.savemat(SR_image[i],{'data':pred})

        # Image.fromarray(np.uint8(prediction)).save(SR_image[i])



    # print("PSNR_predicted=", avg_psnr_predicted / count)

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih % modulo)
    iw = iw - (iw % modulo)
    img = img.crop((0, 0, ih, iw))
    #y, cb, cr = img.split()
    return img


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        float32, [0, 255]
        float32, [0, 255]
    '''
    img.astype(np.float32)
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    rlt = rlt.round()

    return rlt

def PSNR(pred, gt, shave_border):
    pred = pred[shave_border:-shave_border, shave_border:-shave_border]
    gt = gt[shave_border:-shave_border, shave_border:-shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)


def chop_forward(img, network):

    channel_swap = (1, 2, 0)
    # img = transform(img).unsqueeze(0)
    img_patch = img_splitter.split_img_tensor(img)

    testset = utils.TensorDataset(img_patch)
    test_dataloader = utils.DataLoader(testset, num_workers=opt.threads,
                                       drop_last=False, batch_size=opt.testBatchSize, shuffle=False)
    out_box = []

    for iteration, batch in enumerate(test_dataloader, 1):
        input = Variable(batch[0]).to(device)

        with torch.no_grad():
            prediction = network(input)

        for j in range(prediction.shape[0]):
            out_box.append(prediction[j,:,:,:])

    SR = img_splitter.merge_img_tensor(out_box)
    SR = F.interpolate(SR, size=(720, 1440), mode='bicubic')
    SR = torch.squeeze(SR,0)
    # print(SR.size())
    # SR = SR.data[0].numpy().transpose(channel_swap)
    SR = SR.data[0].numpy().astype(float)

    return SR



##Eval Start!!!!
eval()
