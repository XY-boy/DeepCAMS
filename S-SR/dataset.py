import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange
import scipy.io as scio
from torch.nn import functional as F



def mat_np2Tensor(imgIn):
    ## input : T, H, W, C

    imgIn[imgIn > 800] = 800
    imgIn[imgIn < 0] = 0
    # normalization [-1,1]
    imgIn = imgIn / 800
    # ts = (3, 0, 1, 2)  ############# dimension order should be [C, T, H, W]
    imgIn = torch.Tensor(imgIn.astype(float)).mul_(1.0)

    # nan 值替换成0
    imgIn = torch.where(torch.isnan(imgIn), torch.full_like(imgIn, 0), imgIn)  # torch.full_like(a, 0)创造一个a尺寸一样但元素为0的矩阵
    imgIn = imgIn.unsqueeze(0)

    return imgIn

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".mat"])


def load_img(filepath):
    img = scio.loadmat(filepath)['data']
    # print(img.shape)
    # y, _, _ = img.split()
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.shape
    #(th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in[iy:iy + ip, ix:ix + ip]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp]
    #img_bic = img_bic.crop((ty, tx, ty + tp, tx + tp))

    #info_patch = {
    #    'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar


def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = img_in[:, ::-1]
        img_tar = img_tar[:, ::-1]
        info_aug['flip_h'] = True

    if random.random() < 0.5 and rot:
        rot = random.randint(0, 3)  # random rotate
        img_in = np.rot90(img_in, rot)
        img_tar = np.rot90(img_tar, rot)
        info_aug['flip_v'] = True
        info_aug['trans'] = True

    return img_in, img_tar, info_aug


class DatasetFromFolder(data.Dataset):
    def __init__(self, HR_dir, LR_dir, patch_size, upscale_factor, data_augmentation,
                 transform=None):
        super(DatasetFromFolder, self).__init__()
        self.hr_image_filenames = [join(HR_dir, x) for x in listdir(HR_dir) if is_image_file(x)]
        self.lr_image_filenames = [join(LR_dir, x) for x in listdir(LR_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):

        target = load_img(self.hr_image_filenames[index])
        name = self.hr_image_filenames[index]
        # print(name)
        #lr_name = name[:39]+'LR/'+name[42:-4]+'x4.png'
        #lr_name = name[:39]+'LR_16x/'+name[42:]
        lr_name = name.replace('HR', 'LR4x')
        input = load_img(lr_name)

        input, target, = get_patch(input, target, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            input, target, _ = augment(input, target)

        if self.transform:
            input = mat_np2Tensor(input)
            # input = self.transform(input)
            # target = self.transform(target)
            target = mat_np2Tensor(target)
        input2 = F.interpolate(target.unsqueeze(0), scale_factor=1/3, mode='bicubic').squeeze(0)
        # print(input2.size())

        return input2, target

    def __len__(self):
        return len(self.hr_image_filenames)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir, upscale_factor, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])

        bicubic = rescale_img(input, self.upscale_factor)

        if self.transform:
            input = mat_np2Tensor(input)
            # input = self.transform(input)
            # bicubic = self.transform(bicubic)
            bicubic = mat_np2Tensor(bicubic)

        return bicubic, file

    def __len__(self):
        return len(self.image_filenames)
