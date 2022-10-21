from XVFInet import *
import numpy as np
import torch
from my_utils import *
from tensorboardX import SummaryWriter
import socket
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim


parser = argparse.ArgumentParser(description='PyTorch XVFI TEST')

parser.add_argument('--test_data_path', type=str, default='D:\Github-package\PM2.5\XVFI/test')
parser.add_argument('--multiple', type=int, default=3, help='插帧数目')  # 2代表插1帧，3代表插2帧
parser.add_argument('--custom_path', type=str, default='./320_even_frame', help='path for custom video containing frames')
parser.add_argument('--model', default='checkpoint/4_DESKTOP-0NFK80A_XVFInet_epoch_73.pth', help='sr pretrained base model')

args = parser.parse_args()
print(args)

# GPU是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')

# 模型初始化 & 打印
model = XVFInet(num_fea=64, scale=4).to(device)  # 64,4
# 加载预训练模型
model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
print('Pre-trained SR model is loaded.')

print('===> Loading datasets')
final_test_loader = get_test_data(args, multiple=args.multiple,validation=False)  # multiple is only used for X4K1000FPS

def test():
    model_net.eval()
    print("------------------------------------------- Test_Custom ----------------------------------------------")
    with torch.no_grad():
        start_time = time.time()
        for testIndex, (frames, t_value, scene_name, frameRange) in enumerate(final_test_loader):
            frameT = frames[:, :, -1, :, :]  # [1,C,H,W]
            It_Path, I0_Path, I1_Path = frameRange

            frameT = Variable(frameT.to(device))  # ground truth for frameT
            t_value = Variable(t_value.to(device))

            if (testIndex % (multiple - 1)) == 0:
                input_frames = frames[:, :, :-1, :, :]  # [1,C,T,H,W]
                input_frames = Variable(input_frames.to(device))

                B, C, T, H, W = input_frames.size()
                H_padding = (args.divide - H % args.divide) % args.divide
                W_padding = (args.divide - W % args.divide) % args.divide
                if H_padding != 0 or W_padding != 0:
                    input_frames = F.pad(input_frames, (0, W_padding, 0, H_padding), "constant")

            t0 = time.time()
            pred_frameT = model_net(input_frames, t_value, is_training=False)
            t1 = time.time()
            print(t1 - t0)
            if H_padding != 0 or W_padding != 0:
                pred_frameT = pred_frameT[:, :, :H, :W]

            epoch_save_path = args.custom_path
            scene_save_path = os.path.join(epoch_save_path, scene_name[0])
            pred_frameT = np.squeeze(pred_frameT.detach().cpu().numpy())  # (h,w)
            pred_frameT = pred_frameT * 800

            it_path = I0_Path[0][0:len(I0_Path[0]) - 4]
            it_path = int(it_path) + 1
            it_path = '{0:02d}'.format(it_path) + '.mat'
            # ex：It在I1-I4之间，为I2时，命名为1+1=02.mat 下面打印出来确定一下
            print(os.path.join(scene_save_path, I0_Path[0]))  # 显示I0
            print(t_value.item())
            print(it_path)  # 显示It
            print(os.path.join(scene_save_path, I1_Path[0]))  # 显示I1

            # 保存为mat
            savemat(it_path, mdict = {'data': pred_frameT})

    print("------------------------------------------- Test End !!! ----------------------------------------------")
