from XVFInet import *
import numpy as np
import torch
from my_utils import *
from tensorboardX import SummaryWriter
import socket
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import cv2
from matplotlib import pyplot as plt
import scipy.io as scio

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

parser = argparse.ArgumentParser(description='PyTorch XVFI')

parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--net_type', type=str, default='XVFInet', choices=['XVFInet'], help='The type of Net')
parser.add_argument('--net_object', default=XVFInet, choices=[XVFInet], help='The type of Net')
parser.add_argument('--exp_num', type=int, default=1, help='The experiment number')
parser.add_argument('--phase', type=str, default='test_custom', choices=['train', 'test', 'test_custom', 'metrics_evaluation',])
parser.add_argument('--continue_training', action='store_true', default=True, help='continue the training')
parser.add_argument('--pretrained_sr', default='4_DESKTOP-0NFK80A_XVFInet_epoch_190.pth',
                    help='sr pretrained base model default=3x_dl10VDBPNF7_epoch_84.pth')

# 训练集 & 测试集
parser.add_argument('--dataset', default='OURS', choices=['OURS'],help='Training/test Dataset')
parser.add_argument('--train_data_path', type=str, default='D:\Github-package\SR_PM2.5\Dataset\GEOS-CF-1-hour\T-SR-Training-set/2019_mat')
parser.add_argument('--val_data_path', type=str, default='D:/Github-package\SR_PM2.5\XVFI\eval_while_training/val')

# 模型保存及日志
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='checkpoint_dir')
parser.add_argument('--log_dir', type=str, default='./log_dir', help='Directory name to save training logs')

# 训练超参数设置
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--epochs', type=int, default=200, help='The number of epochs to run')
parser.add_argument('--freq_display', type=int, default=100, help='The number of iterations frequency for display')
parser.add_argument('--save_img_num', type=int, default=4,
                    help='The number of saved image while training for visualization. It should smaller than the batch_size')
parser.add_argument('--init_lr', type=float, default=1e-4, help='The initial learning rate')
parser.add_argument('--lr_dec_fac', type=float, default=0.25, help='衰减率0.25即每次衰减变为原来的1/4；step - lr_decreasing_factor')
parser.add_argument('--lr_milestones', type=int, default=[50, 100, 150,200])  # 在100,150,180轮分别衰减一次
parser.add_argument('--lr_dec_start', type=int, default=0,
                    help='When scheduler is StepLR, lr decreases from epoch at lr_dec_start')
parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size.')
parser.add_argument('--weight_decay', type=float, default=0, help='for optim., weight decay (default: 0)')

parser.add_argument('--need_patch', default=True, help='get patch form image while training')
parser.add_argument('--img_ch', type=int, default=1, help='base number of channels for image')  # 产品通道为1
parser.add_argument('--nf', type=int, default=64, help='base number of channels for feature maps')  # 64
parser.add_argument('--module_scale_factor', type=int, default=4, help='sptial reduction for pixelshuffle')  # 大尺寸选4
parser.add_argument('--patch_size', type=int, default=512, help='patch size')  # 720*1440，选360,180都可
parser.add_argument('--num_thrds', type=int, default=0, help='number of threads for data loading')
parser.add_argument('--S_trn', type=int, default=1, help='The lowest scale depth for training')
parser.add_argument('--S_tst', type=int, default=1, help='The lowest scale depth for test')

# LOSS函数设置
parser.add_argument('--loss_type', default='L1', choices=['L1', 'MSE', 'L1_Charbonnier_loss'], help='Loss type')
parser.add_argument('--rec_lambda', type=float, default=1.0, help='Lambda for Reconstruction Loss')  # 重建loss系数为1

# 测试阶段相关参数---------训练不看
parser.add_argument('--test_data_path', type=str, default='D:\Github-package\SR_PM2.5\XVFI/test_custom_data')
parser.add_argument('--test_img_dir', type=str, default='D:\Github-package\PM2.5\XVFI/test/')
parser.add_argument('--saving_flow_flag', default=False)
parser.add_argument('--multiple', type=int, default=3, help='插帧数目')  # 2代表插1帧，3代表插2帧
parser.add_argument('--metrics_types', type=list, default=["PSNR", "SSIM", "tOF"], choices=["PSNR", "SSIM", "tOF"])

""" Settings for test_custom (when [phase=='test_custom']) """
parser.add_argument('--custom_path', type=str, default='H:\SR_PM2.5\T-SR-Geos-CF-Test\Input_9/', help='path for custom video containing frames')
parser.add_argument('--custom_save_path', type=str, default='H:\SR_PM2.5\T-SR-Geos-CF-Test\Output_16/T-SR-XVFI/2020-13-01/', help='path for custom video containing frames')

args = parser.parse_args()
cudnn.benchmark = True
hostname = str(socket.gethostname())

print(args)
writer = SummaryWriter('runs/Final_model_trained_by_Y2019_x900')  # 保存log

# ------一些函数------------
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: {:03f} M'.format((num_params / 1e6)))

def checkpoint(epoch):
    model_out_path = args.checkpoint_dir + str(
        args.module_scale_factor) + '_' + hostname + '_'+ args.net_type + "_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def calculate_correlation_coefficient(img0, img1):
    img0 = img0.reshape(img0.size, order='C')  # 将矩阵转换成向量。按行转换成向量，第一个参数就是矩阵元素的个数
    img1 = img1.reshape(img1.size, order='C')
    return np.corrcoef(img0, img1)[0, 1]
    # return 1 - np.sum((img0 - img1)**2) / np.sum((img0 - np.mean(img0))**2)

def calculate_rmse(img1,img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return math.sqrt(mse)

# --------------------------


# GPU是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')

# 模型初始化 & 打印
model = XVFInet(num_fea=args.nf, scale=args.module_scale_factor).to(device)  # 64,4
print_network(model)

# 加载预训练模型
if args.continue_training:
    model_name = os.path.join(args.checkpoint_dir + args.pretrained_sr)
    if os.path.exists(model_name):
        # model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

# LOSS函数初始化：重建loss + 平滑loss
criterion = [set_rec_loss(args).to(device), set_smoothness_loss().to(device)]

# 优化器
optimizer = optim.Adam(model.parameters(), lr=args.init_lr, betas=(0.9, 0.999),weight_decay=args.weight_decay)  # optimizer
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_dec_fac)

# 加载训练集，验证集
if args.phase == 'train':
    train_loader = get_train_data(args,max_t_step_size=12)  # max_t_step_size (temporal distance) = 24/2=12
    valid_loader = get_test_data(args, multiple=3, validation=True, valid_step=3)  # multiple

    # 开始训练
    best_epoch = 0
    best_test_psnr = 0.0
    multi_scale_recon_loss = criterion[0].cuda()
    smoothness_loss = criterion[1].cuda()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_loss = 0.0  # 记录每一轮的loss
        model.train()
        for trainIndex, (frames, t_value) in enumerate(train_loader):
            input_frames = frames[:, :, :-1, :]  # [B, C, T, H, W]  # 取前两个，即I0，I1
            frameT = frames[:, :, -1, :]  # [B, C, H, W] 需要预测的真值

            input_frames = Variable(input_frames.to(device))
            frameT = Variable(frameT.to(device))  # ground truth for frameT
            t_value = Variable(t_value.to(device))  # [B,1]
            # print(input_frames.size())
            # print(t_value)

            optimizer.zero_grad()
            t0 = time.time()
            pred_frameT_pyramid, pred_flow_pyramid, occ_map, simple_mean = model(input_frames, t_value)

            # ------------ loss 计算：重建loss + 光流平滑loss ----------
            rec_loss = 0.0
            smooth_loss = 0.0
            for l, pred_frameT_l in enumerate(pred_frameT_pyramid):  # 对于多尺度结构（金字塔）的每一层都计算重建loss,真值bicubic降采样
                rec_loss += args.rec_lambda * multi_scale_recon_loss(pred_frameT_l, F.interpolate(frameT, scale_factor=1 / (2 ** l),
                                                                                   mode='bicubic', align_corners=False))
            # 光流平滑loss只计算第四层
            smooth_loss += 0.5 * smoothness_loss(pred_flow_pyramid[0],F.interpolate(frameT, scale_factor=1 / args.module_scale_factor,
                                                                   mode='bicubic',align_corners=False))
            rec_loss /= len(pred_frameT_pyramid)  # 取平均
            total_loss = rec_loss + smooth_loss
            epoch_loss += total_loss.item()
            # print(torch.any(torch.isnan(input_frames)))
            t1 = time.time()
            print("===> Epoch[{}]({}/{}): rec_Loss: {:.8f} smooth_Loss: {:.8f} || Timer: {:.4f} sec.".format(epoch, trainIndex,
                                                                                     len(train_loader), rec_loss.item(), smooth_loss,
                                                                                     (t1 - t0)))
            # --------- loss 计算结束 -----------------------------------

            pred_frameT = pred_frameT_pyramid[0]  # 最终预测结果：I^0_t at original scale (s=0)
            pred_coarse_flow = 2 ** (args.S_trn) * F.interpolate(pred_flow_pyramid[-1], scale_factor=2 ** (
                args.S_trn) * args.module_scale_factor, mode='bicubic', align_corners=False)  # 粗糙流
            pred_fine_flow = F.interpolate(pred_flow_pyramid[0], scale_factor=args.module_scale_factor, mode='bicubic',
                                           align_corners=False)  # 精细流
            # 反向传播
            total_loss.backward()  # Backpropagate
            optimizer.step()  # Optimizer update


        if epoch >= args.lr_dec_start:  # 判断学习率是否衰减
            scheduler.step()
        # 写入tensorboard
        print("===> Epoch {} Complete: Avg. Loss: {:.8f}".format(epoch, epoch_loss / len(train_loader)))
        writer.add_scalar('Avg. Loss', epoch_loss / len(train_loader), epoch)
        # 保存模型
        checkpoint(epoch)

        # -------------------------- test while training ----------------
        avg_test_psnr = 0.0
        avg_linear_psnr = 0.0
        multiple = 3
        args.divide = 2 ** (args.S_tst) * args.module_scale_factor * 4  # 32
        model.eval()
        print("------------------------------------------- eval while training  ----------------------------------------------")
        with torch.no_grad():
            start_time = time.time()
            for testIndex, (frames, t_value, scene_name, frameRange) in enumerate(valid_loader):
                # Shape of 'frames' : [1,C,T+1,H,W]
                frameT = frames[:, :, -1, :, :]  # [1,C,H,W]  -1 代表从末尾索引，frames按照I0,I1,Tt
                It_Path, I0_Path, I1_Path = frameRange

                frameT = Variable(frameT.to(device))  # ground truth for frameT
                frame_0 = np.squeeze(frames[:, :, 0, :, :].detach().cpu().numpy())  # (h, w)
                frame_1 = np.squeeze(frames[:, :, 1, :, :].detach().cpu().numpy())  # (h, w)
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
                pred_frameT = model(input_frames, t_value, is_training=False)
                t1 = time.time()
                if H_padding != 0 or W_padding != 0:
                    pred_frameT = pred_frameT[:, :, :H, :W]

                print(t1 - t0)
                # print(input_frames.size())
                # print(pred_frameT.size())
                # test_loss = args.rec_lambda * multi_scale_recon_loss(pred_frameT, frameT)
                pred_frameT = np.squeeze(pred_frameT.detach().cpu().numpy())  # (h,w)
                frameT = np.squeeze(frameT.detach().cpu().numpy())  # (h, w)



                """ compute PSNR & SSIM """
                output_img = 900 * pred_frameT  # [h,w,c] and [-1,1] to [0,255]
                target_img = 900 * frameT  # [h,w,c] and [-1,1] to [0,255]
                # test_psnr = psnr(target_img, output_img)
                test_psnr = psnr(target_img, output_img)

                # 线性组合结果
                linear = frame_0 * t_value.item() + frame_1 * (1 - t_value.item())
                linear_psnr = psnr(target_img, linear * 900)
                print('ours: {} dB  linear: {} dB'.format(test_psnr,linear_psnr))

                # it_path = I0_Path[0][0:len(I0_Path[0]) - 4]
                # it_path = int(it_path) + int(It_Path[0][3:len(It_Path[0]) - 4]) + 1
                # it_path = '{0:02d}'.format(it_path) + '.mat'
                # print(os.path.join(scene_save_path, it_path))
                # # print(it_path)
                # # 以mat保存
                #
                # scio.savemat(os.path.join(scene_save_path, it_path), {'data': output_img})

                # print(t_value)
                if (epoch % 1)==0:
                    plt.figure(dpi=500)
                    plt.subplot(2, 2, 1).set_title('Ours')
                    plt.imshow(output_img, vmin=0, vmax=200)
                    plt.subplot(2, 2, 2).set_title('GT'+'_'+ str(round(t_value.item(), 4)))
                    plt.imshow(target_img, vmin=0, vmax=200)

                    plt.subplot(2, 2, 3).set_title('linear')
                    plt.imshow(900 * linear, vmin=0, vmax=200)
                    # plt.subplot(2, 2, 4)
                    # plt.imshow(frame_1)

                    save_name = str(epoch) + '_' + str(round(t_value.item(), 4)) + '.png'
                    plt.savefig('./linear_vs_ours/'+save_name)
                    plt.cla()
                    plt.close("all")

                avg_test_psnr += test_psnr
                avg_linear_psnr +=linear_psnr

        writer.add_scalars('Avg. PSNR', { 'Ours':avg_test_psnr / len(valid_loader),'Linear':avg_linear_psnr / len(valid_loader)}, epoch)

                # epoch_save_path = os.path.join(args.test_img_dir, args.model_dir, 'latest' + postfix)
                # check_folder(epoch_save_path)
                # scene_save_path = os.path.join(epoch_save_path, scene_name[0])
                # check_folder(scene_save_path)
# # ------------------------------------------测试部分------------------------------------------------------------------
if args.phase == 'test_custom':
    multiple = 3
    args.divide = 2 ** (args.S_tst) * args.module_scale_factor * 4  # 32

    test_loader = get_test_data(args, multiple=multiple, validation=True, valid_step=multiple)  # multiple
    model.eval()

    print("------------------------------------------- Test_custom----------------------------------------------")
    with torch.no_grad():
        start_time = time.time()
        for testIndex, (frames, t_value, scene_name, frameRange) in enumerate(test_loader):
            # Shape of 'frames' : [1,C,T+1,H,W]
            frameT = frames[:, :, -1, :, :]  # [1,C,H,W]
            It_Path, I0_Path, I1_Path = frameRange

            frameT = Variable(frameT.to(device))  # ground truth for frameT
            frame_0 = np.squeeze(frames[:, :, 0, :, :].detach().cpu().numpy())  # (h, w)
            frame_1 = np.squeeze(frames[:, :, 1, :, :].detach().cpu().numpy())  # (h, w)
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
            pred_frameT = model(input_frames, t_value, is_training=False)
            t1 = time.time()
            # print(t1 - t0)

            if H_padding != 0 or W_padding != 0:
                pred_frameT = pred_frameT[:, :, :H, :W]
            # print(pred_frameT.size())

            pred_frameT = np.squeeze(pred_frameT.detach().cpu().numpy())  # (h,w)预测值
            frameT = np.squeeze(frameT.detach().cpu().numpy())  # (h, w)真值

            """ 反归一化 """
            output_img = (pred_frameT*800)   # [h,w,c] and [-1,1] to [0,255]
            linear = frame_0 * t_value.item() + frame_1 * (1 - t_value.item())
            linear = (linear*800)
            # target_img = frameT  # [h,w,c] and [-1,1] to [0,255]

            # 保存
            save_path = args.custom_save_path  # 结果保存文件夹
            save_linear_path = args.custom_save_path.replace('T-SR-XVFI', 'T-SR-Linear')  # 结果保存文件夹

            scene_save_path = os.path.join(save_path, scene_name[0])
            scene_save_path_linear = os.path.join(save_linear_path, scene_name[0])
            # 不存在则新建
            if not os.path.isdir(scene_save_path):
                os.makedirs(scene_save_path)
            if not os.path.isdir(scene_save_path_linear):
                os.makedirs(scene_save_path_linear)

            # 模拟实验，插出的帧It的命名规则为I0的名字+1，+2，例如I0=01.mat，It=02.mat，03.mat
            # 在模拟实验中，连续挖掉2帧，例如01,04,07，10,13,16,19,22，25，其中25是第二天的第一帧
            # -- CAMS 命名规则----
            # it_path = I0_Path[0][6:len(I0_Path[0])-4]
            # it_path = int(it_path) + int(It_Path[0][9:len(It_Path[0])-4]) + 1
            # it_path = I0_Path[0][0:5] + '_' + '{0:02d}'.format(it_path)+'.mat'
            # ------------------------------------------------------

            # -- GEOS-CF 命名规则----
            it_path = I0_Path[0][0:len(I0_Path[0]) - 4]
            it_path = int(it_path) + int(It_Path[0][3:len(It_Path[0]) - 4]) + 1
            it_path = '{0:02d}'.format(it_path) + '.mat'
            # ------------------------------------------------------

            print('------------------------------------')
            print(I0_Path[0])
            print(I1_Path[0])
            print(t_value.item())

            print(os.path.join(scene_save_path, it_path))
            print(os.path.join(scene_save_path_linear, it_path))

            print('ours PSNR {}'.format(calculate_correlation_coefficient(frame_0,output_img)))
            print('linear PSNR {}'.format(calculate_correlation_coefficient(frame_0, linear)))

            # print(it_path)
            # 以mat保存
            scio.savemat(os.path.join(scene_save_path, it_path), {'data':output_img})
            scio.savemat(os.path.join(scene_save_path_linear, it_path), {'data':linear})

            print('------------------------------------')


