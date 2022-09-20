from utils.utils_cb import EvalCallback
from utils.utils_fit import fit_one_epoch

NUM_CLASSES = 2
DS_PATH = f"VOCdevkit"
BATCH_SIZE = 2
SAVED_PATH = "exp"
LOG_PATH = "log"
EPOCH = 30
INPUT_SHAPE = [512, 512]
SHUFFLE = True
#------------------------------------------------------------------#
#   建议选项：
#   种类少（几类）时，设置为True
#   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
#   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
#------------------------------------------------------------------#
dice_loss       = False
#------------------------------------------------------------------#
#   是否使用focal loss来防止正负样本不平衡
#------------------------------------------------------------------#
focal_loss      = False
NUM_WOKERS = 8
FP16 = True
Init_lr = 7e-3
Min_lr = Init_lr * 0.01
# ------------------------------------------------------------------#
#   optimizer_type  使用到的优化器种类，可选的有adam、sgd
#                   当使用Adam优化器时建议设置  Init_lr=5e-4
#                   当使用SGD优化器时建议设置   Init_lr=7e-3
#   momentum        优化器内部使用到的momentum参数
#   weight_decay    权值衰减，可防止过拟合
#                   adam会导致weight_decay错误，使用adam时建议设置为0。
# ------------------------------------------------------------------#
optimizer_type = "sgd"
momentum = 0.9
weight_decay = 1e-4
# ------------------------------------------------------------------#
#   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
# ------------------------------------------------------------------#
lr_decay_type = 'cos'

import torch
from pathlib import Path
from utils.datasets import get_train_val,get_lines
from nets.deeplabv3 import createDeepLabv3,set_optimizer_lr,get_lr_scheduler
from nets.loss import Focal_Loss, CE_Loss, Dice_loss
from utils.utils_cb import LossHistory
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np

if __name__ == "__main__":
    cls_weights = np.ones([NUM_CLASSES], np.float32)
    if FP16 and torch.cuda.is_available():
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None
    model = createDeepLabv3(NUM_CLASSES)
    gen_train, gen_val, num_train, num_val = get_train_val(DS_PATH, INPUT_SHAPE, SHUFFLE,
                                       BATCH_SIZE, NUM_WOKERS, NUM_CLASSES)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    else:
        model_train = model.train()

    train_lines,val_lines=get_lines(DS_PATH)
    eval_callback = EvalCallback(model, INPUT_SHAPE, NUM_CLASSES, val_lines, DS_PATH, LOG_PATH, torch.cuda.is_available(), \
                                 eval_flag=True, period=5)

    for param in model.backbone.parameters():
        param.requires_grad = True
    nbs = 16
    lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(BATCH_SIZE / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(BATCH_SIZE / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # ---------------------------------------#
    #   根据optimizer_type选择优化器
    # ---------------------------------------#
    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    # ---------------------------------------#
    #   获得学习率下降的公式
    # ---------------------------------------#
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, EPOCH)

    # ---------------------------------------#
    #   判断每一个世代的长度
    # ---------------------------------------#
    epoch_step = num_train // BATCH_SIZE
    epoch_step_val = num_val // BATCH_SIZE
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit,EPOCH)
    loss_his=LossHistory(LOG_PATH, model, input_shape=INPUT_SHAPE)
    for epoch in range(EPOCH):
        nbs = 16
        lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(BATCH_SIZE / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(BATCH_SIZE / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]
        epoch_step = num_train // BATCH_SIZE
        epoch_step_val = num_val // BATCH_SIZE
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(model_train, model, loss_his, eval_callback, optimizer, epoch,
                    epoch_step, epoch_step_val, gen_train, gen_val, EPOCH, torch.cuda.is_available(), dice_loss, focal_loss, cls_weights, NUM_CLASSES, FP16, scaler, 5, SAVED_PATH, 0)
    model_train.eval()
