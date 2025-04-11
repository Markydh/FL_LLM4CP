# import os
# import torch
# import torch.backends.cudnn as cudnn
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from data import Dataset_Pro
# import scipy.io as sio
# from models.GPT4CP1 import Model
# import numpy as np
# import shutil
# # from torch.utils.tensorboard import SummaryWriter
# from metrics import NMSELoss, SE_Loss

# # ============= HYPER PARAMS(Pre-Defined) ==========#
# lr = 0.001
# epochs = 500
# batch_size = 512
# device = torch.device('cuda:1')

# best_loss = 100
# save_path = "Weights/U2U_LLM4CP.pth"
# train_TDD_r_path = "./datas/TrainingDataset/H_U_his_train.mat"   # 历史数据
# train_TDD_t_path = "./datas/TrainingDataset/H_U_pre_train.mat"   # 预测数据
# key = ['H_U_his_train', 'H_U_pre_train', 'H_D_pre_train']        # 指定需要加载的数据字段
# train_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=1, is_U2D=0, is_few=0)  # 创建训练集  is_train:是否为训练集 is_U2D:是否为上行到下行  is_few：是否为小样本模式
# validate_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=0, is_U2D=0)  # 创建验证集 同上

# model = Model(gpu_id=0,
#               pred_len=4, prev_len=16,
#               UQh=1, UQv=1, BQh=1, BQv=1).to(device)
# if os.path.exists(save_path):
#     model = torch.load(save_path, map_location=device, weights_only=False)



# def save_best_checkpoint(model):  # save model function
#     model_out_path = save_path
#     torch.save(model, model_out_path)


# ###################################################################
# # ------------------- Main Train (Run second)----------------------------------
# ###################################################################
# def train(training_data_loader, validate_data_loader):
#     global epochs, best_loss
#     print('Start training...')
#     for epoch in range(epochs):
#         epoch_train_loss, epoch_val_loss = [], []
#         # ============Epoch Train=============== #
#         model.train()

#         for iteration, batch in enumerate(training_data_loader, 1):
#             pred_t, prev = Variable(batch[0]).to(device), \
#                            Variable(batch[1]).to(device)
#             optimizer.zero_grad()  # fixed
#             # 这里的输入prev的shape是（batch_size, 16, 96）
#             pred_m = model(prev, None, None, None)
#             loss = criterion(pred_m, pred_t)  # compute loss
#             epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

#             loss.backward()
#             optimizer.step()

#         #       lr_scheduler.step()  # update lr

#         t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
#         print('Epoch: {}/{} training loss: {:.7f}'.format(epoch+1, epochs, t_loss))  # print loss for each epoch

#         # ============Epoch Validate=============== #
#         model.eval()
#         with torch.no_grad():
#             for iteration, batch in enumerate(validate_data_loader, 1):
#                 pred_t, prev = Variable(batch[0]).to(device), \
#                                Variable(batch[1]).to(device)
#                 optimizer.zero_grad()  # fixed
#                 pred_m = model(prev, None, None, None)
#                 loss = criterion(pred_m, pred_t)  # compute loss
#                 epoch_val_loss.append(loss.item())  # save all losses into a vector for one epoch
#             v_loss = np.nanmean(np.array(epoch_val_loss))
#             print('validate loss: {:.7f}'.format(v_loss))
#             if v_loss < best_loss:
#                 best_loss = v_loss
#                 save_best_checkpoint(model)


# ###################################################################
# # ------------------- Main Function (Run first) -------------------
# ###################################################################
# if __name__ == "__main__":
#     total = sum([param.nelement() for param in model.parameters()])
#     print("总参数量: %.5fM" % (total / 1e6))
#     total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print("可训练的参数量: %.5fM" % (total_learn / 1e6))

#     training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
#                                       pin_memory=True,
#                                       drop_last=True)  # put training data to DataLoader for batches
#     validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size,
#                                       shuffle=True,
#                                       pin_memory=True,
#                                       drop_last=True)  # put training data to DataLoader for batches
#     optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
#     criterion = NMSELoss().to(device)
#     train(training_data_loader, validate_data_loader)  # call train function (

#     total = sum([param.nelement() for param in model.parameters()])
#     print("总参数量: %.5fM" % (total / 1e6))
#     total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print("可训练的参数量: %.5fM" % (total_learn / 1e6))

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from data import Dataset_Pro
import scipy.io as sio
from models.GPT4CP1 import Model
import numpy as np
import shutil
import logging
import sys
# from torch.utils.tensorboard import SummaryWriter
from metrics import NMSELoss, SE_Loss

# 配置日志
logging.basicConfig(
    filename='training.log',  # 日志文件路径
    level=logging.INFO,       # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式：时间 - 级别 - 消息
)

# 重定向print输出到日志文件和控制台
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("training_20.log", "a")  # 以追加模式打开日志文件

    def write(self, message):
        self.terminal.write(message)  # 输出到控制台
        self.log.write(message)       # 写入日志文件

    def flush(self):
        self.log.flush()  # 确保日志文件实时更新

sys.stdout = Logger()

# ============= HYPER PARAMS(Pre-Defined) ==========#
lr = 0.001
epochs = 500
batch_size = 512
device = torch.device('cuda:0')

best_loss = 100
save_path = "Weights/U2U_LLM4CP_20.pth"
train_TDD_r_path = "./datas/TrainingDataset/H_U_his_train.mat"   # 历史数据
train_TDD_t_path = "./datas/TrainingDataset/H_U_pre_train.mat"   # 预测数据
key = ['H_U_his_train', 'H_U_pre_train', 'H_D_pre_train']        # 指定需要加载的数据字段

# 创建完整训练集
full_train_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=1, is_U2D=0, is_few=0)

# 计算20%的数据集大小并随机选择索引
train_size = int(0.2 * len(full_train_set))  # 20%的数据量
indices = list(range(len(full_train_set)))   # 创建索引列表
np.random.shuffle(indices)                   # 随机打乱索引
train_indices = indices[:train_size]         # 选择前20%的索引

# 创建20%的训练子集
train_set = Subset(full_train_set, train_indices)

# 创建验证集（保持不变）
validate_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=0, is_U2D=0)

model = Model(gpu_id=0,
              pred_len=4, prev_len=16,
              UQh=1, UQv=1, BQh=1, BQv=1).to(device)

# if os.path.exists(save_path):
#     model = torch.load(save_path, map_location=device, weights_only=False)

def save_best_checkpoint(model):  # 保存模型函数
    model_out_path = save_path
    torch.save(model, model_out_path)

###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################
def train(training_data_loader, validate_data_loader):
    global epochs, best_loss
    logging.info('Start training...')  # 记录训练开始
    print('Start training...')  # 保留控制台输出
    for epoch in range(epochs):
        epoch_train_loss, epoch_val_loss = [], []
        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            pred_t, prev = Variable(batch[0]).to(device), \
                           Variable(batch[1]).to(device)
            optimizer.zero_grad()
            pred_m = model(prev, None, None, None)
            loss = criterion(pred_m, pred_t)  # 计算损失
            epoch_train_loss.append(loss.item())  # 保存每个batch的损失

            loss.backward()
            optimizer.step()

        t_loss = np.nanmean(np.array(epoch_train_loss))  # 计算epoch平均训练损失
        log_message = 'Epoch: {}/{} training loss: {:.7f}'.format(epoch+1, epochs, t_loss)
        logging.info(log_message)  # 记录训练损失
        print(log_message)  # 保留控制台输出

        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                pred_t, prev = Variable(batch[0]).to(device), \
                               Variable(batch[1]).to(device)
                optimizer.zero_grad()
                pred_m = model(prev, None, None, None)
                loss = criterion(pred_m, pred_t)
                epoch_val_loss.append(loss.item())
            v_loss = np.nanmean(np.array(epoch_val_loss))
            val_message = 'validate loss: {:.7f}'.format(v_loss)
            logging.info(val_message)  # 记录验证损失
            print(val_message)  # 保留控制台输出

            if v_loss < best_loss:
                best_loss = v_loss
                save_best_checkpoint(model)
                save_message = 'Best model saved with validation loss: {:.7f}'.format(v_loss)
                logging.info(save_message)  # 记录最佳模型保存信息
                print(save_message)

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    total = sum([param.nelement() for param in model.parameters()])
    print("总参数量: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("可训练的参数量: %.5fM" % (total_learn / 1e6))

    # 使用20%的训练子集创建DataLoader
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size,
                                      shuffle=True, pin_memory=True, drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
    criterion = NMSELoss().to(device)
    train(training_data_loader, validate_data_loader)  # 调用训练函数

    total = sum([param.nelement() for param in model.parameters()])
    print("总参数量: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("可训练的参数量: %.5fM" % (total_learn / 1e6))