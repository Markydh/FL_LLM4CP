import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from peft import LoraConfig, get_peft_model
from data import Dataset_Pro
import scipy.io as sio
from models.GPT4CP import Model 
import numpy as np
import shutil
import logging
from metrics import NMSELoss, SE_Loss

# 设置日志文件
log_file = "fl_training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# ============= HYPER PARAMS(Pre-Defined) ==========#
lr = 0.001
epochs = 500
batch_size = 512
device = torch.device('cuda:0')
num_client = 5

best_loss = 100
train_TDD_r_path = "./datas/TrainingDataset/H_U_his_train.mat"
train_TDD_t_path = "./datas/TrainingDataset/H_U_pre_train.mat"

# ============= 数据集加载 ==========#
train_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=1, is_U2D=0, is_few=0)
validate_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=0, is_U2D=0)

# ============= LoRA 配置 ==========#
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"], # GPT-2 内部的注意力层模块
    lora_dropout=0.05,
    bias="none",
    task_type="FEATURE_EXTRACTION" # 改为特征提取器任务类型，避免生成任务假设
)

# ============= 初始化全局模型 ==========#
model_global = Model(
    gpu_id=0,
    pred_len=4, prev_len=16, 
    UQh=1, UQv=1, BQh=1, BQv=1
).to(device)
model_global.gpt2 = get_peft_model(model_global.gpt2, lora_config)

# 计算并记录训练前的参数量
logging.info("训练开始前全局模型参数统计：")
total_params = sum([param.nelement() for param in model_global.parameters()])
logging.info(f"总参数量: {total_params / 1e6:.5f}M")
trainable_params = sum(p.numel() for p in model_global.parameters() if p.requires_grad)
logging.info(f"可训练参数量: {trainable_params / 1e6:.5f}M")

# ============= 初始化客户端模型 ==========#
model_list = []
for _ in range(num_client):
    base_model = Model(
        gpu_id=0,
        pred_len=4, prev_len=16, 
        UQh=1, UQv=1, BQh=1, BQv=1
    ).to(device)
    base_model.gpt2 = get_peft_model(base_model.gpt2, lora_config)
    model_list.append(base_model)

# ============= 数据分割为客户端 ==========#
client_data_sizes = [len(train_set) // num_client] * num_client
client_data_sizes[-1] += len(train_set) % num_client
client_datasets = random_split(train_set, client_data_sizes)

training_data_loader_list = [
    DataLoader(
        dataset=client_datasets[i],
        num_workers=0,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    ) for i in range(num_client)
]

validate_data_loader = DataLoader(
    dataset=validate_set,
    num_workers=0,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    drop_last=True
)

# ============= 优化器和损失函数 ==========#
optimizer_list = [optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001) for model in model_list]
optimizer_global = optim.Adam(model_global.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
criterion = NMSELoss().to(device)

# ============= 训练函数 ==========#
def train(training_data_loader_list, validate_data_loader, model_list, optimizer_list, model_global, optimizer_global):
    global epochs, best_loss
    logging.info('Start training...')
    for epoch in range(epochs):
        epoch_train_loss, epoch_val_loss = [], []
        
        # 每个客户端独立训练
        for client_idx in range(num_client):
            model = model_list[client_idx]
            optimizer = optimizer_list[client_idx]
            model.train()
            
            # 复制全局模型参数到客户端模型
            model.load_state_dict(model_global.state_dict())
            
            for iteration, batch in enumerate(training_data_loader_list[client_idx], 1):
                pred_t, prev = Variable(batch[0]).to(device), Variable(batch[1]).to(device)
                optimizer.zero_grad()
                pred_m = model(prev, None, None, None)  
                loss = criterion(pred_m, pred_t)
                epoch_train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                
                # 聚合梯度到全局模型
                for global_param, client_param in zip(model_global.parameters(), model.parameters()):
                    if client_param.grad is not None:
                        if global_param.grad is None:
                            global_param.grad = client_param.grad.clone() / num_client
                        else:
                            global_param.grad += client_param.grad.clone() / num_client
            
            t_loss = np.nanmean(np.array(epoch_train_loss))
            logging.info(f'Epoch: {epoch+1}/{epochs} Client: {client_idx+1} training loss: {t_loss:.7f}')
        
        # 更新全局模型
        optimizer_global.step()
        optimizer_global.zero_grad()
        
        # 使用全局模型进行验证
        model_global.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                pred_t, prev = Variable(batch[0]).to(device), Variable(batch[1]).to(device)
                pred_m = model_global(prev, None, None, None)  
                loss = criterion(pred_m, pred_t)
                epoch_val_loss.append(loss.item())
            v_loss = np.nanmean(np.array(epoch_val_loss))
            logging.info(f'validate loss: {v_loss:.7f}')
            
            if v_loss < best_loss:
                best_loss = v_loss
                torch.save(model_global.state_dict(), "Weights/FL_LLM4CP-1/U2U_LLM4CP_global.pth")
                logging.info(f"Best global model saved with validation loss: {best_loss:.7f}")

if __name__ == "__main__":
    train(training_data_loader_list, validate_data_loader, model_list, optimizer_list, model_global, optimizer_global)
