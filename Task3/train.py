import torch
import pickle
from torch.utils.data import DataLoader
from model.SnliDataSet import SnliDataSet

from model.esim import ESIM

import time
import torch

import torch.nn as nn

from tqdm import tqdm
from model.utils import correct_predictions

worddict_dir="data\\worddict.txt"
data_train_id_dir="data\\train_data_id.pkl"
data_dev_id_dir="data\\dev_data_id.pkl"
embedding_matrix_dir="data\\embedding_matrix.pkl"
model_train_dir="saved_model\\train_model_"



#超参数
batch_size=512
use_gpu=True
patience=5
hidden_size=50
dropout=0.5
num_classes=3
lr=0.0004
epochs=100
max_grad_norm=10.0
device=torch.device("cuda:0" if use_gpu else "cpu")

def getCorrectNum(probs, targets):
    _, out_classes = probs.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def train(model, data_loader, optimizer, criterion, max_gradient_norm):
    model.train()
    device = model.device

    time_epoch_start = time.time()
    running_loss = 0
    correct_cnt = 0
    batch_cnt = 0

    for index, batch in enumerate(data_loader):
        time_batch_start = time.time()
        # 从data_loader中取出数据
        premises = batch["premises"].to(device)
        premises_len = batch["premises_len"].to(device)
        hypothesis = batch["hypothesis"].to(device)
        hypothesis_len = batch["hypothesis_len"].to(device)
        labels = batch["labels"].to(device)

        # 梯度置0
        optimizer.zero_grad()

        # 正向传播
        logits, probs = model(premises, premises_len, hypothesis, hypothesis_len)

        # 求损失，反向传播，梯度裁剪，更新权重
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        running_loss += loss.item()
        correct_cnt += getCorrectNum(probs, labels)
        batch_cnt += 1
        print("Training  ------>   Batch count: {:d}/{:d},  batch time: {:.4f}s,  batch average loss: {:.4f}"
              .format(batch_cnt, len(data_loader), time.time() - time_batch_start, running_loss / (index + 1)))

    epoch_time = time.time() - time_epoch_start
    epoch_loss = running_loss / len(data_loader)
    epoch_accuracy = correct_cnt / len(data_loader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, data_loader, criterion):
    model.eval()
    device = model.device

    time_epoch_start = time.time()
    running_loss = 0
    correct_cnt = 0
    batch_cnt = 0

    for index, batch in enumerate(data_loader):
        time_batch_start = time.time()
        # 从data_loader中取出数据
        premises = batch["premises"].to(device)
        premises_len = batch["premises_len"].to(device)
        hypothesis = batch["hypothesis"].to(device)
        hypothesis_len = batch["hypothesis_len"].to(device)
        labels = batch["labels"].to(device)

        # 正向传播
        logits, probs = model(premises, premises_len, hypothesis, hypothesis_len)

        # 求损失
        loss = criterion(logits, labels)

        running_loss += loss.item()
        correct_cnt += getCorrectNum(probs, labels)
        batch_cnt += 1
        print("Testing  ------>   Batch count: {:d}/{:d},  batch time: {:.4f}s,  batch average loss: {:.4f}"
              .format(batch_cnt, len(data_loader), time.time() - time_batch_start, running_loss / (index + 1)))

    epoch_time = time.time() - time_epoch_start
    epoch_loss = running_loss / len(data_loader)
    epoch_accuracy = correct_cnt / len(data_loader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy

if __name__ == '__main__':
    # 加载数据
    with open(data_train_id_dir, 'rb') as f:
        train_data = SnliDataSet(pickle.load(f), max_premises_len=None, max_hypothesis_len=None)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    with open(data_dev_id_dir, 'rb') as f:
        dev_data = SnliDataSet(pickle.load(f), max_premises_len=None, max_hypothesis_len=None)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)

    # 加载embedding
    with open(embedding_matrix_dir, 'rb') as f:
        embeddings = torch.tensor(pickle.load(f), dtype=torch.float).to(device)

    model = ESIM(embeddings.shape[0],
                 embeddings.shape[1],
                 hidden_size,
                 embeddings=embeddings,
                 dropout=dropout,
                 num_classes=num_classes,
                 device=device).to(device)
    # 准备训练
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=0)

    # 训练过程中的参数
    best_score = 0.0
    train_losses = []
    valid_losses = []
    patience_cnt = 0

    for epoch in range(epochs):
        # 训练
        print("-" * 50, "Training epoch %d" % (epoch), "-" * 50)
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, criterion, max_grad_norm)
        train_losses.append(epoch_loss)
        print("Training time: {:.4f}s, loss :{:.4f}, accuracy: {:.4f}%".format(epoch_time, epoch_loss,
                                                                               (epoch_accuracy * 100)))

        # 验证
        print("-" * 50, "Validating epoch %d" % (epoch), "-" * 50)
        epoch_time_dev, epoch_loss_dev, epoch_accuracy_dev = validate(model, dev_loader, criterion)
        valid_losses.append(epoch_loss_dev)
        print("Validating time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n".format(epoch_time_dev, epoch_loss_dev,
                                                                                   (epoch_accuracy_dev * 100)))

        # 更新学习率
        scheduler.step(epoch_accuracy)

        # early stoping
        if epoch_accuracy_dev < best_score:
            patience_cnt += 1
        else:
            best_score = epoch_accuracy_dev
            patience_cnt = 0
        if patience_cnt >= patience:
            print("-" * 50, "Early stopping", "-" * 50)
            break

        # 每个epoch都保存模型
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                   model_train_dir + str(epoch) + ".dir")