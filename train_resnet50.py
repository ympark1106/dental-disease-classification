import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
from utils import read_conf, validation_accuracy
from torchvision import models
import random

from data import dentex2023
from sklearn.metrics import f1_score


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='dentex2023')
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    # parser.add_argument('--save_path', '-s', type=str)
    # parser.add_argument('--noise_rate', '-n', type=float, default=0.2)
    args = parser.parse_args()

    # config = utils.read_conf('conf/'+args.data+'.json')
    config = read_conf('conf/data/'+args.data+'.yaml')
    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    # noise_rate = args.noise_rate

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lr_decay = [int(0.5*max_epoch), int(0.75*max_epoch), int(0.9*max_epoch)]

    if args.data == 'dentex2023':
        train_loader, valid_loader, _ = dentex2023.get_data_loaders(data_dir=data_path, batch_size=batch_size, num_workers=4)


    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config['num_classes'])
    model.to(device)

    
    print(model)
    
    # exit()
    # num_params = count_trainable_params(model)
    # print(f"Number of trainable parameters: {num_params}")
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir=save_path, max_history=1)

    for epoch in range(max_epoch):
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs) 
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() 
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)
            correct += predicted.eq(targets).sum().item()

            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (total_loss / (batch_idx + 1), 100. * correct / total, correct, total), end='')
        print()

        # validation
        valid_accuracy = validation_accuracy(model, valid_loader, device, mode='resnet')
        scheduler.step()
        saver.save_checkpoint(epoch, metric=valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(
            epoch, total_loss / len(train_loader), correct / total, valid_accuracy))

        print(scheduler.get_last_lr())

    
if __name__ =='__main__':
    train()