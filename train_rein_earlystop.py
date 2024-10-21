import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")


import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TORCH_USE_CUDA_DSA"] = '1'
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np

import random
import rein
from utils import read_conf, validation_accuracy
from data import dentex2023

import dino_variant
from sklearn.metrics import f1_score



def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_requires_grad(model, layers_to_train):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_train):
            param.requires_grad = True
        else:
            param.requires_grad = False

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='dentex2023')
    parser.add_argument('--gpu', '-g', default='0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    args = parser.parse_args()

    config = read_conf('conf/data/' + args.data + '.yaml')
    device = 'cuda:' + args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    
    patience = 10  # Early stopping patience
    best_val_acc = 0
    early_stop_counter = 0  # Counter for early stopping

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lr_decay = [int(0.5 * max_epoch), int(0.75 * max_epoch), int(0.9 * max_epoch)]

    if args.data == 'dentex2023':
        train_loader, valid_loader, _ = dentex2023.get_data_loaders(data_dir=data_path, batch_size=batch_size, num_workers=4)

    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    elif args.netsize == 'l':
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant

    model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = model.state_dict()

    model = rein.ReinsDinoVisionTransformer(**variant)
    set_requires_grad(model, ["reins", "linear"])
    model.load_state_dict(dino_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.to(device)

    num_params = count_trainable_params(model)
    print(f"Number of trainable parameters: {num_params}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir=save_path, max_history=1)

    avg_accuracy = 0.0
    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            targets = targets.type(torch.LongTensor)
            inputs, targets = inputs.to(device), targets.to(device)

            if targets.ndim > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)

            optimizer.zero_grad()

            features = model.forward_features(inputs)
            features = features[:, 0, :]
            outputs = model.linear(features)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)
            correct += predicted.eq(targets).sum().item()
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss / (batch_idx + 1), 100. * correct / total, correct, total), end='')
            train_accuracy = correct / total

        train_avg_loss = total_loss / len(train_loader)
        print()

        ## validation
        model.eval()
        valid_accuracy = validation_accuracy(model, valid_loader, device)
        
        # Check if validation accuracy improved
        if valid_accuracy > best_val_acc:
            best_val_acc = valid_accuracy
            early_stop_counter = 0  # Reset early stopping counter if performance improves
        else:
            early_stop_counter += 1  # Increment if no improvement

        # Check for early stopping
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        if epoch >= max_epoch - 10:
            avg_accuracy += valid_accuracy

        scheduler.step()
        saver.save_checkpoint(epoch, metric=valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())

    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy / 10))

if __name__ == '__main__':
    train()
