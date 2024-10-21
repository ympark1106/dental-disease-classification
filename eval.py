import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
from utils import read_conf, validation_accuracy, evaluate

import random
import rein

import dino_variant
from data import dentex2023


def rein_forward(model, inputs):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    output = torch.softmax(output, dim=1)

    return output


def rein3_forward(model, inputs):
    f = model.forward_features1(inputs)
    f = f[:, 0, :]
    outputs1 = model.linear(f)

    f = model.forward_features2(inputs)
    f = f[:, 0, :]
    outputs2 = model.linear(f)

    f = model.forward_features3(inputs)
    f = f[:, 0, :]
    outputs3 = model.linear(f)

    outputs1 = torch.softmax(outputs1, dim=1)
    outputs2 = torch.softmax(outputs2, dim=1)
    outputs3 = torch.softmax(outputs3, dim=1)

    return (outputs1 + outputs2 + outputs3)/3


def ensemble_forward(models, inputs):
    ensemble_output = 0
    
    for model in models:
        output = model.forward_features(inputs)[:, 0, :]
        output = model.linear(output)
        output = torch.softmax(output, dim=1)
        ensemble_output += output
    
    # Average the outputs
    ensemble_output /= len(models)
    return ensemble_output


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='dentex2023')  
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--type', '-t', default= 'rein', type=str)
    args = parser.parse_args()



    config = read_conf('conf/data/'+args.data+'.yaml')

    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['data_root']
    batch_size = int(config['batch_size'])


    if not os.path.exists(save_path):
        os.mkdir(save_path)


    if args.data == 'dentex2023':
        _, _, test_loader = dentex2023.get_data_loaders(data_dir=data_path, batch_size=batch_size, num_workers=4)
        


        
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant


    model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = model.state_dict()

    if args.type == 'rein':
        model = rein.ReinsDinoVisionTransformer(
            **variant
        )
    if args.type == 'rein3':
        model = rein.ReinsDinoVisionTransformer_3_head(
            **variant,
            # token_lengths = [33, 33, 33]
            token_lengths= [100, 100, 100]
        )
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.load_state_dict(dino_state_dict, strict=False)
    model.to(device)

    state_dict = torch.load(os.path.join(save_path, 'last.pth.tar'), map_location='cpu')['state_dict']
    # state_dict = torch.load(os.path.join(save_path, 'model_best.pth.tar'), map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=True)
            
    # print(model)

    ## validation
    model.eval()
    test_accuracy = validation_accuracy(model, test_loader, device, mode=args.type)
    print('test acc:', test_accuracy)

    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(test_loader):
            # print(f"Batch {batch_idx} targets:", target)
            inputs, target = inputs.to(device), target.to(device)
            if args.type == 'rein':
                output = rein_forward(model, inputs)
                # print(output.shape)  # output shape: (batch_size, num_classes)
            elif args.type == 'rein3':
                output = rein3_forward(model, inputs)
            outputs.append(output.cpu())
            targets.append(target.cpu())
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy()
    targets = targets.astype(int)
    evaluate(outputs, targets, verbose=True)



if __name__ =='__main__':
    train()