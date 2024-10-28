import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from torchvision import models
from utils import read_conf, validation_accuracy, evaluate
from data import dentex2023

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='dentex2023')
    parser.add_argument('--gpu', '-g', default='0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str, required=True, help="Path to saved model checkpoint")
    args = parser.parse_args()

    # Load config file
    config = read_conf('conf/data/' + args.data + '.yaml')
    device = 'cuda:' + args.gpu
    data_path = config['data_root']
    batch_size = int(config['batch_size'])

    # Load test data
    if args.data == 'dentex2023':
        _, _, test_loader = dentex2023.get_data_loaders(data_dir=data_path, batch_size=batch_size, num_workers=4)

    # Initialize model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config['num_classes'])
    model.to(device)

    # Load the trained model checkpoint
    checkpoint_path = os.path.join(config['save_path'], args.save_path, 'model_best.pth.tar')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint '{checkpoint_path}'")
    else:
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    # Set model to evaluation mode
    model.eval()

    # Evaluation
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    
    evaluate(np.array(all_probs), np.array(all_targets), verbose=True)

if __name__ == '__main__':
    eval()
