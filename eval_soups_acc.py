import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
import torch
import torch.nn as nn
import argparse
import numpy as np

from utils import read_conf, validation_accuracy, ModelWithTemperature, validate, evaluate, calculate_ece, calculate_nll
import dino_variant
from data import dentex2023
import rein

# Model forward function
def rein_forward(model, inputs, temp_scaler=None):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    if temp_scaler:
        output = temp_scaler.temperature_scale(output)
    output = torch.softmax(output, dim=1)
    return output

# Data loader setup
def setup_data_loaders(args, data_path, batch_size):
    # Load test data
    if args.data == 'dentex2023':
        _, valid_loader, test_loader = dentex2023.get_data_loaders(data_dir=data_path, batch_size=batch_size, num_workers=4)
        
    return test_loader, valid_loader


# Model initialization
def initialize_models(save_paths, variant, config, device):
    models = []
    for save_path in save_paths:
        model = rein.ReinsDinoVisionTransformer(**variant)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        state_dict = torch.load(os.path.join(save_path, 'last.pth.tar'), map_location='cpu')['state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        models.append(model)
    return models

# Validation and test accuracy calculation
def validate_model(model, valid_loader, device, mode):
    return validation_accuracy(model, valid_loader, device, mode=mode)

# Sort models by accuracy
def sort_models_by_accuracy(models, valid_loader, device, mode):
    model_accuracies = [(model, validate_model(model, valid_loader, device, mode)) for model in models]
    sorted_models = sorted(model_accuracies, key=lambda x: x[1], reverse=True)  # Sort by accuracy (descending)
    return sorted_models

# Greedy soup ensemble function
def greedy_soup_ensemble(models, valid_loader, device):
    # Evaluate and sort models by validation accuracy
    model_accuracies = [(model, validation_accuracy(model, valid_loader, device)) for model in models]
    sorted_models = sorted(model_accuracies, key=lambda x: x[1], reverse=True)
    
    for model, accuracy in sorted_models:
        print(f'Validation accuracy: {accuracy}')
    
    # Initialize greedy soup with the highest-performing model
    max_accuracy = sorted_models[0][1]
    greedy_soup_params = sorted_models[0][0].state_dict()  # Best model's initial parameters
    best_params = {k: v.clone() for k, v in greedy_soup_params.items()}  # Keep a copy of best parameters
    greedy_soup_ingredients = [sorted_models[0][0]] 

    for i in range(1, len(sorted_models)):
        print(f'Testing model {i} of {len(sorted_models)}')
        
        # New model parameters to test as an additional ingredient
        new_ingredient_params = sorted_models[i][0].state_dict()
        num_ingredients = len(greedy_soup_ingredients)
        
        # Create potential new soup parameters by averaging with the new ingredient
        potential_greedy_soup_params = {
            k: greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1)) +
               new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
            for k in new_ingredient_params
        }
        
        # Load the new potential parameters into the base model for evaluation
        sorted_models[0][0].load_state_dict(potential_greedy_soup_params)
        sorted_models[0][0].eval()
        
        # Calculate validation accuracy with the potential new soup parameters
        held_out_val_accuracy = validation_accuracy(sorted_models[0][0], valid_loader, device)
        print(f'Held-out validation accuracy: {held_out_val_accuracy}')
        
        # Update greedy soup if accuracy improves, otherwise revert to original parameters
        if held_out_val_accuracy > max_accuracy:
            greedy_soup_ingredients.append(sorted_models[i][0])
            max_accuracy = held_out_val_accuracy
            greedy_soup_params = potential_greedy_soup_params  # Save the improved parameters
            best_params = {k: v.clone() for k, v in potential_greedy_soup_params.items()}  # Update best params
            print(f'New greedy soup ingredient added. Number of ingredients: {len(greedy_soup_ingredients)}\n')
        else:
            # Revert to the best-known parameters if the new ingredient didn’t improve accuracy
            sorted_models[0][0].load_state_dict(best_params)  # Revert to best params
            print(f'No improvement. Reverting to best-known parameters.\n')

    return best_params, sorted_models[0][0]


# Final evaluation
def evaluate_model(model, test_loader, device):
    outputs, targets = [], []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            output = rein_forward(model, inputs)
            outputs.append(output.cpu())
            targets.append(target.cpu())

    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy().astype(int)
    evaluate(outputs, targets, verbose=True)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='dentex2023')
    parser.add_argument('--gpu', '-g', default='0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--type', '-t', default='rein', type=str)
    args = parser.parse_args()

    config = read_conf(os.path.join('conf', 'data', f'{args.data}.yaml'))
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    
    save_paths = [
        os.path.join(config['save_path'], 'reins_ce1'),
        os.path.join(config['save_path'], 'reins_ce2'),
        os.path.join(config['save_path'], 'reins_ce3'),
        os.path.join(config['save_path'], 'reins_ce4'),
        
        # os.path.join(config['save_path'], 'reins_focal1'),
        # os.path.join(config['save_path'], 'reins_focal2'),
        # os.path.join(config['save_path'], 'reins_focal3'),
        # os.path.join(config['save_path'], 'reins_focal4'),
        # os.path.join(config['save_path'], 'reins_focal5'),
        
        # os.path.join(config['save_path'], 'reins_adafocal1'),
        # os.path.join(config['save_path'], 'reins_adafocal2'),
        # os.path.join(config['save_path'], 'reins_adafocal3'),
        # os.path.join(config['save_path'], 'reins_adafocal4'),
    ]

    variant = dino_variant._small_variant
    models = initialize_models(save_paths, variant, config, device)
    test_loader, valid_loader = setup_data_loaders(args, data_path, batch_size)
    
    # Step 1: Compute greedy soup parameters
    greedy_soup_params, model1 = greedy_soup_ensemble(models, valid_loader, device)

    # Evaluate the final model on the test set
    model1.load_state_dict(greedy_soup_params)
    model1.eval()
    test_accuracy = validation_accuracy(model1, test_loader, device, mode=args.type)
    print('Test accuracy:', test_accuracy)

    outputs, targets = [], []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            output = rein_forward(model1, inputs)
            outputs.append(output.cpu())
            targets.append(target.cpu())
    
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy().astype(int)
    evaluate(outputs, targets, verbose=True)

if __name__ == '__main__':
    train()