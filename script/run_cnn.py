#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import time
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from model.Tdcnn import Conv2DRegressionModel
import pandas as pd
from datasets import MyDataset
import logging
import json

from datetime import datetime
# Configuration dictionary to hold hyperparameters and settings
config = {
    'exp_name': 'CdPrediction_DrivAerNet_r2_100epochs_5k',
    'cuda': True,
    'seed': 1,
    'num_points': 5000,
    'lr': 0.001,
    'batch_size': 64,
    'epochs': 200,
    'dropout': 0.0,
    'emb_dims': 512,
    'optimizer': 'adam',
    'log' : 'best',
    'output_channels': 1,
    'dataset_path': r'data/Drivernet_execldata_90du_100se_np_max_random',  # Update this with your dataset path
    'aero_coeff': r'data/DrivAerNetPlusPlus_Drag_8k.csv',
    'subset_dir': r'train_val_test_splits'
}

# Set the device for training
device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")

def setup_seed(seed: int):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True  # For reproducibility
    torch.backends.cudnn.benchmark = False  # Set to False to ensure deterministic behavior
def r2_score(output, target):
    """Compute R-squared score."""
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def setup_logging(log_dir, exp_name):
    log_file = os.path.join(log_dir, f'{exp_name}.log')
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def initialize_model(config: dict) -> torch.nn.Module:


    # Instantiate the RegDGCNN model with the specified configuration parameters
    model = Conv2DRegressionModel(dropout=config['dropout']).to(device)

    # If CUDA is enabled and more than one GPU is available, wrap the model in a DataParallel module
    # to enable parallel computation across multiple GPUs. Specifically, use GPUs with IDs 0, 1, 2, and 3.
    if config['cuda'] and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    # Return the initialized model
    return model


def get_dataloaders(dataset_path: str, aero_coeff: str, subset_dir: str, batch_size: int) -> tuple:


    # Initialize the full dataset
    full_dataset = MyDataset(root_dir=dataset_path,num_points=8000, csv_file=aero_coeff, take= -1)

    # Helper function to create subsets from IDs in text files
    def create_subset(dataset, ids_file):
        try:
            with open(os.path.join(subset_dir, ids_file), 'r') as file:
                subset_ids = file.read().split()
            # Filter the dataset DataFrame based on subset IDs
            subset_indices = dataset.data_frame[dataset.data_frame['Design'].isin(subset_ids)].index.tolist()
            return Subset(dataset, subset_indices)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading subset file {ids_file}: {e}")

    # Create each subset using the corresponding subset file
    train_dataset = create_subset(full_dataset, 'train_design_ids.txt')
    val_dataset = create_subset(full_dataset, 'val_design_ids.txt')
    test_dataset = create_subset(full_dataset, 'test_design_ids.txt')

    # Initialize DataLoaders for each subset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader


def train_and_evaluate(model: torch.nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, config: dict,timestamp):

    # 使用时间戳创建新的文件夹

    folder_path = os.path.join(f'logs/{config["log"]}', timestamp)
    os.makedirs(folder_path, exist_ok=True)

    # 设置日志文件保存路径
    setup_logging(folder_path, config['exp_name'])

    # 保存配置到 json 文件中
    config_path = os.path.join(folder_path, f'{config["exp_name"]}_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f"Config saved to {config_path}")

    logging.info("Starting training...")
    train_losses, val_losses = [], []
    training_start_time = time.time()  # Start timing for training

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4) if config['optimizer'] == 'adam' else optim.SGD(
        model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)

    # Initialize the learning rate scheduler (ReduceLROnPlateau) to reduce the learning rate based on validation loss
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.1)
    
    best_mse = float('inf')  # Initialize the best MSE as infinity
    l1_lambda = 0.01
    # Training loop over the specified number of epochs
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()  # Start timing for this epoch
        model.train()  # Set the model to training mode
        total_loss, total_r2 = 0, 0

        # Iterate over the training data 
        for data, targets in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']} [Training]"):
            data, targets = data.to(device), targets.to(device).squeeze()  # Move data to the GPU
            optimizer.zero_grad()

            outputs = model(data)
            loss = F.mse_loss(outputs.squeeze(), targets)

            # # 加入正则化
            # l1_loss = 0
            # for param in model.parameters():
            #     l1_loss += torch.sum(torch.abs(param))
            # loss = mse_loss + l1_lambda * l1_loss
            



            r2 = r2_score(outputs.squeeze(), targets)  # Compute R2 score

            loss.backward()
            optimizer.step()
            total_loss += loss.item()  # Accumulate the loss
            total_r2 += r2.item()

        epoch_duration = time.time() - epoch_start_time
        avg_loss = total_loss / len(train_dataloader)
        avg_r2 = total_r2 / len(train_dataloader)
        train_losses.append(avg_loss)

        logging.info(f"Epoch {epoch+1}: Training Loss: {avg_loss:.6f}, R2: {avg_r2:.4f}, Time: {epoch_duration:.2f}s")

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss, val_r2 = 0, 0
        inference_times = []

        with torch.no_grad():
            for data, targets in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']} [Validation]"):
                inference_start_time = time.time()
                data, targets = data.to(device), targets.to(device).squeeze()
                outputs = model(data)
                loss = F.mse_loss(outputs.squeeze(), targets)
                val_loss += loss.item()
                r2 = r2_score(outputs.squeeze(), targets)  # Compute R2 score
                val_r2 += r2.item()
                inference_duration = time.time() - inference_start_time
                inference_times.append(inference_duration)

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_r2 = val_r2 / len(val_dataloader)
        avg_inference_time = sum(inference_times) / len(inference_times)
        val_losses.append(avg_val_loss)

        logging.info(f"Epoch {epoch+1}: Validation Loss: {avg_val_loss:.6f}, R2: {avg_val_r2:.4f}, Avg Inference Time: {avg_inference_time:.4f}s")

        # Check if this is the best model based on MSE
        if avg_val_loss < best_mse:
            best_mse = avg_val_loss
            best_model_path = os.path.join(folder_path, f'{config["exp_name"]}_best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved at epoch {epoch+1} with Validation MSE: {best_mse:.6f}, R2: {avg_val_r2:.4f}")

        # Step the scheduler based on the validation loss
        scheduler.step(avg_val_loss)

    training_duration = time.time() - training_start_time
    logging.info(f"Total training time: {training_duration:.2f}s")

    # Save the final model state to disk
    final_model_path = os.path.join(folder_path, f'{config["exp_name"]}_final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")

    # Save losses for plotting
    np.save(os.path.join(folder_path, f'{config["exp_name"]}_train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(folder_path, f'{config["exp_name"]}_val_losses.npy'), np.array(val_losses))

    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config['epochs']+1), train_losses, label='Training Loss')
    plt.plot(range(1, config['epochs']+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Convergence')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 显示图像
    plt.show()

def test_model(model: torch.nn.Module, test_dataloader: DataLoader, config: dict):
    """
    Test the model using the provided test DataLoader and calculate different metrics,
    including MSLE, and plot the predicted vs actual values as two line plots.

    Args:
        model (torch.nn.Module): The trained model to be tested.
        test_dataloader (DataLoader): DataLoader for the test set.
        config (dict): Configuration dictionary containing model settings.
    """
    model.eval()  # Set the model to evaluation mode
    total_mse, total_mae, total_r2, total_msle = 0, 0, 0, 0  # Add MSLE to total metrics
    max_mae = 0
    total_inference_time = 0  # To track total inference time
    total_samples = 0  # To count the total number of samples processed
    
    # Collect predictions and actuals for plotting
    all_predictions = []
    all_targets = []
    accruce = []
    
    # Disable gradient calculation
    with torch.no_grad():
        for data, targets in test_dataloader:
            start_time = time.time()  # Start time for inference

            data, targets = data.to(device), targets.to(device).squeeze()
            outputs = model(data)

            end_time = time.time()  # End time for inference
            inference_time = end_time - start_time
            total_inference_time += inference_time  # Accumulate total inference time

            mse = F.mse_loss(outputs.squeeze(), targets)  # Mean Squared Error (MSE)
            mae = F.l1_loss(outputs.squeeze(), targets)  # Mean Absolute Error (MAE)
            r2 = r2_score(outputs.squeeze().cpu(), targets.cpu())  # R-squared
            
            # Compute MSLE (using logarithms)
            msle = F.mse_loss(torch.log1p(outputs.squeeze()), torch.log1p(targets))  # MSLE

            # Accumulate metrics to compute averages later
            total_mse += mse.item()
            total_mae += mae.item()
            total_r2 += r2
            total_msle += msle.item()  # Accumulate MSLE
            max_mae = max(max_mae, mae.item())
            total_samples += targets.size(0)  # Increment total sample count

            # Append predictions and targets for plotting
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Compute average metrics over the entire test set
    avg_mse = total_mse / len(test_dataloader)
    avg_mae = total_mae / len(test_dataloader)
    avg_r2 = total_r2 / len(test_dataloader)
    avg_msle = total_msle / len(test_dataloader)  # Compute average MSLE

    # 将列表转换为 NumPy 数组
    predictions_array = np.array(all_predictions)
    targets_array = np.array(all_targets)
    predictions_array = np.squeeze(predictions_array)

    # 计算准确率
    accruce = 1 - np.abs(predictions_array - targets_array) / targets_array

    # 计算平均准确率
    accruce = np.mean(accruce)

    # Output test results
    print(f"Test MSE: {avg_mse:.6f}, Test MAE: {avg_mae:.6f}, Max MAE: {max_mae:.6f}, Test R2: {avg_r2:.4f}, Test MSLE: {avg_msle:.6f}, Accuracy: {accruce:.4f}")
    print(f"Total inference time: {total_inference_time:.2f}s for {total_samples} samples")

    # Convert lists of predictions and targets to numpy arrays
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # Randomly select 200 samples for plotting
    sample_size = 200
    if len(all_targets) > sample_size:
        random_indices = np.random.choice(len(all_targets), sample_size, replace=False)
        selected_predictions = all_predictions[random_indices]
        selected_targets = all_targets[random_indices]
    else:
        selected_predictions = all_predictions
        selected_targets = all_targets


    # Plot the predictions vs actual values as two line plots
    # plt.figure(figsize=(10, 6))
    # plt.plot(selected_targets, label='Actual Values', color='blue', linestyle='-', linewidth=2)
    # plt.plot(selected_predictions, label='Predicted Values', color='orange', linestyle='--', linewidth=2)
    # plt.xlabel('Sample Index')
    # plt.ylabel('Values')
    # plt.title('Predicted vs Actual Comparison (Random 200 Samples)')
    # plt.legend()
    # plt.show()

 
def load_and_test_model(model_path, test_dataloader, device):
    """Load a saved model and test it."""
    model =Conv2DRegressionModel().to(device)  # Initialize a new model instance
    # model = torch.nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(torch.load(model_path, weights_only=True))  # Load the saved weights

    test_model(model, test_dataloader, config)

if __name__ == "__main__":
    setup_seed(config['seed'])
    model = initialize_model(config).to(device)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config['dataset_path'],config['aero_coeff'],
                                                          config['subset_dir'],
                                                          config['batch_size'])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # timestamp = 20241005_231142

    train_and_evaluate(model, train_dataloader, val_dataloader, config,timestamp)

    # Load and test both the best and final models
    final_model_path = os.path.join(f'logs/{config["log"]}/{timestamp}', f'{config["exp_name"]}_final_model.pth')
    # final_model_path = os.path.join(f'logs/lr_dropout/20241005_231142', f'{config["exp_name"]}_final_model.pth')

    print("Testing the final model:")
    load_and_test_model(final_model_path, test_dataloader, device)

    best_model_path = os.path.join(f'logs/{config["log"]}/{timestamp}', f'{config["exp_name"]}_best_model.pth')
    # best_model_path = os.path.join(f'logs/lr_dropout/20241005_231142', f'{config["exp_name"]}_best_model.pth')


    print("Testing the best model:")
    load_and_test_model(best_model_path, test_dataloader, device)
