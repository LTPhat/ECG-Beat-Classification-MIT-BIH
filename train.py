import torch
import os
import numpy as np
from torch import nn
from networks.data_loader import ECGDataset
from networks.model import CNN_LSTM_Net, ResNetClassifier
from networks.trainer import train_model, test_model


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Load data
    # train_data = ECGDataset(data_path='data/ecg_classification/processed/train_data.npy', labels_path='data/ecg_classification/processed/train_labels.npy')
    test_data = ECGDataset(data_path='data/ecg_classification/processed/test_data.npy', labels_path='data/ecg_classification/processed/test_labels.npy')
    train_data = ECGDataset(data_path='data/ecg_classification/processed/train_data_aug.npy', labels_path='data/ecg_classification/processed/train_labels_aug.npy')
    # #   
    print(f"Train data shape: {train_data.data.shape}, Train labels shape: {train_data.labels.shape}")
    print(f"Validation data shape: {test_data.data.shape}, Validation labels shape: {test_data.labels.shape}")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    # # # Initialize model
    # model = CNN_LSTM_Net(num_classes=5).to(device)
    model = ResNetClassifier(input_size=train_data.data.shape[1], num_classes=5).to(device)

    # # # Train model
    train_model(model, train_loader, test_loader, num_epochs=10, lr=1e-3, device=device, save_path='./logs')
    
    # # Load the best model
    
    model.load_state_dict(torch.load('./logs/best_model.pth'))
    # # Test model
    test_model(model, test_loader, device=device)


if __name__ == "__main__":
    main()