import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-3, device='cuda', save_path='./logs_aug', class_names=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_save_path = os.path.join(save_path, 'best_model.pth')

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    # Initialize metrics history
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # --------------------
        # TRAINING PHASE
        # --------------------
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        train_pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Training")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            train_pbar.set_postfix(loss=loss.item(), acc=100 * correct_train / total_train)

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # --------------------
        # VALIDATION PHASE
        # --------------------
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Train Acc={train_acc:.2f}%, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")

        # --------------------
        # CONFUSION MATRIX & CLASSIFICATION REPORT
        # --------------------
        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix:")
        print(cm)

        plt.figure(figsize=(8, 6))  # Optional: adjust figure size
        annot_fmt = [[f"{cell}" for cell in row] for row in cm]

        sns.heatmap(cm, annot=annot_fmt, fmt="", cmap="Blues")  # Set fmt="" when using custom annot
        plt.title(f"Confusion Matrix - Augmentation - Epoch {epoch+1}")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(save_path, f'aug_confusion_matrix_epoch_{epoch+1}.png'))
        plt.close()

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names if class_names else None))

        # --------------------
        # SAVE BEST MODEL
        # --------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Best model saved at epoch {epoch+1} with Val Acc={val_acc:.2f}%")

    # --------------------
    # PLOT TRAINING HISTORY
    # --------------------
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Acc')
    plt.plot(epochs_range, val_accuracies, label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()


def test_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    total_test = 0
    correct_test = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)
    test_acc = 100 * correct_test / total_test
    print(f"Test Accuracy: {test_acc:.2f}%")
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    return all_preds, all_labels


