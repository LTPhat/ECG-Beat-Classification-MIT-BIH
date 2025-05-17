import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)



class CNN_LSTM_Net(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN_LSTM_Net, self).__init__()

        # CNN Block: (batch, 1, 250) → (batch, 64, ~31)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),  # (batch, 32, 250)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=2),  # (batch, 32, 125)

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),  # (batch, 64, 125)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=2),  # (batch, 64, 62)

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),  # (batch, 128, 62)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=2),  # (batch, 128, 31)
        )

        # LSTM input: sequence of 31 timesteps, each with 128 features
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1,
                            batch_first=True, bidirectional=True)

        # Classification: bidirectional → 64*2 = 128
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes), 
            nn.Softmax(dim=1)  # (batch, num_classes)   
        )
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    def forward(self, x):
        # x: (batch, 250)
        x = x.unsqueeze(1)  # → (batch, 1, 250)
        features = self.feature_extractor(x)  # → (batch, 128, 31)

        # Transpose for LSTM: (batch, seq_len=31, input_size=128)
        features = features.permute(0, 2, 1)

        out, _ = self.lstm(features)  # → (batch, 31, 128)

        # # Use last time step
        out = out[:, -1, :]  # → (batch, 128)
        # out = out.reshape(out.size(0), -1)  # Flatten: → (batch, 31*128)

        out = self.classifier(out)  # → (batch, num_classes)
        return out
############# RESNET 1D #############


class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet1DBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetClassifier(nn.Module):
    def __init__(self, input_size=210, num_classes=5):
        super(ResNetClassifier, self).__init__()
        
        # ResNet1D Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            ResNet1DBlock(64, 64),
            ResNet1DBlock(64, 64),
            
            ResNet1DBlock(64, 128, stride=2),
            ResNet1DBlock(128, 128),
        )
        
        # LSTM Decoder
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Input shape: (batch, 210)
        # Add channel dimension: (batch, 1, 210)
        x = x.unsqueeze(1)
        
        # Encoder: ResNet1D
        # Output shape: (batch, 256, seq_len)
        x = self.encoder(x)
        
        # # Prepare for LSTM: (batch, seq_len, 256)
        x = x.permute(0, 2, 1)
        
        # # LSTM Decoder
        # # Output shape: (batch, seq_len, 128)
        x, _ = self.lstm(x)
        
        # # # Take the last time step
        x = x[:, -1, :]
        
        # # # Classification
        # Output shape: (batch, 5)
        x = self.fc(x)
        
        return x


if __name__ == "__main__":
    import numpy as np
    model = ResNetClassifier(input_size=210, num_classes=5)
    input_data = np.load("data/ecg_classification/processed/train_data.npy")
    input_tensor = torch.tensor(input_data[:32], dtype=torch.float32)
    print("input_tensor shape:", input_tensor.shape)
    out = model(input_tensor)
    print("output shape:", out.shape)
    print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))