import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self):
        super(AttentionBlock, self).__init__()

    def forward(self, x):
        # Assuming x is of shape (batch_size, channels, sequence_length)
        # Calculate attention weights
        attn_weights = torch.matmul(x, x.permute(0, 2, 1))
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=2)

        # Apply attention to input
        attended_x = torch.matmul(attn_weights, x)

        return attended_x


class CA(nn.Module):
    def __init__(self):
        super(CA, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=64, stride=1, padding='same')
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=4)
        self.norm1 = nn.BatchNorm1d(64)

        self.attention1 = AttentionBlock()

        self.conv2 = nn.Conv1d(64, 128, kernel_size=64, stride=1, padding='same')
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.norm2 = nn.BatchNorm1d(128)

        self.attention2 = AttentionBlock()

        self.conv3 = nn.Conv1d(128, 256, kernel_size=64, stride=1, padding='same')
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        self.norm3 = nn.BatchNorm1d(256)

        self.attention3 = AttentionBlock()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1792, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.norm1(x)
        x = self.attention1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.norm2(x)
        x = self.attention2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        x = self.norm3(x)
        x = self.attention3(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x
