class Net(nn.Module):
    def __init__(self, label_size):
        super().__init__()

        # Pooling Layer
        self.pool = nn.MaxPool2d(2, 2)

        # Convolutional Layer 
        # (in_channels, out_channels, kernel_size) 
        # first in channel is 1 because its greyscale 
        self.conv1 = nn.Conv2d(1, 6, 5) # (6, 350, 350) -> (6, 173, 173)
        self.conv2 = nn.Conv2d(6, 16, 5) # (6, 173, 173) -> (16, 84, 84)

        # Linear Layer
        self.fc1 = nn.Linear(16 * 84 * 84, 120) # shape of image after pooling twice (16, 84, 84)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, label_size)

    def forward(self, x):
        # Conv1 -> Pool -> Conv2 -> Pool -> Lin1 -> Lin2 -> Lin3
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x