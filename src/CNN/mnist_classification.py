import torch
import torch.nn as nn
import torch.optim as optim
from CNN import Conv2D, Pool2D, Flatten, ReLU
from engine import Engine, DataLoader

from torchvision import datasets, transforms

# Small model due to the loop-based implementation
class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Conv2D(1, 4, kernel_size=(5, 5), padding=2),
            ReLU(),
            Pool2D(pool_size=(2, 2), stride=2),           
            Conv2D(4, 8, kernel_size=(5, 5)),           
            ReLU(),
            Pool2D(pool_size=(2, 2), stride=2),           
            Flatten(),                                     
            nn.Linear(8 * 5 * 5, 10)                             
        )
    def forward(self, x):
        return self.net(x)

# MNIST Data (Using a small subset because loops are slow!)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Extract small subset (e.g., 200 train, 50 test)
X_train = train_set.data[:200].float().unsqueeze(1) / 255.0
y_train = train_set.targets[:200]
X_test = test_set.data[:50].float().unsqueeze(1) / 255.0
y_test = test_set.targets[:50]

train_loader = DataLoader(X_train, y_train, batch_size=20)
test_loader = DataLoader(X_test, y_test, batch_size=20)

# Setup Training
# Using cpu because of the loop-based implementation to avoid CUDA overhead
device = torch.device("cpu")
model = MyCNN().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Initialize Engine
engine = Engine(model, loss_fn, optimizer, device=device)

# Train
engine.fit(train_loader, test_loader, epochs=5)

# Visualize
engine.plot()
engine.plot_predictions(test_loader, num_images=10)