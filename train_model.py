import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn_architecture import CIFAR_CNN

# 1. Transform — prepare images for the network
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 2. Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                  download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                 download=True, transform=transform)

# 3. DataLoader — feeds data in batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# 4. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = CIFAR_CNN().to(device)

# 5. Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training loop
def train(num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients from last step
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}  |  Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), 'cifar_cnn.pth')
        print("model saved")

if __name__ == '__main__':
    train(num_epochs=10)
