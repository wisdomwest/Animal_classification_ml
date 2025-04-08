import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to data directories
data_dir = 'data/extracted'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Image transformations for training and testing
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load image datasets
datasets_ = {
    'train': datasets.ImageFolder(train_dir, transform['train']),
    'test': datasets.ImageFolder(test_dir, transform['test'])
}

# Create data loaders
dataloaders = {
    phase: DataLoader(datasets_[phase], batch_size=32, shuffle=(phase == 'train'), num_workers=4)
    for phase in ['train', 'test']
}

# Get class names and number of classes
class_names = datasets_['train'].classes
num_classes = len(class_names)
print(f"Classes: {class_names}")

# Load pre-trained ResNet152 and replace the final layer
model = models.resnet152(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Define loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}\n{"-"*10}')

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward and optimize if training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Update stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            # Step the scheduler after training phase
            if phase == 'train':
                scheduler.step()

            # Compute average loss and accuracy
            epoch_loss = running_loss / len(datasets_[phase])
            epoch_acc = running_corrects.double() / len(datasets_[phase])
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            # Save the best model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_model.pth')

    print(f'\nBest Test Accuracy: {best_acc:.4f}')
    return model

# Train and save best model
model = train_model(model, criterion, optimizer, scheduler)

