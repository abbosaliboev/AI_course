import torch
import torch.nn as nn
from torchvision import models
from torch.optim import lr_scheduler

# Load pre-trained ResNet18 model
model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modify the final fully-connected layer (we have 2 classes: ants and bees)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

# Define loss function (cross-entropy) and optimizer (SGD)
criterion = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Learning rate scheduler (decays LR by 0.1 every 7 epochs)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print("✅ Model ready: ResNet18 (transfer learning initialized)")
