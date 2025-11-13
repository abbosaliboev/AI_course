import os
import time
import copy
import random
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

# ---------------------------
# Config
# ---------------------------
DATA_DIR = os.path.join("data", "Sample_Computer_Vision")  # your dataset path
BATCH_SIZE = 8
NUM_EPOCHS = 10
VAL_SPLIT = 0.2   # if no val folder, split train into train/val by this fraction
NUM_WORKERS = 0   # use 0 on Windows to avoid multiprocessing issues
IMAGE_SIZE = 224  # input size for pretrained ResNet
SEED = 42

# ---------------------------
# 1) Data transforms
# ---------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# ---------------------------
# 2) Prepare datasets & dataloaders
# ---------------------------
def prepare_dataloaders(data_dir):
    """
    Supports the following dataset layouts:
    1) data_dir/train/<class_x>/*.jpg and data_dir/val/<class_x>/*.jpg
    2) data_dir/train/<class_x>/*.jpg and data_dir/test/<class_x>/*.jpg
       -> will split train into train+val automatically
    3) data_dir/<class_x>/*.jpg (no train/val folders) -> will split into train/val
    """

    # Case A: train and val folders exist
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        image_datasets = {
            'train': datasets.ImageFolder(train_dir, data_transforms['train']),
            'val': datasets.ImageFolder(val_dir, data_transforms['val'])
        }
    else:
        # If only train + test present, or only class folders are present
        # Try: if train exists and test exists -> use train and split train->train+val
        # Else: if root contains class folders -> use root and split
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            full_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
        else:
            # Check if root contains class subfolders (class folders directly under DATA_DIR)
            # e.g., data_dir/cat/*, data_dir/dog/*
            subfolders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            # If these subfolders look like class folders, use ImageFolder on root
            if len(subfolders) > 0 and all(os.listdir(os.path.join(data_dir, d)) for d in subfolders):
                full_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
            else:
                raise RuntimeError(f"Could not find a recognizable dataset layout in '{data_dir}'. "
                                   "Expected train/val or train/test or class-folders at root.")
        # split full_dataset into train and val
        total = len(full_dataset)
        val_len = int(total * VAL_SPLIT)
        train_len = total - val_len
        torch.manual_seed(SEED)
        train_set, val_set = random_split(full_dataset, [train_len, val_len])
        # For val we want to use val transforms (resize/center crop) — wrap with a simple dataset wrapper
        # random_split returns Subset objects; we can attach transform by replacing dataset.transform for val
        train_set.dataset.transform = data_transforms['train']
        val_set.dataset.transform = data_transforms['val']
        image_datasets = {'train': train_set, 'val': val_set}

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    }

    # Determine class names reliably
    if isinstance(image_datasets['train'], datasets.ImageFolder):
        class_names = image_datasets['train'].classes
    else:
        # Subset -> get underlying dataset's classes
        class_names = image_datasets['train'].dataset.classes

    dataset_sizes = {
        'train': len(image_datasets['train']),
        'val': len(image_datasets['val'])
    }

    return dataloaders, dataset_sizes, class_names

# ---------------------------
# 3) Model setup (automatic num_classes)
# ---------------------------
def create_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ---------------------------
# 4) Training loop
# ---------------------------
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model = model.to(device)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print('-' * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # load best weights
    model.load_state_dict(best_model_wts)
    return model

# ---------------------------
# 5) Main
# ---------------------------
if __name__ == "__main__":
    print("DATA_DIR =", DATA_DIR)
    if not os.path.exists(DATA_DIR):
        raise RuntimeError(f"Dataset folder not found: {DATA_DIR}")

    dataloaders, dataset_sizes, class_names = prepare_dataloaders(DATA_DIR)
    print("✅ Dataset prepared:")
    print("  - Train samples:", dataset_sizes['train'])
    print("  - Val samples:  ", dataset_sizes['val'])
    print("  - Classes:", class_names)

    # Create model with dynamic number of classes
    num_classes = len(class_names)
    model_ft = create_model(num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    print("✅ Model ready: ResNet18 (transfer learning initialized)")
    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, device, num_epochs=NUM_EPOCHS)

    out_path = "best_resnet18.pth"
    torch.save(model_ft.state_dict(), out_path)
    print(f"✅ Training finished and model saved as '{out_path}'")


model_conv = models.efficientnet_b3(weights='IMAGENET1K_V1')
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.classsifier[1].in_features
model_conv.classifier[1] = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)