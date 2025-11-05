import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import datasets, transforms, models

# ===============================
# 1. Data transforms (same as training)
# ===============================
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# 2. Load validation dataset
# ===============================
data_dir = os.path.join("data", "sample_computer_vision", "val")
val_dataset = datasets.ImageFolder(data_dir, data_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)

class_names = val_dataset.classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===============================
# 3. Load trained model
# ===============================
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("best_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

# ===============================
# 4. Helper function to show images
# ===============================
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.axis('off')

# ===============================
# 5. Visualize a few predictions
# ===============================
inputs, classes = next(iter(val_loader))
inputs = inputs.to(device)
classes = classes.to(device)

with torch.no_grad():
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

# Show images with predicted and true labels
plt.figure(figsize=(10, 10))
for i in range(inputs.size(0)):
    plt.subplot(2, 2, i+1)
    imshow(inputs.cpu().data[i], title=f"pred: {class_names[preds[i]]}\ntrue: {class_names[classes[i]]}")

plt.show()
