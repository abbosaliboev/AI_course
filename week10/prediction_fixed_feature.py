# prediction_fixed_feature.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import datasets 

# ===============================
# 1. Device
# ===============================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


model_conv = models.efficientnet_b3(weights=None)  
num_ftrs = model_conv.classifier[1].in_features
model_conv.classifier[1] = nn.Linear(num_ftrs, 2)

model_conv.load_state_dict(torch.load("best_efficientnet_b3.pth", map_location=device))
model_conv = model_conv.to(device)
model_conv.eval()


data_dir = os.path.join("data", "hymenoptera_data")
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=data_transforms['val'])
class_names = val_dataset.classes


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



def visualize_model_predictions(model, img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        plt.figure(figsize=(4,4))
        ax = plt.subplot(1,1,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        imshow(img.cpu().data[0])

    model.train(mode=was_training)


test_images = [
    'data/hymenoptera_data/val/bees/72100438_73de9f17af.jpg',
]

for img_path in test_images:
    visualize_model_predictions(model_conv, img_path)

plt.ioff()
plt.show()
