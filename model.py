import torch
import torch.nn as nn
from torchvision import models
from configuratation import DEVICE

def initialize_model(num_classes=2, feature_extract=True):
  
    model = models.resnet18(weights='IMAGENET1K_V1')

    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False  # Freeze all layers

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model.to(DEVICE)