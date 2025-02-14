import torch
import time
import os
from configuratation import DEVICE, NUM_EPOCHS
from dataset import load_datasets
from torchvision import models
from model import initialize_model
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

def train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    
    dataloaders, dataset_sizes, _ = load_datasets()
    best_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}\n" + "-" * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase} Loss: {running_loss / dataset_sizes[phase]:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f"Training complete. Best val Acc: {best_acc:.4f}")
    #torch.save(best_model_wts, MODEL_SAVE_PATH)
    #model.load_state_dict(best_model_wts)
    return model

# Initialize model, loss function, and optimizer
model = initialize_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


