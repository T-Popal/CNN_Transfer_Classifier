import torch
import os

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths
DATA_DIR = "data/"
#MODEL_SAVE_PATH = "models/best_model.pth"

# Training parameters
BATCH_SIZE = 4
NUM_EPOCHS = 2 #25
LEARNING_RATE = 0.001
MOMENTUM = 0.9
STEP_SIZE = 7
GAMMA = 0.1