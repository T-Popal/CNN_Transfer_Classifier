import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(inp, title=None):
   
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    
    if title:
        plt.title(title)
    plt.show()


