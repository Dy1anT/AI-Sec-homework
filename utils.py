import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import torchvision

def test_accuracy(test_loader, model, num):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            img, labels = data
            img, labels = img.to(device), labels.to(device)
            out = model(img)
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            if (total >= num): 
                break
    
    return correct / total
