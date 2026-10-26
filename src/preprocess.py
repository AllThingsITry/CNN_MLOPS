import os
from torchvision import datasets, transforms
from torch.utils.data import random_split, ConcatDataset
import torch

# Input directories (raw images)
INPUT_DIR = ['/opt/ml/processing/input/test', '/opt/ml/processing/input/train']

# Output directories for preprocessing step
TRAIN_DIR = '/opt/ml/processing/output/train'
VAL_DIR   = '/opt/ml/processing/output/validation'
TEST_DIR  = '/opt/ml/processing/output/test'

# Create output directories if they don't exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Transform: resize + to tensor (currently has normalization)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5), (0.5, 0.5))  # remove if you don't want normalization
])

# Load and combine datasets
dataset_list = [datasets.ImageFolder(d, transform=transform) for d in INPUT_DIR]
dataset = ConcatDataset(dataset_list)

# Random split: 70% train, 15% validation, 15% test
torch.manual_seed(42)  # reproducible splits
n = len(dataset)
n_train = int(0.7 * n)
n_val   = int(0.15 * n)
n_test  = n - n_train - n_val
train_data, val_data, test_data = random_split(dataset, [n_train, n_val, n_test])

# Function to save each split as a .pt file
def save_split(subset, path):
    images, labels = [], []
    for img, lbl in subset:
        images.append(img)
        labels.append(lbl)
    torch.save({
    'images': torch.stack(images), 
    'labels': torch.tensor(labels, dtype=torch.long)  # ← explicit dtype
}, path)

# Save the splits
save_split(train_data, os.path.join(TRAIN_DIR, 'train.pt'))
save_split(val_data, os.path.join(VAL_DIR, 'val.pt'))
save_split(test_data, os.path.join(TEST_DIR, 'test.pt'))

# Debug output
print("Preprocessing complete.")
print("Train files:", os.listdir(TRAIN_DIR))
print("Validation files:", os.listdir(VAL_DIR))
print("Test files:", os.listdir(TEST_DIR))