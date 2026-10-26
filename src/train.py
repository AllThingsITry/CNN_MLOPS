import argparse, os, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ── CNN Architecture (from your CNN.ipynb) ──────────────────────
class MyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

 
# ── Parse hyperparameters from SageMaker ────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--epochs',       type=int,   default=20)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--batch-size',    type=int,   default=32)
parser.add_argument('--num-classes',   type=int,   default=10)
# SageMaker automatically sets these environment variables
parser.add_argument('--model-dir',  default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
parser.add_argument('--train',      default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
parser.add_argument('--validation', default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
args = parser.parse_args()
import glob
print("=== DEBUG ===")
print("Train dir:", args.train)
print("Val dir:", args.validation)
print("Files in train dir:", os.listdir(args.train))
print("Files in val dir:", os.listdir(args.validation))

train_pt = os.path.join(args.train, 'train.pt')
print("Loading:", train_pt)
d = torch.load(train_pt)
print("Type:", type(d))
print("Keys:", d.keys() if isinstance(d, dict) else "NOT A DICT")
if isinstance(d, dict):
    print("Images:", d['images'].shape, d['images'].dtype)
    print("Labels:", d['labels'].shape, d['labels'].dtype)
print("=== END DEBUG ===")
 

# ── Training ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = MyCNN(num_classes=args.num_classes).to(device)

train_dict = torch.load(os.path.join(args.train, 'train.pt'))
val_dict   = torch.load(os.path.join(args.validation, 'val.pt'))

train_dataset = torch.utils.data.TensorDataset(train_dict['images'], train_dict['labels'])
val_dataset   = torch.utils.data.TensorDataset(val_dict['images'],   val_dict['labels'])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size)
 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
 
for epoch in range(args.epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{args.epochs} complete')
 
# ── Save model — SageMaker uploads /opt/ml/model/ to S3 ─────────

os.makedirs(args.model_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
print(f'Model saved to {args.model_dir}')
# updated
print(" NEW CODE IS RUNNING ")