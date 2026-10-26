import json, os, torch
import tarfile
from torch.utils.data import DataLoader
import torch.nn as nn

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

    def forward(self, x):
        return self.classifier(self.features(x)) # reuse the class definition
 
MODEL_DIR  = '/opt/ml/processing/model'
TEST_DIR   = '/opt/ml/processing/test'
OUTPUT_DIR = '/opt/ml/processing/evaluation'

# Extract model.tar.gz first
tar_path = os.path.join(MODEL_DIR, 'model.tar.gz')
if os.path.exists(tar_path):
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(MODEL_DIR)
    print("Extracted model.tar.gz")
 
# Load model
model = MyCNN()
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'model.pth'), map_location='cpu'))
model.eval()
 
# Load test data
test_dict    = torch.load(os.path.join(TEST_DIR, 'test.pt'))
test_dataset = torch.utils.data.TensorDataset(test_dict['images'], test_dict['labels'])
test_loader  = DataLoader(test_dataset, batch_size=64)

# Evaluate
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        preds    = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
 
accuracy = correct / total
print(f'Test accuracy: {accuracy:.4f}')
 
# Write evaluation_report.json — pipeline reads 'metrics.accuracy.value'
os.makedirs(OUTPUT_DIR, exist_ok=True)
report = {
    "metrics": {
        "accuracy": {"value": accuracy},
        "correct":  {"value": correct},
        "total":    {"value": total}
    }
}
with open(os.path.join(OUTPUT_DIR, 'evaluation_report.json'), 'w') as f:
    json.dump(report, f, indent=2)
print('Evaluation report saved.')
