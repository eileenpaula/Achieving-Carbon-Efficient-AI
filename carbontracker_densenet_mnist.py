import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from carbontracker.tracker import CarbonTracker
from carbontracker import parser
import time
from openpyxl import load_workbook

transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

class DenseNetMNIST(nn.Module):
    def __init__(self):
        super(DenseNetMNIST, self).__init__()
        self.densenet = models.densenet121(pretrained=False)
        self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.densenet.classifier = nn.Linear(512, 10) 

    def forward(self, x):
        return self.densenet(x)

model = DenseNetMNIST().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
tracker = CarbonTracker(
    epochs=10,
    epochs_before_pred=-1,
    monitor_epochs=-1,
    interpretable=True,
    log_dir="./densenet_mnist",
    verbose=2
)
start_time = time.time()

for epoch in range(10):
    tracker.epoch_start()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:   
            print(f'[Epoch: {epoch + 1}, Batch: {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    tracker.epoch_end()

tracker.stop()
training_time = time.time() - start_time

print('Finished Training')
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
# Parse logs
log_dir = "./densenet_mnist"
logs = parser.parse_all_logs(log_dir=log_dir)

# Initialize variables to handle missing data
energy_kwh = 0.0
co2_eq_g = 0.0

first_log = logs[0]

densenet_data = {
    "Model": "DenseNet",
    "Dataset": "MNIST",
    "Evaluation Framework": "carbontracker",
    "Total Energy (kWh)": first_log.get("actual", {}).get("energy (kWh)", 0.0),
    "Total CO2 Emissions (kgCO2e)": (first_log.get("actual", {}).get("co2eq (g)", 0.0)) / 1000,
    "Training Time (minutes)": training_time / 60,
    "Final Accuracy (%)": accuracy,
    "Number of Epochs": 10
}

wb = load_workbook("results.xlsx")
ws = wb.active
ws.append(["", ""])
for key, value in densenet_data.items():
    ws.append([key, value])

wb.save("results.xlsx")
print("DenseNet data added to results.xlsx")
