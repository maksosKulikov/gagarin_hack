import torch
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

size = 128
batch_size = 1
device = "cpu"

root_train = "C://Users//96224//PycharmProjects//gagarin_hack//datasets//data_after_seg//train"
root_valid = "C://Users//96224//PycharmProjects//gagarin_hack//datasets//data_after_seg//valid"

TRANSFORM = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize([size, size]),
                                transforms.ToTensor()])

data_train = torchvision.datasets.ImageFolder(root_train, TRANSFORM)
data_valid = torchvision.datasets.ImageFolder(root_valid, TRANSFORM)

mean, std = 0, 0
for data in data_train:
    mean += data[0].mean([1, 2])
    std += data[0].std([1, 2])

for data in data_valid:
    mean += data[0].mean([1, 2])
    std += data[0].std([1, 2])

mean /= (len(data_train) + len(data_valid))
std /= (len(data_train) + len(data_valid))

TRANSFORM = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize([size, size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

TRANSFORM_AU1 = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize([int(1.3 * size), int(1.3 * size)]),
                                    transforms.RandomRotation([90, 90]),
                                    transforms.RandomCrop(size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

TRANSFORM_AU2 = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize([int(1.3 * size), int(1.3 * size)]),
                                    transforms.RandomRotation([180, 180]),
                                    transforms.RandomCrop(size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

TRANSFORM_AU3 = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize([int(1.3 * size), int(1.3 * size)]),
                                    transforms.RandomRotation([270, 270]),
                                    transforms.RandomCrop(size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

data_train = torchvision.datasets.ImageFolder(root_train, TRANSFORM)
data_valid = torchvision.datasets.ImageFolder(root_valid, TRANSFORM)

for i in range(1):
    data_train += torchvision.datasets.ImageFolder(root_train, TRANSFORM_AU1)
    data_train += torchvision.datasets.ImageFolder(root_train, TRANSFORM_AU2)
    data_train += torchvision.datasets.ImageFolder(root_train, TRANSFORM_AU3)

train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size, shuffle=True)

model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(in_features=512, out_features=6, bias=True)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

accuracy_memory = [0]

model = model.to(device)
best_model = model

print("Кол-во параметров = ", sum(p.numel() for p in model.parameters() if p.requires_grad))

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

lambda1 = lambda epoch: 0.98 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

for epoch in tqdm(range(100)):
    accuracy_list = []

    for batch in train_loader:
        optimizer.zero_grad()
        images = batch[0]
        targets = batch[1]
        preds = model.forward(images.to(device))

        loss_batch = loss(preds, targets.to(device))

        loss_batch.backward()
        optimizer.step()

        accuracy_batch = (preds.argmax(dim=1) == targets.to(device)).float().mean()
        accuracy_list.append(accuracy_batch.item())

    accuracy = (sum(accuracy_list) / len(accuracy_list))
    # print(f"Точность тренировки {epoch + 1} эпохи = ", accuracy * 100, "%", sep="")
    accuracy, items_count = 0, 0

    classes = data_valid.classes
    errors_dict = {}
    for name in classes:
        errors_dict[name] = 0
    for batch in valid_loader:
        with torch.no_grad():
            images = batch[0]
            targets = batch[1]
            preds = model.forward(images.to(device))
            for i in range(len(preds.argmax(dim=1))):
                if preds.argmax(dim=1)[i] != targets[i].to(device):
                    errors_dict[classes[targets[i]]] += 1
                else:
                    accuracy += 1
                items_count += 1

    accuracy = (accuracy / items_count)
    if accuracy > max(accuracy_memory):
        best_model = model
    accuracy_memory.append(accuracy)
    scheduler.step()
    # print(f"Точность валидации {epoch + 1} эпохи = ", accuracy * 100, "%", sep="")

print(max(accuracy_memory))
torch.save(best_model.state_dict(), "model_class")
