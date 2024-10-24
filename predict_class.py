import torchvision
import torch
import torchvision.transforms as transforms

mean, std = [0.6927], [0.1174]
size = 128

TRANSFORM = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize([size, size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(in_features=512, out_features=6, bias=True)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.load_state_dict(torch.load("C:/Users/96224/PycharmProjects/gagarin_hack/train_models/model_class", weights_only=True, map_location=torch.device('cpu')))


def predict_label():
    img = torchvision.datasets.ImageFolder("need_predict", TRANSFORM)
    img = img[0][0]
    img = img.reshape(1, 1, 128, 128)
    with torch.no_grad():
        label = model(img).argmax(dim=1).item()
    return label

