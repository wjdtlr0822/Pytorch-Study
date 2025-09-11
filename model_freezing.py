import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
from tqdm import trange #진행률바

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4), #이미지 크롭
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#R,G,B 각각 0.5씩 빼고 0.5로 나누기 / 각각의 평균과 표준편차 / 정규화

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)

trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
testloader = DataLoader(testset, batch_size=16, shuffle=False)

model = torchvision.models.resnet18(weights="DEFAULT") #사전학습된 모델 불러오기
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False) #커널 크기 3x3, 스트라이드 1, 패딩 1
model.fc = nn.Linear(512, 10) #출력 클래스 10개로 변경
model = model.to(device) #모델을 GPU로 이동

print(model)


#모델 프리징
i = 0
for name, param in model.named_parameters():
    print(i, name)
    i+=1

frozen = range(3,60, 3)
for i, (name, param) in enumerate(model.named_parameters()):
    if i in frozen:
        param.requires_grad = False

#requires_grad확인
print(model.layer4[1].conv2.weight.requires_grad) #False
print(model.fc.weight.requires_grad) #True


criterion = nn.CrossEntropyLoss() #다중 클래스 분류에 적합한 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001) #Adam 옵티마이저

num_epochs = 20
ls = 2
pbar = trange(num_epochs)

for epoch in pbar:
    correct = 0
    total = 0
    running_loss = 0.0
    for data in trainloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    cost = running_loss / len(trainloader)
    acc = 100 * correct / total

    if cost < ls:
        ls = cost
        torch.save(model.state_dict(), './model_freezing.pth')

    pbar.set_postfix({'loss' : cost, 'train acc' : acc})        


model.load_state_dict(torch.load('./model_freezing.pth')) #가중치 불러오기

correct = 0
total = 0   
with torch.no_grad():
    model.eval() #평가모드로 변경 (드롭아웃, 배치정규화 등 비활성화)
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))        