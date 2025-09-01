#CNN을 활용한 이미지 분류 CIFAR-10

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#CIFAR-10은 클래스 10개를 가진 이미지 데이터

transforms = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#R,G,B 각각 0.5씩 빼고 0.5로 나누기 / 각각의 평균과 표준편차 / 정규화


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transforms)
trainloader = DataLoader(trainset, batch_size=100,
                                    shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transforms)
testloader = DataLoader(testset, batch_size=100,
                                    shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.feature_extractor = nn.Sequential(nn.Conv2d(3,6,5),
                                                nn.ReLU(),
                                                nn.MaxPool2d(2,2),
                                                nn.Conv2d(6,16,5),
                                                nn.ReLU(),
                                                nn.MaxPool2d(2,2))
        self.classifier = nn.Sequential(nn.Linear(16*5*5,120),
                                        nn.ReLU(),
                                        nn.Linear(120,84),
                                        nn.ReLU(),
                                        nn.Linear(84,10))
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 16*5*5)
        x = self.classifier(x)
        return x
        

net = Net().to(device) #모델을 GPU로 이동

print(net)#모델 구조 확인

criterion = nn.CrossEntropyLoss() #다중 클래스 분류에 적합한 손실 함수
optimizer = optim.Adam(net.parameters(), lr=0.001) #Adam 옵티마이저

loss_ = [] #손실값 저장용
n = len(trainloader) #배치 개수

for epoch in range(10):  # 데이터셋을 수차례 반복합니다.
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) #데이터를 GPU로 이동

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계 출력
        running_loss += loss.item()
        
    print(f'[Epoch {epoch + 1}, {i + 1:5d} / {n}] loss: {running_loss / n:.3f}')
    loss_.append(running_loss / n)
    #running_loss = 0.0
print('Finished Training')


plt.plot(loss_)
plt.xlabel('Epoch') 
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH) #모델 저장
#모델 불러오기
net = Net().to(device) #모델을 GPU로 이동 / 기존에 save를 gpu에서 했다면 load도 gpu에서 해야함
net.load_state_dict(torch.load(PATH)) #저장된 모델 매개변수를 불러옴

coorect = 0
total = 0
with torch.no_grad(): #평가시에는 기울기 계산 안함
    net.eval() #평가 모드로 전환(드롭아웃, 배치정규화 등 비활성화)
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device) #데이터를 GPU로 이동
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1) #가장 높은 값을 가진 클래스 선택 / _값은 필요없으므로 무시 / predicted는 예측된 클래스 / torch.max는 (값, 인덱스) 반환 
        total += labels.size(0) #전체 이미지 개수
        coorect += (predicted == labels).sum().item() #맞춘 이미지 개수

print(f'Accuracy of the network on the 10000 test images: {100 * coorect / total} %')