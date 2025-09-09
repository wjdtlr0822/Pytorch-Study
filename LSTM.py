import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

df = pd.read_csv('.\kospi.csv')
print(df.head()) #처음 5줄 확인

scaler = MinMaxScaler() #모든 숫자를 0~1사이로 변환

df[['Open','High','Low','Close','Volume']] = scaler.fit_transform(df[['Open','High','Low','Close','Volume']]) #fit_transform : 데이터를 변환하는 함수
print(df.head()) #처음 5줄 확인
#여기서 Adj Close는 예측에 사용하지 않을거라서 제외를 했다.

df.info() #컬럼 정보, null 확인

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
x = df[['Open','High','Low','Volume']].values #독립변수
y = df['Close'].values #종속변수

#시계열 데이터로 변환 (예: 5일치 데이터로 다음날 종가 예측) RNN은 시계열 데이터에 특화된 모델
def seq_data(x,y,sequence_length):
    x_seq = []
    y_seq = []
    
    for i in range(len(x)-sequence_length):
        _x = x[i:i+sequence_length]
        _y = y[i+sequence_length]
        x_seq.append(_x)
        y_seq.append(_y)
        
    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view([-1,1])

split = 200
sequence_length = 5

x_seq, y_seq = seq_data(x,y,sequence_length)
x_train = x_seq[:split] #train 200개
y_train = y_seq[:split]
x_test = x_seq[split:] #test 226개
y_test = y_seq[split:] 
print("-------------------------------------------------")
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape) 
print("-------------------------------------------------")

train = torch.utils.data.TensorDataset(x_train, y_train) #TensorDataset : 텐서로 데이터셋을 만듦
test = torch.utils.data.TensorDataset(x_test, y_test)

batch_size = 20
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False) #DataLoader : 데이터셋을 배치 단위로 나누고 섞어줌
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

#hyperparameter
input_size = x_seq.size(2) #4
num_layers = 2 #너무 깊으면 안좋음 overffitting
hidden_size = 8 #너무 크면 안좋음 overfitting

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm= nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) 
        #batch_first=True : 입력 데이터의 첫번째 차원이 배치 크기임을 나타냄
        self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length,1), nn.Sigmoid()) #nn.Sigmoid() : 0~1사이로 변환
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device) #초기 은닉 상태
        c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device) #초기 셀 상태
        out, _ = self.lstm(x, (h0,c0)) #RNN 통과
        out = out.reshape(out.shape[0],-1) # many to many 전략 / 각각의 일마다 out이 나오면 output을 합쳐서 진행
        out = self.fc(out)
        return out
    
model = RNN(input_size, hidden_size, num_layers).to(device)
print(model)
criterion = nn.MSELoss() #회귀 문제에 적합한 손실 함수  

lr = 0.001
num_epochs = 200
optimizer = optim.Adam(model.parameters(), lr=lr) #Adam 옵티마이저

loss_ = [] #손실값 저장용
n = len(train_loader) #배치 개수\

for epoch in range(num_epochs):  # 데이터셋을 수차례 반복합니다.
    running_loss = 0.0

    for data in train_loader:
        
        seq, target = data
        out = model(seq) #순전파
        loss = criterion(out, target) #손실 계산

        optimizer.zero_grad() #변화도(Gradient) 매개변수를 0으로 만들고
        loss.backward()
        optimizer.step() #역전파 + 최적화를 한 후
        running_loss += loss.item() #통계 출력

    loss_.append(running_loss/n)
    if epoch % 100 == 0:
        print('[epoch: %d] loss: %.4f' %(epoch, running_loss/n))

plt.figure(figsize=(20,1))
plt.plot(loss_)
plt.show()

def plotting(train, test, actual):
    with torch.no_grad():
        train_pred = []
        test_pred = []
        
        for data in train_loader:
            seq, target = data
            out = model(seq)
            train_pred += out.cpu().numpy().tolist()

        for data in test_loader:
            seq, target = data
            out = model(seq)
            test_pred += out.cpu().numpy().tolist()

    total = train_pred + test_pred
    plt.figure(figsize=(12,6))
    plt.plot(np.ones(100)*len(train_pred), np.linspace(0,1,100), '--', linewidth=0.5) #train, test 나누는 선
    plt.plot(actual,'--')
    plt.plot(total,'b',linewidth=0.5)

    plt.legend(['train boundary', 'actual', 'prediction'])
    plt.show()

plotting(train_loader, test_loader, df['Close'].values[sequence_length:])
