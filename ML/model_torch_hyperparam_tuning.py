'''
Description: PyTorch를 사용하여 LSTM 모델의 하이퍼파라미터 튜닝을 반복적으로
수행하고, 학습에 사용된 Scaler와, 학습이 완료된 모델을 저장하는 스크립트
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import os

# Matplotlib의 백엔드를 'Agg'로 설정하여 Tkinter 관련 문제 방지
import matplotlib
matplotlib.use('Agg')

torch.backends.cudnn.benchmark = True

from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

# 시퀀스 생성 함수
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
    return np.array(sequences)

# LSTM 모델 정의 (dropout 없이)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)  # 초기 hidden state
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)  # 초기 cell state
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # 마지막 타임스텝의 출력
        return out

# 데이터 로드 및 StandardScaler 적용
data = pd.read_csv('processed_data.csv')
features = ['spread', 'mid_price', 'obi', 'price_volatility', 'buy_sell_ratio']
target = 'trade_price'

X = data[features].values
y = data[target].values

# StandardScaler 생성 및 데이터 정규화
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# 시계열 데이터 분할: 70% 학습, 10% 검증, 20% 테스트
train_size = int(len(X_scaled) * 0.7)
val_size = int(len(X_scaled) * 0.1)
test_size = len(X_scaled) - train_size - val_size

X_train, y_train = X_scaled[:train_size], y_scaled[:train_size]
X_val, y_val = X_scaled[train_size:train_size + val_size], y_scaled[train_size:train_size + val_size]
X_test, y_test = X_scaled[train_size + val_size:], y_scaled[train_size + val_size:]

# 하이퍼파라미터 튜닝 범위
epoch_range = range(10, 101, 10)
hidden_size_range = range(30, 101, 10)
num_layers_range = [1]
sequence_length_range = range(10, 61, 10)
learning_rates = [0.001, 0.003, 0.005, 0.01, 0.03]

# 디렉토리 생성
if not os.path.exists('hyperparameter_results'):
    os.makedirs('hyperparameter_results')

# CSV 파일에 결과 기록을 위한 초기 설정
csv_file_path = 'hyperparameter_results/test_losses.csv'
if not os.path.exists(csv_file_path):
    with open(csv_file_path, 'w') as f:
        f.write('file_suffix,test_loss\n')

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 하이퍼파라미터 튜닝
for num_epochs in epoch_range:
    for hidden_size in hidden_size_range:
        for num_layers in num_layers_range:
            for sequence_length in sequence_length_range:
                for lr in learning_rates:

                    file_suffix = f"e{num_epochs}_s{sequence_length}_h{hidden_size}_l{num_layers}_lr{lr}_adamax_b1024_SScaler_normal"
                    model_path = f'hyperparameter_results/lstm_model_{file_suffix}.pth'
                    result_exists = os.path.exists(model_path)

                    # 이미 테스트한 조합은 스킵
                    if result_exists:
                        print(f"Skipping already tested configuration: {file_suffix}")
                        continue

                    # 시퀀스 생성
                    X_train_seq = create_sequences(X_train, sequence_length)
                    X_val_seq = create_sequences(X_val, sequence_length)
                    X_test_seq = create_sequences(X_test, sequence_length)

                    y_train_seq = y_train[sequence_length:]
                    y_val_seq = y_val[sequence_length:]
                    y_test_seq = y_test[sequence_length:]

                    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
                    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
                    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
                    y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)
                    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
                    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

                    # 데이터 로더 생성
                    batch_size = 1024
                    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

                    # 모델 생성 (dropout 없이)
                    model = LSTMModel(len(features), hidden_size, num_layers, 1).to(device)

                    optimizer = optim.Adamax(model.parameters(), lr=lr)
                    criterion = nn.MSELoss()

                    scaler = GradScaler('cuda')

                    # 학습
                    train_losses, val_losses = [], []
                    for epoch in range(num_epochs):
                        model.train()
                        train_loss = 0
                        for batch_X, batch_y in train_loader:
                            batch_X = batch_X.to(device)
                            batch_y = batch_y.to(device).squeeze()

                            optimizer.zero_grad()
                            with autocast('cuda'):
                                outputs = model(batch_X)
                                loss = criterion(outputs.squeeze(), batch_y)

                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                            train_loss += loss.item()
                        train_losses.append(train_loss / len(train_loader))

                        # Validation
                        model.eval()
                        val_loss = 0
                        with torch.no_grad():
                            for batch_X, batch_y in val_loader:
                                batch_X = batch_X.to(device)
                                batch_y = batch_y.to(device).squeeze()

                                with autocast('cuda'):
                                    val_outputs = model(batch_X)
                                    loss = criterion(val_outputs.squeeze(), batch_y)
                                    val_loss += loss.item()
                        val_losses.append(val_loss / len(val_loader))

                        print(f"Epoch [{epoch+1}/{num_epochs}]")  # 잘 돌아가는지 출력

                    # 결과 저장
                    torch.save(model.state_dict(), model_path)
                    joblib.dump(scaler_X, f'hyperparameter_results/scaler_X_{file_suffix}.pkl')
                    joblib.dump(scaler_y, f'hyperparameter_results/scaler_y_{file_suffix}.pkl')

                    # 학습 및 검증 손실 그래프 저장
                    plt.figure(figsize=(10, 5))
                    plt.plot(train_losses, label='Train Loss')
                    plt.plot(val_losses, label='Validation Loss')
                    plt.legend()
                    plt.title(f"Train vs Validation Loss\nEpochs: {num_epochs}, Hidden: {hidden_size}, Layers: {num_layers}, LR: {lr}")
                    plt.savefig(f"hyperparameter_results/loss_curve_{file_suffix}.png")
                    plt.close()

                    # 테스트 평가
                    model.eval()
                    test_loss = 0
                    predictions = []
                    actuals = []
                    with torch.no_grad():
                        for batch_X, batch_y in test_loader:
                            batch_X = batch_X.to(device)
                            batch_y = batch_y.to(device).squeeze()

                            with autocast('cuda'):
                                test_outputs = model(batch_X)
                                loss = criterion(test_outputs.squeeze(), batch_y)
                                test_loss += loss.item()

                            predictions.append(test_outputs.cpu().squeeze().numpy())
                            actuals.append(batch_y.cpu().numpy())

                    # 테스트 손실 출력
                    test_loss_avg = test_loss / len(test_loader)
                    print(f'Test Loss for {file_suffix}: {test_loss_avg:.4f}')

                    # 테스트 손실을 CSV에 저장
                    with open(csv_file_path, 'a') as f:
                        f.write(f'{file_suffix},{test_loss_avg}\n')

                    # 테스트 데이터 예측 vs 실제 값 시각화 및 저장
                    predictions = np.concatenate(predictions)
                    actuals = np.concatenate(actuals)

                    plt.figure(figsize=(10, 5))
                    plt.plot(actuals, label='Actual Price')
                    plt.plot(predictions, label='Predicted Price')
                    plt.legend()
                    plt.title(f"Actual vs Predicted Price\nTest Loss: {test_loss_avg:.4f}\nEpochs: {num_epochs}, Hidden: {hidden_size}, Layers: {num_layers}, LR: {lr}")
                    plt.savefig(f"hyperparameter_results/test_results_{file_suffix}.png")
                    plt.close()

                    # GPU 메모리 해제
                    torch.cuda.empty_cache()

                    print(f"Model saved and test evaluation completed for {file_suffix}")