# Description: 저장된 LSTM 모델을 로드하고 새로운 데이터에 대한 예측을 수행하는 스크립트

import torch
import joblib
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt

# LSTM 모델 정의
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

# 시퀀스 생성 함수
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
    return np.array(sequences)

# 모델 및 스케일러 파일 경로 확인
model_path = 'hyperparameter_results/lstm_model_e10_s10_h40_l1_lr0.01_adamax_b1024_SScaler_normal.pth'
scaler_X_path = 'hyperparameter_results/scaler_X_e10_s10_h40_l1_lr0.01_adamax_b1024_SScaler_normal.pkl'
scaler_y_path = 'hyperparameter_results/scaler_y_e10_s10_h40_l1_lr0.01_adamax_b1024_SScaler_normal.pkl'

# 필요한 변수 정의
input_size = 5  # features 크기와 동일하게 설정
hidden_size = 40
num_layers = 1
output_size = 1
sequence_length = 10  # 시퀀스 길이 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 저장된 모델과 스케일러 로드
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()  # 모델을 평가 모드로 전환

scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

# 새로운 데이터 로드 및 전처리
new_data = pd.read_csv('processed_eval_data.csv')  # 새로운 데이터를 csv 파일에서 불러온다고 가정

# 필요한 피처를 선택
features = ['spread', 'mid_price', 'obi', 'price_volatility', 'buy_sell_ratio']
X_new = new_data[features].values

# 데이터 스케일링 적용
X_new_scaled = scaler_X.transform(X_new)

# 시퀀스 데이터 생성
X_new_seq = create_sequences(X_new_scaled, sequence_length)

# Tensor로 변환
X_new_tensor = torch.tensor(X_new_seq, dtype=torch.float32)

# 배치 처리
new_loader = torch.utils.data.DataLoader(X_new_tensor, batch_size=1024, shuffle=False)

# 예측 수행
predictions = []
with torch.no_grad():
    for batch_X in new_loader:
        batch_X = batch_X.to(device)  # 시퀀스 차원은 이미 추가되어 있음
        output = model(batch_X)
        predictions.append(output.cpu().numpy())

# 예측 결과를 한 배열로 결합
predictions = np.concatenate(predictions)

# 스케일러를 사용하여 원래 값으로 역변환
predictions_rescaled = scaler_y.inverse_transform(predictions.reshape(-1, 1))

# 실제 값과 비교 (시퀀스 길이를 고려하여 실제값도 동일하게 처리)
y_actual = new_data['trade_price'].values[sequence_length:]  # 시퀀스 길이만큼 앞의 값 제거

# 성능 평가
rmse = root_mean_squared_error(y_actual, predictions_rescaled)
mae = mean_absolute_error(y_actual, predictions_rescaled)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")

# 예측 vs 실제값 시각화
plt.figure(figsize=(10, 5))
plt.plot(y_actual, label='Actual Price')
plt.plot(predictions_rescaled, label='Predicted Price')
plt.legend()
plt.title('Actual vs Predicted Prices')
plt.show()
