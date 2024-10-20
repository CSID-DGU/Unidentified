import pandas as pd
import numpy as np
import json
import glob
import os

# 데이터 파일 경로 설정
orderbook_path = './eval_orderbooks/*.txt'  # 호가창 데이터가 저장된 디렉토리 경로
ticks_path = './eval_ticks/*.txt'  # 거래 데이터가 저장된 디렉토리 경로

# 모든 .txt 파일 불러오기
orderbook_files = glob.glob(orderbook_path)
ticks_files = glob.glob(ticks_path)

# 빈 리스트 생성
orderbook_data = []
ticks_data = []

# 배치 크기 설정
batch_size = 4  # 한 번에 처리할 파일 개수

# 필요한 feature만 추출 (예: 가격 스프레드, 중간 가격, 호가창 불균형 등)
def preprocess_orderbook(orderbook_df):
    orderbook_df['spread'] = orderbook_df['orderbook_units'].apply(lambda x: x[0]['ask_price'] - x[0]['bid_price'])
    orderbook_df['mid_price'] = orderbook_df['orderbook_units'].apply(lambda x: (x[0]['ask_price'] + x[0]['bid_price']) / 2)
    orderbook_df['obi'] = (orderbook_df['total_bid_size'] - orderbook_df['total_ask_size']) / \
                          (orderbook_df['total_bid_size'] + orderbook_df['total_ask_size'])
    return orderbook_df[['timestamp', 'spread', 'mid_price', 'obi']]  # 필요한 feature만 반환

# ticks 데이터에서 추가적인 feature 추출 (가격 변동성, 매수/매도 비율 등)
def preprocess_ticks(ticks_df):
    ticks_df['price_volatility'] = ticks_df['trade_price'].rolling(window=5).std()
    ticks_df['buy_sell_ratio'] = ticks_df['ask_bid'].apply(lambda x: 1 if x == 'BID' else 0).rolling(window=5).mean()
    return ticks_df[['timestamp', 'trade_price', 'price_volatility', 'buy_sell_ratio']]  # 필요한 feature만 반환

# 배치 단위로 orderbook 데이터 처리
for batch_start in range(0, len(orderbook_files), batch_size):
    batch_files = orderbook_files[batch_start:batch_start + batch_size]
    
    for file_idx, file in enumerate(batch_files):
        print(f"Processing orderbook file {file_idx + 1}/{len(batch_files)} in batch: {file}")
        with open(file, 'r') as f:
            for line in f:
                # 작은따옴표를 큰따옴표로 변환하여 JSON 파싱
                line = line.replace("'", '"')
                try:
                    data = json.loads(line)  # JSON 형식의 데이터를 불러옴
                    orderbook_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file}: {e}")
    
    # Orderbook 데이터프레임으로 변환 및 전처리
    orderbook_df = pd.DataFrame(orderbook_data)
    
    if not orderbook_df.empty:
        # 중복 제거 및 타임스탬프 기준 정렬
        orderbook_df.drop_duplicates(subset=['timestamp'], inplace=True)
        orderbook_df.sort_values(by='timestamp', inplace=True)
        
        # 필요한 feature만 추출
        orderbook_processed = preprocess_orderbook(orderbook_df)
        
        # 중간 데이터 처리 후 메모리 비우기
        orderbook_data.clear()
    
        # 중간 결과를 저장 (여기서는 결합할 데이터를 임시 저장)
        orderbook_processed.to_csv(f'orderbook_batch_{batch_start}.csv', index=False)

# 배치 단위로 ticks 데이터 처리
for batch_start in range(0, len(ticks_files), batch_size):
    batch_files = ticks_files[batch_start:batch_start + batch_size]
    
    for file_idx, file in enumerate(batch_files):
        print(f"Processing ticks file {file_idx + 1}/{len(batch_files)} in batch: {file}")
        with open(file, 'r') as f:
            for line in f:
                # 작은따옴표를 큰따옴표로 변환하여 JSON 파싱
                line = line.replace("'", '"')
                try:
                    data = json.loads(line)  # JSON 형식의 데이터를 불러옴
                    ticks_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file}: {e}")
    
    # Ticks 데이터프레임으로 변환 및 전처리
    ticks_df = pd.DataFrame(ticks_data)
    
    if not ticks_df.empty:
        # 중복 제거 및 타임스탬프 기준 정렬
        ticks_df.drop_duplicates(subset=['timestamp'], inplace=True)
        ticks_df.sort_values(by='timestamp', inplace=True)
        
        # 필요한 feature만 추출
        ticks_processed = preprocess_ticks(ticks_df)
        
        # 중간 데이터 처리 후 메모리 비우기
        ticks_data.clear()
    
        # 중간 결과를 저장 (여기서는 결합할 데이터를 임시 저장)
        ticks_processed.to_csv(f'ticks_batch_{batch_start}.csv', index=False)

# 최종적으로 각각 저장한 배치 파일을 병합하여 하나의 데이터로 생성
def merge_batches(batch_prefix, output_file):
    batch_files = glob.glob(f'{batch_prefix}_batch_*.csv')
    merged_df = pd.concat([pd.read_csv(file) for file in batch_files])
    
    # 결측치 제거
    merged_df.dropna(inplace=True)
    
    # 최종 CSV 파일로 저장
    merged_df.to_csv(output_file, index=False)

# Orderbook과 Ticks 데이터를 각각 병합
merge_batches('orderbook', 'final_orderbook_data.csv')
merge_batches('ticks', 'final_ticks_data.csv')

# Orderbook과 Ticks 데이터를 병합하기 전에 timestamp 기준으로 정렬
final_orderbook_df = pd.read_csv('final_orderbook_data.csv')
final_ticks_df = pd.read_csv('final_ticks_data.csv')

final_orderbook_df.sort_values(by='timestamp', inplace=True)
final_ticks_df.sort_values(by='timestamp', inplace=True)

# 두 데이터를 타임스탬프를 기준으로 병합
merged_data = pd.merge_asof(final_orderbook_df, final_ticks_df, on='timestamp', direction='nearest')

# 최종 데이터를 저장
merged_data.to_csv('processed_eval_data.csv', index=False)

# 배치 파일 및 중간 파일 삭제
def delete_files(file_pattern):
    files = glob.glob(file_pattern)
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted file: {file}")
        except OSError as e:
            print(f"Error deleting file {file}: {e}")

# 배치 처리된 파일과 최종 결합된 파일 삭제
delete_files('orderbook_batch_*.csv')
delete_files('ticks_batch_*.csv')
delete_files('final_orderbook_data.csv')
delete_files('final_ticks_data.csv')

print("Orderbook 및 Ticks 데이터가 성공적으로 처리되고 중간 파일들이 삭제되었습니다!")
