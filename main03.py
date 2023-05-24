# 新增機率 / 改為line傳輸
import yfinance as yf
from finta import TA
from keras.models import load_model
import pickle
import datetime as dt
import requests
import schedule
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def send_line(msg, token='Csar0JwlKTUcd2ozNUhQQN7tq5JFJlnSBMTzIgaNZ3R'):
    url = "https://notify-api.line.me/api/notify"  # --> 不支援http, 只能用https
    headers = {"Authorization" : "Bearer " + token}
    title = 'AI(2W漲跌3%預測)'
    message = '[%s] %s' % (title, msg)
    payload = {"message" : message}
    r = requests.post(url, headers=headers, params=payload)

# 計算金融技術指標函數
def calculate_ta(df):
    ta_functions = [TA.RSI, TA.WILLIAMS, TA.SMA, TA.EMA, TA.WMA, TA.HMA, TA.TEMA, TA.CCI, TA.CMO, TA.MACD, TA.PPO,
                    TA.ROC, TA.CFI, TA.DMI, TA.SAR]
    ta_names = ['RSI', 'Williams %R', 'SMA', 'EMA', 'WMA', 'HMA', 'TEMA', 'CCI', 'CMO', 'MACD', 'PPO', 'ROC', 'CFI',
                'DMI', 'SAR']
    for i, ta_func in enumerate(ta_functions):
        try:
            df[ta_names[i]] = ta_func(df)
        except:
            if ta_names[i] == 'MACD':
                df[ta_names[i]] = ta_func(df)['MACD'] - ta_func(df)['SIGNAL']
            if ta_names[i] == 'PPO':
                df[ta_names[i]] = ta_func(df)['PPO'] - ta_func(df)['SIGNAL']
            if ta_names[i] == 'DMI':
                df[ta_names[i]] = ta_func(df)['DI+'] - ta_func(df)['DI-']
    return df


# 給定歷史數據、scaler、model，輸出買賣訊號的函數
def process(df, min_max_scaler, model):
    df = calculate_ta(df)
    df = df.dropna(axis=0)
    features = ['RSI', 'Williams %R', 'SMA', 'EMA', 'WMA', 'HMA', 'TEMA', 'CCI', 'CMO', 'MACD', 'PPO', 'ROC', 'CFI',
                'DMI', 'SAR']
    Close = df[['Close']]
    df = df[features]
    # 數據轉換
    df[features] = min_max_scaler.transform(df[features])
    # 製作X（15x15的array）
    days = 15
    start_index = 0
    end_index = len(df) - days
    Xs = []
    indexs = []
    for i in tqdm(range(start_index, end_index + 1, 1)):
        X = df.iloc[i:i + days, :][features]
        X = np.array(X)
        Xs.append(X)
        indexs.append((df.iloc[[i]].index, df.iloc[[i + days - 1]].index))
    Xs = np.array(Xs)
    # 模型預測買賣訊號
    predictions = model.predict(Xs)
    signals = [np.argmax(pred) for pred in predictions]
    probabilities = [pred.tolist() for pred in predictions]

    # 繪圖
    Close = Close.iloc[-len(Xs):, :]
    Close['SIGNAL'] = signals
    Close['PROBABILITY'] = probabilities
    buy = Close[Close['SIGNAL'] == 1]['Close']
    sell = Close[Close['SIGNAL'] == 2]['Close']

    Close['Close'].plot()
    plt.scatter(list(buy.index), list(buy.values), color='red', marker="^")
    Close['Close'].plot()
    plt.scatter(list(sell.index), list(sell.values), color='green', marker='v')
    plt.show()

    return signals[-1], [f'{p:.6f}' for p in probabilities[-1]]

def main():
    # 載入模型和scaler
    model = load_model('model_TWII.h5')
    with open('scaler_TWII.pkl', 'rb') as f:
        min_max_scaler = pickle.load(f)
    # 資料時間範圍設定
    start_date = (dt.datetime.now() - dt.timedelta(days=180)).strftime("%Y-%m-%d")
    end_date = dt.datetime.now().strftime("%Y-%m-%d")
    symbol = '^TWII'
    df = yf.download(symbol, start=start_date, end=end_date)
    # 取得交易訊號和機率
    signal, probability = process(df, min_max_scaler, model)
    if signal == 0:
        action = '不動作'
    elif signal == 1:
        action = '建議買入'
    elif signal == 2:
        action = '建議賣出'
    # 發送消息至Line
    current_time = dt.datetime.now().strftime("%Y-%m-%d")
    message1 = f"\n日期: {current_time}\n項目: {symbol}\n建議: {action}\n機率: \n{', '.join(probability)}"
    print(message1)
    send_line(message1)

    model = load_model('model_TWOII.h5')
    with open('scaler_TWOII.pkl', 'rb') as f:
        min_max_scaler = pickle.load(f)
    # 資料時間範圍設定
    start_date = (dt.datetime.now() - dt.timedelta(days=180)).strftime("%Y-%m-%d")
    end_date = dt.datetime.now().strftime("%Y-%m-%d")
    symbol = '^TWOII'
    df = yf.download(symbol, start=start_date, end=end_date)
    # 取得交易訊號和機率
    signal, probability = process(df, min_max_scaler, model)
    if signal == 0:
        action = '不動作'
    elif signal == 1:
        action = '建議買入'
    elif signal == 2:
        action = '建議賣出'
    # 發送消息至Line
    current_time = dt.datetime.now().strftime("%Y-%m-%d")
    message2 = f"\n日期: {current_time}\n項目: {symbol}\n建議: {action}\n機率: \n{', '.join(probability)}"
    print(message2)
    send_line(message2)

if __name__ == '__main__':
    main()
