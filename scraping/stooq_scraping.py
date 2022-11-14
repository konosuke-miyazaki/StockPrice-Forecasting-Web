from calendar import weekday
import os
import datetime as dt
from datetime import timedelta
import pandas_datareader.data as web
import matplotlib.pyplot as plt

# 銘柄コード入力(7177はGMO-APです。)
ticker_symbol = "7177"
ticker_symbol_dr = ticker_symbol + ".JP"

# 2022-01-01以降の株価取得
start = dt.datetime(2022,1,1)
end = dt.date.today()

# データ取得
df = web.DataReader(ticker_symbol_dr, data_source='stooq',
                    start=start, end=end)
# データに曜日を追加
df['weekday'] = df.index.weekday

df.info()

# 2列目に銘柄コード追加
df.insert(0, "code", ticker_symbol, allow_duplicates=False)

# データの並び替え
df.sort_values(by='Date', ascending=True, inplace=True)

#カラム情報を1行上にずらしたデータフレームを作成する
df_shift = df.shift(-1)
df_shift

#翌日の始値と本日の終値の差分を追加する
df['delta_Close'] = df_shift['Close'] - df['Close']

#目的変数Upを追加する(翌日の終値が上がる場合1、それ以外は0とする)、'delta_Close'カラムの削除
df['Up'] = 0
df['Up'][df['delta_Close'] > 0] = 1
df = df.drop('delta_Close', axis=1)


# 'Open', 'High', 'Low', 'Close'グラフ化のためにカラム抽出
df_new = df[['Open', 'High', 'Low', 'Close']]

# 時系列折れ線グラフの作成
df_new.plot(kind='line')
# plt.show()

# 終値の前日比の追加
df_shift = df.shift(1)

df['Close_ratio'] = (df['Close'] - df_shift['Close']) / df_shift['Close']

# 始値と終値の差分を追加
df['Body'] = df['Open'] - df['Close']



# plt.figure(figsize=(10, 6))
# plt.plot(df['Close'], label='Close', color='orange')
# plt.xlabel('Date')
# plt.ylabel('USD')
# plt.legend()
# plt.show()


# # csv保存
# df.to_csv(os.path.dirname(__file__) +
#           '/scraping_data/s_stock_data_' + ticker_symbol + '.csv')
