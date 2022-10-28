import os
import datetime as dt
import pandas_datareader.data as web

# 銘柄コード入力(7177はGMO-APです。)
ticker_symbol = "7177"
ticker_symbol_dr = ticker_symbol + ".JP"

# 2022-01-01以降の株価取得
start = '2022-01-01'
end = dt.date.today()

# データ取得
df = web.DataReader(ticker_symbol_dr, data_source='stooq',
                    start=start, end=end)
# 2列目に銘柄コード追加
df.insert(0, "code", ticker_symbol, allow_duplicates=False)
# csv保存
df.to_csv(os.path.dirname(__file__) +
          '/scraping_data/s_stock_data_' + ticker_symbol + '.csv')
