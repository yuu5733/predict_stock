import streamlit as st
import numpy as np
import pandas as pd
import datetime as datetime
import pandas_datareader
import plotly.graph_objects as go

import sklearn.linear_model
import sklearn.model_selection
from PIL import Image
import yfinance as yfin
# yf.pdr_override()

st.title("AIで株価予測アプリ")
st.write('AIを使って、株価を予測してみましょう。')

# トップ画像の表示
image = Image.open('stock_predict.png')
st.image(image, use_container_width=True) # st.image(image, use_column_width=True)

st.write('※あくまでAIによる予測です(参考)。こちらのアプリによる損害や損失は一切補償しかねます。')

st.header("検索銘柄のティッカーシンボルを入力してください。")
stock_name = st.text_input("例:AAPL,FB,SFTBY(大文字・小文字どちらでも可)","AAPL")

stock_name = stock_name.upper()

link = 'https://search.sbisec.co.jp/v2/popwin/info/stock/popa640_usecurity_list.html'
st.markdown(link)
st.write('ティッカーシンボルについては上記のリンク(SBI証券)をご参照ください。')


try:
    # df_stock = pandas_datareader.data.get_data_yahoo(stock_name, '2021-01-05')
    df_stock = yfin.download(stock_name, '2021-01-05')
    if isinstance(df_stock.columns, pd.MultiIndex):
        # 'Close', 'Open', 'High' など、値が入っているレベルの列名を取得
        df_stock.columns = df_stock.columns.get_level_values(0)

    st.header(stock_name + " 2022年1月5日から現在までの価格(USD)")
    st.write(df_stock)

    st.header(stock_name + " 終値と14日間平均(USD)")
    df_stock['SMA'] = df_stock['Close'].rolling(window=14).mean()
    df_stock2 = df_stock[['Close', 'SMA']]
    st.line_chart(df_stock2)

    st.header(stock_name + " 値動き(USD)")
    df_stock['change'] = ((df_stock['Close'] - df_stock['Open']) / (df_stock['Open']) * 100)
    st.line_chart(df_stock['change'].tail(100))

    #キャンドルスティックの準備
    fig = go.Figure(
        data=[go.Candlestick(
            x=df_stock.index,
            open=df_stock['Open'],
            high=df_stock['High'],
            low=df_stock['Low'],
            close=df_stock['Close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        )]
    )

    st.header(stock_name + " キャンドルスティック")
    st.plotly_chart(fig, use_container_width=True) # st.plotly_chart(fig, use_column_width=True)

    df_stock['label'] = df_stock['Close'].shift(-30)

    st.header(stock_name + ' 1か月後を予測しよう (USD)')
    def stock_predict():
        # 機械学習(マシンラーニング)
        X = np.array(df_stock.drop(['label', 'SMA'], axis=1))
        X = sklearn.preprocessing.scale(X)
        predict_data = X[-30:]
        X = X[:-30]
        Y = np.array(df_stock['label'])
        Y = Y[:-30]
        # データの分割
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, Y, test_size = 0.2
        )
        # 訓練データを用いて学習する
        model = sklearn.linear_model.LinearRegression()
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        # 小数点第一位で四捨五入
        st.write(f'正答率は{round(accuracy * 100, 1)}%です。')

        # accuracyより信頼度を表示
        if accuracy > 0.75:
            st.write('信頼度: 高')
        elif accuracy > 0.5:
            st.write('信頼度: 中')
        else:
            st.write('信頼度: 低')
        st.write('**オレンジの線(Predict)が予測値です。**')

        # 検証データーを用いて検証してみる
        predicted_data = model.predict(predict_data)
        df_stock['Predict'] = np.nan
        last_date = df_stock.iloc[-1].name
        one_day = 86400
        next_unix = last_date.timestamp() + one_day

        for data in predicted_data:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += one_day
            df_stock.loc[next_date] = np.append([np.nan] * (len(df_stock.columns) - 1), data)

        df_stock['Close'].plot(figsize=(15, 6), color="green")
        df_stock['Predict'].plot(figsize=(15, 6), color="orange")

        df_stock3 = df_stock[['Close', 'Predict']]
        st.line_chart(df_stock3)

    # 関数をボタンを押すとstock_predict()が発動
    if st.button('予測する'):
        stock_predict()

except:
    st.error("エラーがおきているようです。")

st.write('Copyright © 2021 Tomoyuki Yoshikawa. All Rights Reserved.')