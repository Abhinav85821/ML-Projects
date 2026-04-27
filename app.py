import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("📈 SENSEX Prediction Dashboard")

# Download data
data = yf.download("^BSESN", start="2015-01-01")

st.subheader("SENSEX Data")
st.write(data.tail())

# Candlestick chart
st.subheader("📊 Candlestick Chart")

import plotly.graph_objects as go

# Reset index to use Date properly
data_reset = data.reset_index()

fig = go.Figure(
    data=[
        go.Candlestick(
            x=data_reset["Date"],
            open=data_reset["Open"],
            high=data_reset["High"],
            low=data_reset["Low"],
            close=data_reset["Close"]
        )
    ]
)

fig.update_layout(
    title="SENSEX Candlestick Chart",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)
# RSI calculation
st.subheader("RSI Indicator")

delta = data['Close'].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

rsi = 100 - (100/(1+rs))

st.line_chart(rsi)

# LSTM Prediction
st.subheader("Deep Learning Prediction")

close = data[['Close']]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(close)

X=[]
y=[]

for i in range(60,len(scaled)):
    X.append(scaled[i-60:i])
    y.append(scaled[i])

X=np.array(X)
y=np.array(y)

model = Sequential()

model.add(LSTM(50,return_sequences=True,input_shape=(X.shape[1],1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer="adam",loss="mean_squared_error")

model.fit(X,y,epochs=3,batch_size=32)

pred = model.predict(X)

predicted = scaler.inverse_transform(pred)

st.subheader("Actual vs Predicted")

chart_data = pd.DataFrame({
    "Actual":close.values.flatten()[60:],
    "Predicted":predicted.flatten()
})

st.line_chart(chart_data)

# Future prediction
st.subheader("Future Prediction")

days = st.slider("Predict days ahead",1,30)

future = model.predict(X[-1].reshape(1,60,1))

future_price = scaler.inverse_transform(future)

st.success(f"Predicted SENSEX after {days} days: {future_price[0][0]:.2f}")