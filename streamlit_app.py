import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model # type: ignore
import streamlit as st 

modell = load_model('stpp.keras')


st.header('Stock Price Prediction')
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2005-01-01'
end = '2015-12-31'


web_data = yf.download(stock , start , end)

st.subheader('data from 2005 - 2015')
st.write(web_data.describe())


#Visualizations
st.subheader('chart of Closing Price and Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(web_data.Close)
st.pyplot(fig)

st.subheader('chart of Closing Price and Time Chart with moving average of 100')
ma_100 = web_data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma_100)
plt.plot(web_data.Close)
plt.show()
st.pyplot(fig)

st.subheader('chart of Closing Price and Time Chart with moving average of 100 and 200')
ma_100 = web_data.Close.rolling(100).mean()
ma_200 = web_data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma_100)
plt.plot(ma_200)
plt.plot(web_data.Close)
plt.show()
st.pyplot(fig)

data_for_training=pd.DataFrame(web_data['Close'][0:int(len(web_data)*0.70)])
data_for_testing =pd.DataFrame(web_data['Close'][int(len(web_data)*0.70):int(len(web_data))])

print(data_for_training.shape)
print(data_for_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler(feature_range=(0,1))

scaled_web_data = scaler1.fit_transform(data_for_training)

p_100_days = data_for_training.tail(100)
final_dataframe = pd.concat((p_100_days , data_for_testing) , ignore_index=True)
final_dataframe2 = scaler1.fit_transform(final_dataframe)

X_test=[]
Y_test=[]

for i in range (100 , final_dataframe2.shape[0]):
    X_test.append(final_dataframe2[i-100:i])
    Y_test.append(final_dataframe2[i, 0])
    #x_train.append(d_train.iloc[i-100:i].values) # use .iloc and .values
    #y_train.append(d_train.iloc[i].values) # use .iloc and .values
X_test , Y_test = np.array(X_test) , np.array(Y_test)
Y_Prediction = modell.predict(X_test)
scaler1 = scaler1.scale_

scal_factor = 1/scaler1[0]
Y_Prediction = Y_Prediction * scal_factor
Y_test = Y_test * scal_factor


#Final stock Graph
st.subheader('Predictions versus Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(Y_test , 'b', label = 'Original')
plt.plot(Y_Prediction , 'r', label = 'Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)