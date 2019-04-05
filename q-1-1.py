#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# # Importing the training set
df = pd.read_csv('./GoogleStocks.csv')
df = df.sort_values(by='date')
op = np.array(df['open'])
volumes = np.array(df['volume'])
cp = np.array(df['close'])
hp = np.array(df['high'])
lp = np.array(df['low'])

ap = np.mean([hp, lp], axis=0)

df['varience'] = ap

df1 = df

X_adm_train,X_adm_test = train_test_split(df,shuffle=False)

X_train = X_adm_train[['volume','varience']]
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)
# print(X_train)

sc_t = MinMaxScaler(feature_range = (0, 1))
o_vals = X_adm_train[['open']]
y_train = sc_t.fit_transform(o_vals)

# print(y_train)
real_stock_prices = X_adm_test['open']
print(X_train[0:10,0:2])


# In[4]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[5]:


#l , l1 test
l = 0
def RNN1(num_hidden_layers,num_cells,num_ts):
    l1 = l + 1
    # Initialising the RNN
    
    ##################training data###############################
    X_train1 = []
    y_train1 = []
    for i in range(num_ts, X_train.shape[0]):
        l1 = l1 + 1
        X_train1.append(X_train[i-num_ts:i,0:2])
        y_train1.append(y_train[i])
    X_train1, y_train1 = np.array(X_train1), np.array(y_train1)
    X_train1 = np.reshape(X_train1, (X_train1.shape[0], X_train1.shape[1]*2, 1))
#     print(X_train1.shape, y_train1.shape)
    ##############################################################
    
    ###################RNN########################################
    model = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = num_cells, return_sequences = True, input_shape = (X_train1.shape[1], 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    if(num_hidden_layers==3):
        model.add(LSTM(units = num_cells, return_sequences = True))
        model.add(Dropout(0.2))
    else:
        model.add(LSTM(units = num_cells))
        model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    if(num_hidden_layers==3):
        model.add(LSTM(units = num_cells))
        model.add(Dropout(0.2))


    # Adding the output layer
    model.add(Dense(units = 1))

    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    model.fit(X_train1, y_train1, epochs = 100, batch_size = 32)

    ###############testingdata############################
    af = df1[['volume','varience']]
    sc1 = MinMaxScaler(feature_range = (0, 1))
    XY = sc1.fit_transform(af)
    inputs = XY[len(df1) - len(X_adm_test) - num_ts:]
    print(inputs)
    X_test1 = []
    l2 = 0
    for i in range(num_ts, len(X_adm_test)+num_ts):
        l2 = l2 + 1
        X_test1.append(inputs[i-num_ts:i])
    X_test1 = np.array(X_test1)
    print(X_test1)
    # print(X_test.shape)

    X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1]*2, 1))
    predicted_stock_price = model.predict(X_test1)
    predicted_stock_price = sc_t.inverse_transform(predicted_stock_price)
    print(predicted_stock_price)
    plt.plot(real_stock_prices, color = 'black', label = 'Price of Google stock')
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Google Stock Price')
    plt.title('Prediction of Google Stock')
    plt.ylabel('Prediction of Google Stock Price')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


# In[6]:


predicted_stock_price = RNN1(3,80,75)


# In[ ]:


predicted_stock_price = RNN1(2,80,75)


# In[ ]:


predicted_stock_price = RNN1(2,50,75)

