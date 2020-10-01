import matplotlib.pyplot as plt
import numpy as np
import MetaTrader5 as mt
from datetime import datetime
import math
from scipy.fftpack import fft
import pandas_datareader as data_reader
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import dump, load
from scipy.signal import medfilt,lfilter,firwin
import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_dim=9))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dense(units=1))
model.load_weights("stock_pred.h5")
model.compile(optimizer = 'RMSprop', loss = 'mean_squared_error', metrics=['accuracy'])

class Scaler():
    def __init__(self):
        self.models={}

    def loadModel(self, model):
        self.models[model]=load('scalers/{}.bin'.format(model))

    def trainModel(self, data, model):
        self.models[model] = StandardScaler()
        output = self.models[model].fit_transform(np.array(data))
        dump(self.models[model], 'scalers/{}.bin'.format(model), compress=True)
        return output

    def scale(self, data, model):
      if not model in self.models:
        try:
          self.loadModel(model)
        except:
          return self.trainModel(data, model)
      return self.models[model].transform(np.array(data))

class Tools:
    def __init__(self):
        self.EMA={}
        self.U={}
        self.D={}

    def ema(self,data,n):
        if not n in self.EMA:
            self.EMA[n]=data[0]
        self.EMA[n]=(self.EMA[n]*(n-1)+data[-1])/n
        return self.EMA[n]

    def rsi(self,data,n):
        if not (n in self.D and n in self.U):
            self.D[n]=1e-4
            self.U[n]=1e-4
        delta = data[-1]-data[-2]
        self.D[n]=(self.D[n]*(n-1)+[-delta,0][delta>0])/n
        self.U[n]=(self.U[n]*(n-1)+[delta,0][delta<0])/n
        return(100/(1+self.D[n]/self.U[n]))

    def diffs(self,data):
        return[data[i]-data[i-1] for i in range(1,len(data))]

    def voltality(self, data, n):
        delta=[]
        vol=[0]*(n+1)
        for d in range(len(data)-1):
            delta.append(abs(data[d]-data[d+1]))
        for d in range(len(delta)-n):
            vol.append(np.mean(delta[d:d+n]))
        return vol

    def pwr(self, data, n, c_off=30):
        vals=[0]*n
        for i in range(len(data)-n):
            sample=fft(data[i:i+n])[1:c_off]
            v=0
            for s in range(len(sample)):
                v+=abs(sample[s])*(c_off-s)
            vals.append(v)
        return vals


scaler=Scaler()

def evaluate(stock, src='yahoo', pts=20):
    m = data_reader.DataReader(stock, data_source=src)
    r_val=m["Close"].values
    weekdays=m.index.dayofweek.values
    closes=m["Close"].values
    volumes=m["Volume"].values

    RSI14=[50]
    RSI7=[50]
    EMA15=[closes[0]]
    EMA7=[closes[0]]
    EMA200=[closes[0]]
    tls=Tools()
    diffs=[0]+tls.diffs(closes)

    for i in range(1,len(closes)):
        data=closes[i-1:i+1]
        RSI14.append(tls.rsi(data,14))
        RSI7.append(tls.rsi(data,7))
        EMA15.append(tls.ema(data,15))
        EMA7.append(tls.ema(data,7))
        EMA200.append(tls.ema(data,200))

    closes=closes-EMA200
    EMA7=np.array(EMA7)-EMA200
    EMA15=np.array(EMA15)-EMA200
    pwr=tls.pwr(r_val,150)
    vol=tls.voltality(r_val,150)

    ds=np.array([weekdays,closes,volumes,RSI14,RSI7,EMA15,EMA7,diffs,pwr,vol]).T
    ds=scaler.scale(ds, stock)[-pts:]

    X=ds[:,list(filter(lambda x: x not in [1], list(range(len(ds[0])))))]
    y=ds[:,1]
    closes_pred=model.predict(X)

    dif=closes_pred[:,0]-y
    dif_f=medfilt(dif,5)

    return dif_f

def plot(stock, days):
    up=1
    dw=1
    dif=evaluate(stock, pts=days)
    for i in range(100):
        pts=np.where(dif>up)[0]
        if len(pts)>days/10 and np.max(pts)-np.min(pts)>days/4:
            break
        up-=0.01
    for i in range(100):
        pts=np.where(dif<-dw)[0]
        if len(pts)>days/10 and np.max(pts)-np.min(pts)>days/4:
            break
        dw-=0.01

    plt.axhline(0,color="gray")
    plt.plot(dif)
    plt.axhline(up, color='green')
    plt.axhline(-dw, color='red')
    plt.show()

    m = data_reader.DataReader(stock, data_source='yahoo')
    r_val=m["Close"].values[-days:]
    plt.plot(r_val)
    for i in range(len(dif)):
        if dif[i]>up:
            plt.plot(i,r_val[i],'go')
        elif dif[i]<-dw:
            plt.plot(i,r_val[i],'ro')
    plt.show()

