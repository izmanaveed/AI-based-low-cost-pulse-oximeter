#Prediction on Arduino input

import serial
import numpy as np
#import mdr_oxi_new as oxi
import time
import pickle
from numpy.random import randint
import numpy as np
import pandas as pd
from pandas.core.indexes.base import Index
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from keras.models import load_model
from sklearn import preprocessing
#Load model and Variables
model=tf.keras.models.load_model('save_model')
pickle_in=open("data.pickle","rb")
history=pickle.load(pickle_in)

#Defining Parameters for plots
plt.ion()
fig1=plt.figure(1)
i=0
x=list()
y=list()
z=list()
ax1=fig1.add_subplot(121)
plt.title("BPM")
ax2=fig1.add_subplot(122)
ser=serial.Serial('COM3',115200)
ser.close()
ser.open()

#Data normalization
def normalized(data):
    df_scaled = data.copy()
    for column in df_scaled.columns:
        df_scaled[column] = df_scaled[column] / df_scaled[column].abs().max()
        return df_scaled
while True:
    data=ser.readline()
#print(data)
    decoded_values = str(data.decode("utf-8"))
#print(decoded_values)
    list_values = decoded_values.split('x')
    a=float(list_values[0])
    b=float(list_values[1])
    SPOdata=np.array([a,b])
    normalizedSPO=preprocessing.normalize(SPOdata)
#normalizedSPO=normalized(SPOdata)
    y_pred=model.predict(normalizedSPO)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    print("The prediction is :",y_pred )
    test_loss,test_acc=model.evaluate(normalizedSPO)
    print('Accuracy: %.2f' % (test_acc*100))
    x.append(i)
    y.append(a)
    z.append(b)
    ax1.plot(x,y,color='b')
    plt.title("SpO2")
    ax2.plot(x,z,color='g')
    fig1.canvas.draw()
    time.sleep(0.1)
    i=i+1
    plt.pause(0.001)