import serial
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
plt.ion()
fig1=plt.figure(1)
i=0
x=list()
y=list()
z=list()
ax1=fig1.add_subplot(121)
plt.title("BPM")
ax2=fig1.add_subplot(122)
fig1.show()
ser=serial.Serial('COM3',115200)
ser.close()
ser.open()
while True:
    data=ser.readline()
    #print(data)
    decoded_values = str(data.decode("utf-8"))
    #print(decoded_values)
    list_values = decoded_values.split('x')
    a=float(list_values[0])
    b=float(list_values[1])
    x.append(i)
    y.append(a)
    z.append(b)
    #ax1.title("BPM")
    ax1.plot(x,y,color='b')
    plt.title("SpO2")
    ax2.plot(x,z,color='g')
    fig1.canvas.draw()
    time.sleep(0.1)
    i=i+1
    plt.pause(0.001)