#Data generation and preprocessing code

####SPO2 data####
spo=randint(0,85,600000)
sp=randint(86,94,300000)
spo2=randint(95,100,101000)

#####BPM data####
abBPM=randint(0,59,600000)
ab1BPM=randint(101,200,300000)
nBPM=randint(60,100,101000)
####Data set generation####
dataSpo=np.concatenate([spo,spo2,sp])
dataBpm=np.concatenate([ab1BPM,nBPM,abBPM])
data=np.vstack((dataSpo,dataBpm))
print(data[0:5])
dataT=np.transpose(data)
####assign labels####
labels=[0,1,2,3,4,5,6,7,8]
empty=np.zeros((1,1001000))
rows=dataT.shape[0]
cols=dataT.shape[1]
empty=np.transpose(empty)
for iy in range(0,cols):
for ix in range(0,rows):
if dataT[ix,0] >= 95 and dataT[ix,0] <=100 and dataT[ix,iy] >= 60 and dataT[ix,iy] <= 100:
empty[ix,0]=labels[0]
if dataT[ix,0] >= 0 and dataT[ix,0] <= 85 and dataT[ix,iy] >= 60 and dataT[ix,iy] <= 100:
empty[ix]=labels[1]
if dataT[ix,0] >= 86 and dataT[ix,0] <= 94 and dataT[ix,iy] >= 60 and dataT[ix,iy] <= 100:
empty[ix]=labels[2]
if dataT[ix,0] >= 95 and dataT[ix,0] <= 100 and dataT[ix,iy] >= 0 and dataT[ix,iy] <= 59:
empty[ix]=labels[3]
if dataT[ix,0] >= 0 and dataT[ix,0] <= 85 and dataT[ix,iy] >= 0 and dataT[ix,iy] <= 59:
empty[ix]=labels[4]
if dataT[ix,0] >= 86 and dataT[ix,0] <= 94 and dataT[ix,iy] >= 0 and dataT[ix,iy] <= 59:
empty[ix]=labels[5]
if dataT[ix,0] >= 95 and dataT[ix,0] <= 100 and dataT[ix,iy] >= 101 and dataT[ix,iy] <= 200:
empty[ix]=labels[6]
if dataT[ix,0] >= 0 and dataT[ix,0] <= 85 and dataT[ix,iy] >= 101 and dataT[ix,iy] <= 200:
empty[ix]=labels[7]
if dataT[ix,0] >= 86 and dataT[ix,0] <= 94 and dataT[ix,iy] >= 101 and dataT[ix,iy] <= 200:
empty[ix]=labels[8]
####save data####
df=pd.DataFrame(dataT,columns=['SPO','BPM'])
df.to_csv('data.csv', encoding='utf-8',index=False)
label=pd.DataFrame(empty,columns=['labels'])
label.to_csv('labels.csv', encoding='utf-8',index=False)

#Neural Network model
####load data####
SPOdata=pd.read_csv('data.csv',header=None,skiprows=1)
Y=pd.read_csv('labels.csv',header=None,skiprows=1)
labels=[0,1,2,3,4,5,6,7,8]

####normalize SPOdata####
def normalized(data):
    df_scaled = data.copy()
    for column in df_scaled.columns:
        df_scaled[column] = df_scaled[column] / df_scaled[column].abs().max()
        return df_scaled
normalizedSPO=normalized(SPOdata)
normalizedSPO=pd.DataFrame(normalizedSPO)
Y=pd.DataFrame(Y)
normalizedSPO=normalizedSPO.to_numpy()
Y=Y.to_numpy()
print(normalizedSPO[0:5])
print(Y[0:5])
####data split####
X_traing_data,X_testing_data,Y_training_data,Y_testing_data=train_test_split(normalizedSPO,Y,test_size=0.3,random_state=25)
####train model####
model=models.Sequential()
model.add(layers.Flatten())
#model.add(layers.Dense(50,activation=tf.nn.relu))
model.add(layers.Dense(20,activation=tf.nn.relu))
model.add(layers.Dense(9,activation=tf.nn.softmax))
np.shape(X_traing_data)
predictions = model(X_traing_data,Y_training_data)
tf.nn.softmax(predictions).numpy()
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
history=model.fit(X_traing_data, Y_training_data, epochs=5,batch_size=10,validation_split=0.1)
train_acc=model.evaluate(X_traing_data,Y_training_data)
####save trained model####
models.save_model(model,'save_model1')
####accuracy####
load_model=models.load_model('save_model1')
y_pred=load_model.predict(X_testing_data)
y_pred = np.argmax(y_pred, axis=1)
test_loss,test_acc=load_model.evaluate(X_testing_data,Y_testing_data)
print('Accuracy: %.2f' % (test_acc*100))
###plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'],loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'],loc='upper left')
plt.show()
###confusion matrix
from sklearn.metrics import confusion_matrix
mat=confusion_matrix(Y_testing_data,y_pred)
plot_confusion_matrix(conf_mat=mat,figsize=(8,8))
plt.show()