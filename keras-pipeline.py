import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
import datetime as dt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import sklearn as sk
import sklearn.preprocessing as proc
from keras.utils import plot_model
import pydot

import graphviz


def expand_data(xdata,timesteps):

    data_shape=(xdata.shape[0],timesteps,xdata.shape[1])   # define shape of expanded data to include repeated timesteps 
    
    x_large = np.zeros(data_shape)
    
    for i in range(timesteps,xdata.shape[0]-1):
        for j in range(0,timesteps):
            x_large[i,j,:]=xdata[i-timesteps+j,:]
            x_large[i,j,:]=xdata[i-j,:] # reversed version

            

    return x_large

        
            

    



data=pd.read_csv('../merged_grid_and_weather.csv')

drop_features=['PCP06','PCP24','SKC','GUS']

data_pruned=data.drop(drop_features, axis=1)

data_pruned['DateTime']=pd.to_datetime(data_pruned['DateTime'])
#print(data_pruned['DateTime'])

data_pruned=data_pruned.set_index('DateTime')
data_pruned['DateTime']=data_pruned.index
#print data_pruned
data_resampled=data_pruned.resample('H').mean()
data_resampled=data_resampled.fillna(method='pad')

data_resampled['DateTime']=data_resampled.index
#print data_resampled

#add columns for year, year day, and hour
data_resampled['year'] = data_resampled['DateTime'].apply(lambda x: x.timetuple().tm_year-2014)
data_resampled['y_day'] = data_resampled['DateTime'].apply(lambda x: x.timetuple().tm_yday)
data_resampled['hour'] = data_resampled['DateTime'].apply(lambda x: x.timetuple().tm_hour)
#print data_resampled

data_resampled=data_resampled.drop('DateTime',axis=1)
#shifting data

shifted_realtime=data_resampled[['HB_NORTH','LZ_RAYBN']].shift(-1,freq='24H')
shifted_realtime.columns=['HB_NORTH_24H','LZ_RAYBN_24H']
#print data_resampled
#print shifted_realtime

full_data=pd.merge(data_resampled,shifted_realtime,how='inner',left_index=True,right_index=True)
# print data_pruned

#print(full_data[['HB_NORTH','LZ_RAYBN','HB_NORTH_24H','LZ_RAYBN_24H']])

# reshape data
timesteps=1;
full_data=full_data.fillna(0)
time=full_data.index

x_train=full_data.drop(['HB_NORTH_24H','LZ_RAYBN_24H'],axis=1)
#print(x_train)
x_train=proc.normalize(x_train,axis=0)

y_train=full_data[['HB_NORTH_24H','LZ_RAYBN_24H']]
#print(y_train)
x_train=x_train[:24000,:]

newData=expand_data(x_train,48)

print "expanded data"
print newData.shape
print newData[11,:,0:3]
print newData[12,:,0:3]
print newData[13,:,0:3]

x_train=x_train.reshape(x_train.shape[0]/timesteps,timesteps,x_train.shape[1])
y_train=y_train.as_matrix()


y_train=proc.normalize(np.nan_to_num(y_train),axis=0)

test_split=20000

y=y_train
y_train=y[48:test_split,:]
y_test=y[test_split:24000,:]





print(x_train.shape)
print y_train[:,0]
print x_train[:,0,1]
x_train=newData[48:test_split,:,:]
x_test=newData[test_split:24000,:,:]

print x_test.shape

#val split
# design network
model = Sequential()
model.add(LSTM(5,return_sequences=False,input_shape=(x_train.shape[1], x_train.shape[2])))
#model.add(LSTM(25))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(2))
model.compile(loss='mse', optimizer='adam')
#fit network
history = model.fit(x_train, y_train[0::timesteps], epochs=20, batch_size=7200, validation_split=0.10,verbose=2, shuffle=False)
#history = model.fit(x_train, x_train[0::timesteps], epochs=5, batch_size=720, validation_split=0.10,verbose=2, shuffle=False)

# plot history
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')



plot_model(model, to_file='model.png',show_shapes=True)
yhat=model.predict(x_train,batch_size=x_train.shape[0])
yhat_test=model.predict(x_test,batch_size=x_test.shape[0])
plt.plot(time[48:test_split],y_train[:,0],time[48:test_split],yhat[:,0])
plt.figure()
plt.plot(time[test_split:24000],y_test[:,0],time[test_split:24000],yhat_test[:,0])
plt.legend()

plt.show()
