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


# this function expands the data so the lstm is fed short time series sequences
def expand_data(xdata,timesteps):

    data_shape=(xdata.shape[0],timesteps,xdata.shape[1])   # define shape of expanded data to include repeated timesteps 
    
    x_large = np.zeros(data_shape)
    
    for i in range(timesteps,xdata.shape[0]-1):
        for j in range(0,timesteps):
            x_large[i,j,:]=xdata[i-timesteps+j,:]
            #x_large[i,j,:]=xdata[i-j,:] # reversed version

            

    return x_large

        
            

    


#####################Loading and Cleaning Data ###############################

data=pd.read_csv('../merged_grid_and_weather.csv')  # reads merged data

drop_features=['PCP06','PCP24','SKC','GUS']     # drop dirty or non-important data columns
data_pruned=data.drop(drop_features, axis=1) 

data_pruned['DateTime']=pd.to_datetime(data_pruned['DateTime']) # cast the date time column to datetime format

data_pruned=data_pruned.set_index('DateTime')   #sets index as a datetime index
data_pruned['DateTime']=data_pruned.index       # datetime column is also set to index, i had to do this because DateTime was removed by set_index 

data_resampled=data_pruned.resample('H').mean()     # resample data by the hour
data_resampled=data_resampled.fillna(method='pad')  # fills empty values by filling in with previous values. this needs to be improved

data_resampled['DateTime']=data_resampled.index     #creates a DateTime column from the datetime index

#add columns for year, year day, and hour
data_resampled['year'] = data_resampled['DateTime'].apply(lambda x: x.timetuple().tm_year-2014)
data_resampled['y_day'] = data_resampled['DateTime'].apply(lambda x: x.timetuple().tm_yday)
data_resampled['hour'] = data_resampled['DateTime'].apply(lambda x: x.timetuple().tm_hour)

data_resampled=data_resampled.drop('DateTime',axis=1)   #drop the datetime column


#shifting data to create y labels 

shifted_realtime=data_resampled[['HB_NORTH','LZ_RAYBN']].shift(-1,freq='24H')   #shifts grid data forward 24 hours
shifted_realtime.columns=['HB_NORTH_24H','LZ_RAYBN_24H']    # names columns

#merge input data with y labels to create a full dataset
full_data=pd.merge(data_resampled,shifted_realtime,how='inner',left_index=True,right_index=True) 

full_data=full_data.fillna(0) #fill nas with 0
print(full_data.columns)
full_data=full_data.drop(['EB1_MNSES','Unnamed: 0','USAF'],axis=1) 

################### Reshaping data so it is compatible with keras ################################

# reshape data
timesteps=1;

time=full_data.index #create an index for time that we can use to plot things

x_train=full_data.drop(['HB_NORTH_24H','LZ_RAYBN_24H'],axis=1) # create training data


x_train=proc.scale(x_train,axis=0) #scale data so it is zero mean and unit variance
x_train=proc.normalize(x_train,axis=0) #normalize data so it is u
x_train=x_train[:24000,:]   # only data datapoints up to hour 24000 

#TODO: save normalization and scaler so we can apply it consistently to test data

y_train=full_data[['HB_NORTH_24H','LZ_RAYBN_24H']]  # create y_train data




lookback=100  #the number of hours in the past that the lstm looks at

#expand data so its dimensions are nsamples X lookback X features 
newData=expand_data(x_train,lookback) 

#
#x_train=x_train.reshape(x_train.shape[0]/timesteps,timesteps,x_train.shape[1])

y_train=y_train.as_matrix()     # cast as a ndarray

#scale and normalize y_train
y_train=proc.scale(np.nan_to_num(y_train),axis=0) 
y_train=proc.normalize(np.nan_to_num(y_train),axis=0)

#set the point where samples are split for testing and training
test_split=20000

y=y_train   #save y_train in another variable, sorry this is confusing and not good practice

y_train=y[lookback:test_split,:] #takes a splice of y to create the ytrain data

y_test=y[test_split:24000,:]    #creat ytest 

# split data
x_train=newData[lookback:test_split,:,:]
x_test=newData[test_split:24000,:,:]


################## Keras Neural Network Design, Training, and Prediction ######################################################

# design network
input_shape=(x_train.shape[1], x_train.shape[2])

model = Sequential()

#network layers###########################

model.add(LSTM(50,return_sequences=False,input_shape=input_shape,activation='selu'))
#model.add(LSTM(2))
#model.add(Dense(15))
#model.add(Dense(10))
model.add(Dense(2))

#network compiling#########################

model.compile(loss='mae', optimizer='adam')

#fit network

history = model.fit(x_train, y_train[0::timesteps], epochs=50, batch_size=720, validation_split=0.0,verbose=2, shuffle=False)
#history = model.fit(x_train, x_train[0::timesteps], epochs=5, batch_size=720, validation_split=0.10,verbose=2, shuffle=False)

# plot history
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')



#plot_model(model, to_file='model.png',show_shapes=True)

#predct and plot data#######################

yhat=model.predict(x_train,batch_size=x_train.shape[0])
yhat_test=model.predict(x_test,batch_size=x_test.shape[0])
plt.plot(time[lookback:test_split],y_train[:,0],time[lookback:test_split],yhat[:,0])
plt.figure()
plt.plot(time[test_split:24000],y_test[:,0],time[test_split:24000],yhat_test[:,0])
plt.legend()

plt.show()
