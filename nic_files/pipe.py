import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
import datetime as dt
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import sklearn as sk
import sklearn.preprocessing as proc
from keras.utils import plot_model
from sklearn.model_selection import train_test_split


def clean_data(data):
	drop_features=['PCP06','PCP24','SKC','GUS', 'Unnamed: 0', 'Unnamed: 0.1', 'EB1_MNSES_RealTime', 'AMELIA2_8W_DayAhead']
	data_pruned=data.drop(drop_features, axis=1)
	data_pruned['DateTime']=pd.to_datetime(data_pruned['DateTime'])

	data_pruned=data_pruned.set_index('DateTime')
	data_pruned['DateTime']=data_pruned.index

	data_resampled=data_pruned.resample('H').mean()
	data_resampled=data_resampled.fillna(method='pad')

	data_resampled['DateTime']=data_resampled.index

	#add columns for year, year day, and hour
	data_resampled['year'] = data_resampled['DateTime'].apply(lambda x: x.timetuple().tm_year-2014)
	data_resampled['y_day'] = data_resampled['DateTime'].apply(lambda x: x.timetuple().tm_yday)
	data_resampled['hour'] = data_resampled['DateTime'].apply(lambda x: x.timetuple().tm_hour)
	data_resampled['w_day'] = data_resampled['DateTime'].apply(lambda x: x.timetuple().tm_wday)

	data_resampled=data_resampled.drop('DateTime',axis=1)

	shifted_realtime=data_resampled[['HB_NORTH_RealTime','LZ_RAYBN_RealTime']].shift(-1,freq='24H')
	shifted_realtime.columns=['HB_NORTH_24H','LZ_RAYBN_24H']

	full_data=pd.merge(data_resampled,shifted_realtime,how='inner',left_index=True,right_index=True)
	return full_data

def expand_data(xdata,timesteps):
	data_shape=(xdata.shape[0],timesteps,xdata.shape[1])   # define shape of expanded data to include repeated timesteps 
	x_large = np.zeros(data_shape)
	for i in range(timesteps,xdata.shape[0]-1):
		for j in range(0,timesteps):
			x_large[i,j,:]=xdata[i-timesteps+j,:]
	return x_large

def create_splits(data, timesteps, features, target):
	time=data.index

	# data = proc.scale(data,axis=0)
	# data = proc.normalize(data, axis=0)

	data = data[features+target]

	data = np.array(data)
	newData=expand_data(data,timesteps)

	X = np.array(newData[:,:,:-1])
	Y = np.array(newData[:,:,-1:])

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

	return X_train, X_test, y_train, y_test

def fit_model(X_train, X_test, y_train, y_test, n_layers, size, loss='mse'):
	input_shape=(X_train.shape[1], X_train.shape[2])

	model = Sequential()
	model.add(LSTM(size,return_sequences=True,input_shape=input_shape,activation='tanh'))
	for _ in range(n_layers-2):
		model.add(LSTM(size, return_sequences=True, activation='tanh'))
	model.add(LSTM(size, return_sequences=False, activation='tanh'))
	model.add(Dense(1))

	model.compile(loss=loss, optimizer='adam')

	# get only last value for targets
	y_train = np.squeeze(y_train[:,-1,:])
	y_test = np.squeeze(y_test[:,-1,:])

	history = model.fit(X_train, y_train, epochs=50, batch_size=720, validation_data=(X_test,y_test),verbose=2, shuffle=True)


if __name__ == '__main__':
	data = clean_data(pd.read_csv('all_the_data.csv'))

	n_layers = 3
	size = 50
	timesteps = 1
	loss = 'mae'
	features = ['HB_NORTH_RealTime','OilBarrelPrice','year', 'y_day', 'hour']
	target = ['HB_NORTH_24H']

	X_train, X_test, y_train, y_test = create_splits(data, timesteps, features, target)

	fit_model(X_train, X_test, y_train, y_test, n_layers, size, loss=loss)
