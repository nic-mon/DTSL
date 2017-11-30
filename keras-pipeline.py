import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
import datetime as dt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

data=pd.read_csv('../merged_grid_and_weather.csv')

drop_features=['PCP06','PCP24','SKC','GUS']

data_pruned=data.drop(drop_features, axis=1)

data_pruned['DateTime']=pd.to_datetime(data_pruned['DateTime'])
print(data_pruned['DateTime'])

data_pruned=data_pruned.set_index('DateTime')

data_resampled=data_pruned.resample('H').mean()
data_resampled=data_resampled.fillna(method='pad')

shifted_realtime=data_resampled[['HB_NORTH','LZ_RAYBN']].shift(-1,freq='D')
shifted_realtime.columns=['HB_NORTH_24H','LZ_RAYBN_24H']
print data_resampled
print shifted_realtime

full_data=pd.merge(data_resampled,shifted_realtime,how='inner',left_index=True,right_index=True)
# print data_pruned

print(full_data[['HB_NORTH','LZ_RAYBN','HB_NORTH_24H','LZ_RAYBN_24H']])

# design network
model = Sequential()
#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(Dense(1))
#model.compile(loss='mae', optimizer='adam')
