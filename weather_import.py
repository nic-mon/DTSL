import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
import datetime as dt


if not (os.path.exists('../dallas_weather.pkl')):
    hourly_data = pd.read_csv('../hourly_weather_2_dat.txt',delim_whitespace=True,na_values=['*','**','***','****','*****','******'])
    #print (hourly_data)
    #uasf code numbers for dfw area weather stations
    dfw=722590
    lovefield=722580
    fort_worth=747390
    #hourly_data = pd.read_csv('../hourly_weather_2_dat.txt',delim_whitespace=True)
    #index=hourly_data.index[hourly_data['USAF']]
    index=hourly_data.loc[0:hourly_data.size-1,'USAF']==dfw
    #print(hourly_data[index])

    dallas_data=hourly_data.loc[index]
    dallas_data=dallas_data.replace('0.00T',0.00,regex=True) # remove erroneous trace values

    # one hot encoding for sky conditions
    overcast = dallas_data.loc[:,'SKC']=='OVC'
    clear = dallas_data.loc[:,'CLR']=='CLR'
    broken = dallas_data.loc[:,'BKN']=='BKN'
    scattered = dallas_data.loc[:,'SCT']=='SCT'


    #dallas_data.loc[:,'YR--MODAHRMN']=str(hourly_data['YR--MODAHRMN'])
    #print (dallas_data['YR--MODAHRMN'])
    dallas_data.loc[:,'YR--MODAHRMN']=pd.to_datetime(dallas_data.loc[:,'YR--MODAHRMN'],format='%Y%m%d%H%M')
    # (dallas_data['YR--MODAHRMN'])
    time=pd.to_datetime(201710011353,format='%Y%m%d%H%M')
    #print(time)


    #clean up trace issues in pcp01
    #for row in dallas_data.itertuples():
            #print(row)
    #	if row.PCP01=='0.00T':
    #		print(row)
                    #print( row.Index )
    #		dallas_data.loc[row.Index,'PCP01']=0.00
    #		dallas_data.loc[row.Index,'PCP06']=np.nan

                    #print (row)
    dallas_data.to_pickle('../dallas_weather.pkl')

else:
    dallas_data=pd.read_pickle('../dallas_weather.pkl')
    dallas_data.to_csv('../dallas_weather.csv')
    
ax=dallas_data.plot('YR--MODAHRMN','TEMP')
ax=dallas_data.plot('YR--MODAHRMN','DEWP')
ax=dallas_data.plot('YR--MODAHRMN','SPD')

ax=dallas_data.plot('YR--MODAHRMN','CLG')
ax=dallas_data.plot('YR--MODAHRMN','VSB')
#print (dallas_data['PCP01'])

dallas_data.loc[:,'PCPXX']=pd.to_numeric(dallas_data['PCPXX'],errors='coerce')
dallas_data.loc[:,'PCP24']=pd.to_numeric(dallas_data['PCP24'],errors='coerce')
dallas_data.loc[:,'PCP06']=pd.to_numeric(dallas_data['PCP06'],errors='coerce')
dallas_data.loc[:,'PCP01']=pd.to_numeric(dallas_data['PCP01'],errors='coerce')
ax=dallas_data.plot('YR--MODAHRMN','PCP06',kind='hist')
ax=dallas_data.plot('YR--MODAHRMN','PCP01',kind='area')
#ax=dallas_data.plot('YR--MODAHRMN','PCP24',kind='hist')
ax=dallas_data.plot('YR--MODAHRMN','PCPXX',kind='hist')

print(dallas_data.loc[:,'PCP01'].max())
print(dallas_data.loc[:,'PCP01'].sum())
print(dallas_data.loc[:,'PCP06'].sum())
print(dallas_data.loc[:,'PCP24'].sum())

dallas_data=dallas_data.drop(['WBAN','L','M','H','MW','MW.1','MW.2','MW.3','AW','AW.1','AW.2','AW.3','W','MAX','MIN','SD','PCPXX'],1)
dallas_data.loc[:,'PCP01']=dallas_data.loc[:,'PCP01'].fillna(value=0)   # change nans to 0 for precip
dallas_data.loc[dallas_data.loc[:,'DIR']>360,'DIR']=0   # change windspeeds of over 360 to 0
print(list(dallas_data))
print(dallas_data)
dallas_data.loc[:,'YR--MODAHRMN']=dallas_data.loc[:,'YR--MODAHRMN'].dt.round('1H')
ax=dallas_data.plot('YR--MODAHRMN','DIR')
ax=dallas_data.plot('YR--MODAHRMN','PCP01',kind='area')
print(list(dallas_data))
print(dallas_data)

dallas_data=dallas_data.drop_duplicates(keep='first',subset='YR--MODAHRMN')


mask=(dallas_data.loc[:,'YR--MODAHRMN']>=dt.datetime(2014,1,1, 1)) & (dallas_data.loc[:,'YR--MODAHRMN']<= dt.datetime(2017,5,31,23))

dallas_data=dallas_data.loc[mask]
dallas_data.rename(columns={'YR--MODAHRMN': 'DateTime'},inplace=True)
print(dallas_data)

grid_data=pd.read_csv('../real_time_grid_data.csv')
dallas_data.to_csv('../weather_data.csv')
print(grid_data)
grid_data['DateTime']=pd.to_datetime(grid_data['DateTime'])
merged_data=pd.merge(dallas_data,grid_data,on='DateTime',how='inner')
ax=merged_data.plot('DateTime','WOODROW69W')
print(merged_data)
plt.show()
