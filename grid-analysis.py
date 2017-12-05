import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('../real_time_grid_data.csv')
data=data.drop(['Unnamed: 0','DateTime','DaylightSavings'],axis=1)
data=data.fillna(0)
data=data.as_matrix()
mean=np.mean(data,axis=0)
data=data-mean
print data
U,s,V=np.linalg.svd(data[:8000,:],full_matrices=0)
print U[:,0:2].shape
print V.shape
start=0
rank=5

pc=np.dot(U[:,start:rank],np.diag(s[start:rank]))
print pc.shape
approx=np.dot(pc,V[start:rank,:])
plt.subplot(511)
plt.plot(pc[:,0])
plt.subplot(512)
plt.plot(pc[:,1])
plt.subplot(513)
plt.plot(pc[:,2])

plt.subplot(514)
plt.plot(pc[:,3])

plt.subplot(515)
plt.plot(pc[:,4])



total_var=np.sum(s)
reduced_var=np.sum(s[start:rank])
print reduced_var/total_var
#plt.plot(data[:,0])
print s
plt.show()


