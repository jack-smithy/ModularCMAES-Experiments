import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_default = pd.DataFrame()
df_small = pd.DataFrame()
df_large = pd.DataFrame()

for fid in range(1):
    for rep in range(5):
        PATH = f'./data/experiment-default-lambda-fid{fid+1}-2D-rep{rep+1}/data_f1_Sphere/IOHprofiler_f1_DIM2.dat'
        df_rep = pd.read_csv(PATH, delimiter=' ')
        df_default = pd.concat((df_default, df_rep))
        
    df_default = df_default.groupby('evaluations').mean()
        
for fid in range(1):
    for rep in range(3):
        PATH = f'./data/experiment-small-lambda-fid{fid+1}-2D-rep{rep+1}/data_f1_Sphere/IOHprofiler_f1_DIM2.dat'
        df_rep = pd.read_csv(PATH, delimiter=' ')
        df_small = pd.concat((df_small, df_rep))
        
    df_small = df_small.groupby('evaluations').mean()
        
for fid in range(1):
    for rep in range(3):
        PATH = f'./data/experiment-large-lambda-fid{fid+1}-2D-rep{rep+1}/data_f1_Sphere/IOHprofiler_f1_DIM2.dat'
        df_rep = pd.read_csv(PATH, delimiter=' ')
        df_large = pd.concat((df_large, df_rep))
        
    df_large = df_large.groupby('evaluations').mean()
        
fig, ax = plt.subplots(3, 1, sharex=True)

print(df_default.index)

ax[0].plot(df_default.iloc[:,0], df_default['raw_y'])
ax[0].set_yscale('log')

ax[1].plot(df_small.iloc[:,0], df_small['raw_y'])
ax[1].set_yscale('log')

ax[2].plot(df_large.iloc[:,0], df_large['raw_y'])
ax[2].set_yscale('log')

plt.show()