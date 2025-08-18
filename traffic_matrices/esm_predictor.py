import pandas as pd
import numpy as np 
import os
import pickle
import sys

args = sys.argv[1:]
topo = args[0]
cwd = os.getcwd()
path = f"{cwd}"


for i in range(3):
    try:
        os.mkdir(f"{path}/{topo}_{i+1}_esm")
    except:
        pass

def read_pickle(fname, prio):
    file = open(f"{path}/{topo}_{prio}/{fname}", "rb")
    data = pickle.load(file).reshape(-1)
    file.close()
    
    return data

def write_pickle(fname, data, prio):
    file = open(f"{path}/{topo}_{prio}_esm/{fname}", "wb")
    pickle.dump(data, file)
    file.close()

tms = [i for i in os.listdir(f"{path}/{topo}_1") if i.endswith("pkl")]
tms = sorted(tms, key=lambda x: int(x.split(".")[0][1:]))

high = []
mid = []
low = []
for fname in tms:
    tm = read_pickle(fname, 1)
    high.append(tm)
    tm = read_pickle(fname, 2)
    mid.append(tm)
    tm = read_pickle(fname, 3)
    low.append(tm)

high = np.array(high)
mid = np.array(mid)
low = np.array(low)


df_high = pd.DataFrame(high)
df_mid = pd.DataFrame(mid)
df_low = pd.DataFrame(low)


for i, df in enumerate([df_high, df_mid, df_low]):
    esm = df.ewm(alpha=0.5, adjust=False).mean()
    esm = esm.iloc[:-1, :]
    
    x = pd.DataFrame(df.iloc[0, :]).transpose()
    
    esm = pd.concat([x, esm], ignore_index=True)
    for j, fname in enumerate(tms):
        temp_tm = esm.iloc[j, :].values.reshape(-1, 1)
        write_pickle(fname, temp_tm, i+1)
