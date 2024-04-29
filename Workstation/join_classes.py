import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def read_input(input_file,nJets=None):
    
        '''
        es = [f"E_{i}" for i in range(200)]
        px = [f"PX_{i}" for i in range(200)]
        py = [f"PY_{i}" for i in range(200)]
        pz = [f"PZ_{i}" for i in range(200)]
        cols = [item for sublist in zip(es, px, py, pz) for item in sublist]
        '''
        df = pd.read_hdf(
            input_file,
            key="raw",
            stop=nJets,
        )
        '''
        df = df[df["is_signal_new"] == class_label]
        df = df[cols]
        
        data = data.reshape((-1, 200, 4))
        '''
        data = df.to_numpy()
        return data


input_file='/net/data_t2k/transformers-hep/JetClass/val/TTBar_val.h5'
data=read_input(input_file)

print(np.shape(data))

