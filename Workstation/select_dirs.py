import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats


def read_file(file_name):

    f = open(file_name, "r")
    lines = f.readlines()

    return lines

def extract_value(var,lines):

    for line in lines:
        if 'lr_' in line:
            continue
        if var in line:
            line=line.replace(' ','')
            line=line.replace('\n','')

            value=line.split(var)[-1]
    
    return value


main_dir='/net/data_t2k/transformers-hep/JetClass/Classification_finetune/'
dirs=os.listdir(main_dir)
val='epochs'
new_subdir='/test_dropout1/'
for dir in dirs:

    if 'optyes_3epochs_test_1' not in dir:
        continue
    
    file_name=main_dir+'/'+dir+'/arguments.txt'
    lines=read_file(file_name)
    value=int(extract_value(val,lines))
    
    if value==50:
        os.system('mv '+main_dir+'/'+dir+' '+main_dir+'/'+new_subdir+'/')
        print(dir)
    else:
        continue
    
    

    
