
import os
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import random
import string

def random_string():
    # initializing size of string
    N = 7
 
    # using random.choices()
    # generating random strings
    res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=N))
    # print result
    print("The generated random string : " + str(res))
    return str(res)
    
main_dir_discrete='/net/data_t2k/transformers-hep/JetClass/'

sig_list=['/TTBar_models/Part_pt_1/TTBar_run_test__part_pt_const128_403030_3_O0KHIRP/samples_samples_nsamples1000000_trunc_5000.h5']
bg_list=['/ZJetsToNuNu_models/Part_pt_1_zjetnunu/ZJetsToNuNu_run_test__part_pt_const128_403030_3_N5LN6TI/samples_samples_nsamples1000000_trunc_5000.h5']
num_epochs_list=[50]
dropout_list=[0.0]
num_heads_list=[4]
num_layers_list=[8]
hidden_dim_list=[256]
batch_size_list=[100]
num_events_list=[100000,999900]
num_const_list=[128]
lr_list=[.001]

tag_of_train='top_vs_qcd_transformerdata_pt_part_classifier_test_128const_1M'
log_dir='/net/data_t2k/transformers-hep/JetClass/Classification_optclass/'+tag_of_train

for sig in sig_list:
    for bg in bg_list:
        sig_path=main_dir_discrete+sig
        bg_path=main_dir_discrete+bg

        for num_events in num_events_list:
            for num_const in num_const_list:
                for batch_size in batch_size_list:
                    for num_epochs in num_epochs_list:
                                for num_layers in num_layers_list:
                                    for dropout in dropout_list:
                                        for num_heads in num_heads_list:
                                            for lr in lr_list:
                                                for hidden_dim in hidden_dim_list:
                                                
                                                
                                                    name_sufix=random_string()
                                                    train_command='python train_classifier.py   --log_dir '+str(log_dir)+' --bg '+str(bg_path)+' --sig '+str(sig_path)+' --num_const '+str(num_const)+' --num_epochs '+str(num_epochs)+'  --lr '+str(lr)+' --batch_size '+str(batch_size)+' --num_events '+str(num_events)+' --dropout '+str(dropout)+' --num_heads '+str(num_heads)+' --num_layers '+str(num_layers)+' --hidden_dim '+str(hidden_dim)+' --name_sufix '+str(name_sufix)
                                                    os.system(train_command)

