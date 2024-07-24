import os

import json
from pprint import pprint


def OpenConfig(config_file_name):
    f=open(config_file_name)
    d = json.load(f)

    data=d.get('data')

    return d

def UpdateConfig(config_dict,sig_file,bg_file,out_dir):

    config_dict.get('data').update({'sig_file':sig_file})
    config_dict.get('data').update({'bg_file':bg_file})
    config_dict.get('logging').update({'logfolder':out_dir})
    
    print(config_dict)
    return config_dict
    
def SaveNewConfig(config_dict,config_file_name):

    save_file = open(config_file_name, "w")
    json.dump(config_dict, save_file,indent=6)
    save_file.close()


    return



def train(config_file):

    os.system('python train.py '+config_file)

    return

    
def evaluate(config_file_trained):
    print('evaluating')
    os.system('python evaluate.py '+config_file_trained)
    
    return
    



config_file='config_forgen.json'


prefix_znunu='/net/data_t2k/transformers-hep/JetClass/discretized/ZJetsToNuNu_test___10M_bins_tag_ZJetsToNuNu.h5'
prefix_top='/net/data_t2k/transformers-hep/JetClass/discretized/TTBar_test___10M_bins_tag_TTBar.h5'


bg_file='/net/data_t2k/transformers-hep/JetClass/JetClass_pt_part/TTBar_test.h5'
sig_dirs_path='/net/data_t2k/transformers-hep/JetClass/TTBar_models/fine_tunning_pt_part/'

sig_dirs=os.listdir(sig_dirs_path)



for sig_dir in sig_dirs:

    sig_file=sig_dirs_path+sig_dir+'/samples__nsamples50000_trunc_5000.h5'



    
    config_dict=OpenConfig(config_file)
    out_dir='logs/retrain_gen_ttbar_50k_test_'+str(sig_dir)
    UpdateConfig(config_dict,sig_file,bg_file,out_dir)
    SaveNewConfig(config_dict,config_file)
    train(config_file)
    config_file_trained=out_dir+'/config.json'
    
    evaluate(config_file_trained)

