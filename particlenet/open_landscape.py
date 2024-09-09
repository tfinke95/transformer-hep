
import pandas as pd
import h5py

# Open the HDF5 file in read mode
file_path = '/home/home3/institut_thp/hreyes/Transformers/transformers_finke/datasets/TopQuarkTag/topquarktagging_data/discretized//train_top_40_30_30_forall.h5'



dat_train = pd.read_hdf(file_path, key='raw')
print(dat_train)

file_path_val = '/home/home3/institut_thp/hreyes/Transformers/transformers_finke/datasets/TopQuarkTag/topquarktagging_data/discretized//val_top_40_30_30_forall.h5'
dat_val = pd.read_hdf(file_path_val, key='raw')
print(dat_val)


file_path_all = '/home/home3/institut_thp/hreyes/Transformers/transformers_finke/datasets/TopQuarkTag/topquarktagging_data/discretized//all_top_40_30_30_forall.h5'
data_all=pd.concat([dat_train,dat_val],axis=0)
print(data_all.shape)

data_all.to_csv(file_path_all,index=False)
print(data_all)
