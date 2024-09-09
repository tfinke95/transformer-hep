
import pandas as pd
import h5py

# Open the HDF5 file in read mode
file_path = '/home/home3/institut_thp/hreyes/Transformers/transformers_finke/datasets/TopQuarkTag/topquarktagging_data/discretized//val_qcd_all_bins403030.h5'
with h5py.File(file_path, 'r') as h5_file:
    # List all the datasets and groups in the file
    print("Keys: %s" % h5_file.keys())


dat = pd.read_hdf(file_path, key='raw', stop=1000)
print(dat)
