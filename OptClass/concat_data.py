
import pandas as pd



maindir='/net/data_t2k/transformers-hep/JetClass/ZJetsToNuNu_models//ZJetsToNuNu_run_scan_10M_N1G96CW/'
data_file_1=maindir+'samples_samples_nsamples5000000_trunc_5000.h5'
tmp_1 = pd.read_hdf(data_file_1, key="discretized", stop=None)

data_file_2=maindir+'samples_samples_nsamples_2_5000000_trunc_5000.h5'
tmp_2 = pd.read_hdf(data_file_2, key="discretized", stop=None)


tmp_all=pd.concat([tmp_1,tmp_2],axis=0)

print(tmp_all.shape)


tmp_all.to_hdf(maindir+'samples_samples_nsamples_10M_trunc_5000.h5', key='discretized')

tmp_all = pd.read_hdf(maindir+'samples_samples_nsamples_10M_trunc_5000.h5', key="discretized", stop=None)

tmp_all =tmp_all.to_numpy()[:, :600].reshape(len(tmp_all), -1, 3)
print(tmp_all)
print(np.shape(tmp_all))
