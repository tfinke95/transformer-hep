import pandas as pd
import matplotlib.pyplot as plt
from data_eval_helpers import make_continues,Make_Plots,LoadTrue,LoadSGenamples,GetHighLevel,Wasserstein_distance
import os
import numpy as np


def readFrameCont(path,n_samples):
    tmp = pd.read_hdf(discrete_truedata_filename, key="discretized", stop=None)
    tmp=tmb.sample(n_samples)
    return tmp


bins_path_prefix='../preprocessing_bins/'
bin_tag='10M_TTBar_ZJetsToNuNu'
pt_bins = np.load(bins_path_prefix+'pt_bins_'+bin_tag+'.npy')
eta_bins = np.load(bins_path_prefix+'eta_bins_'+bin_tag+'.npy')
phi_bins = np.load(bins_path_prefix+'phi_bins_'+bin_tag+'.npy')


file_name_samples='/net/data_t2k/transformers-hep/JetClass/TTBar_ZJetsToNuNu_models//TTBar_ZJetsToNuNu_run_scan_600k_XGNLMOG/samples__nsamples200000_trunc_5000.h5'

jets,ptj,mj=LoadSGenamples(file_name_samples,pt_bins,eta_bins,phi_bins)

discrete_truedata_filename='/net/data_t2k/transformers-hep/JetClass/discretized/TTBar_ZJetsToNuNu_test___10M_TTBar_ZJetsToNuNu.h5'
n_test_samples=200000
jets_true,ptj_true,mj_true=LoadTrue(discrete_truedata_filename,n_test_samples,pt_bins,eta_bins,phi_bins)


path_to_plots='test_plots_join_'+str(n_test_samples)
os.makedirs(path_to_plots,exist_ok=True)
Make_Plots(jets,pt_bins,eta_bins,phi_bins,mj,jets_true,ptj_true,mj_true,path_to_plots)
