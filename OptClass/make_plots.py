import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
from data_eval_helpers import make_continues,LoadSGenamples, LoadTrue



def ReadHDF(outpult_path):


    #tmp = pd.read_hdf(outpult_path, key="raw", stop=None)
    tmp = h5py.File(outpult_path, 'r')
    print(np.array(tmp.get('raw')))
    
    
    ptj=np.array(tmp.get('ptj'))
    m_jet=np.array(tmp.get('m_jet'))
    
    raw=np.array(tmp.get('raw'))
    
    return raw, ptj, m_jet
    
    

def PlotMjet(mj,outpult_path):
    path_to_plots=outpult_path
    mj_bins = np.linspace(-20, 500, 100)
    plt.hist(np.clip(mj, mj_bins[0], mj_bins[-1]), bins=mj_bins,color='blue',histtype='step',density=True)
    plt.yscale('log')
    plt.xlabel('$m_{jet}$')
    plt.savefig(path_to_plots+'/plot_mj_trans_1M.png')
    plt.close()
    
    return


def PlotPTjet(ptj,outpult_path):
    path_to_plots=outpult_path
    ptj_bins = np.linspace(-20, 1500, 100)
    plt.hist(np.clip(ptj, ptj_bins[0], ptj_bins[-1]), bins=ptj_bins,color='blue',histtype='step',density=True)
    plt.yscale('log')
    plt.xlabel('$p_{jet}$')
    plt.savefig(path_to_plots+'/plot_ptj_trans_1M.png')
    plt.close()
    
    return


outpult_path='/Users/humbertosmac/Documents/work/Transformers/Optimal_Classifier/TheSamples/OptimalClassifierSamples/qcd/raw/'

output_file=outpult_path+'/train_nsamples1M_trunc_5000.h5'
n_samples=1000000


raw, ptj, mj=ReadHDF(output_file)
PlotMjet(mj,outpult_path)
PlotPTjet(ptj,outpult_path)
