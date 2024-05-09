import pandas as pd
import matplotlib.pyplot as plt
from data_eval_helpers import make_continues,Make_Plots,LoadTrue,LoadSGenamples,GetHighLevel,Wasserstein_distance
import os
import numpy as np

'''
def readFrameCont(path,n_samples):
    tmp = pd.read_hdf(discrete_truedata_filename, key="discretized", stop=None)
    tmp=tmb.sample(n_samples)
    return tmp


bins_path_prefix='../preprocessing_bins/'
bin_tag='10M_TTBar_ZJetsToNuNu'
pt_bins = np.load(bins_path_prefix+'pt_bins_'+bin_tag+'.npy')
eta_bins = np.load(bins_path_prefix+'eta_bins_'+bin_tag+'.npy')
phi_bins = np.load(bins_path_prefix+'phi_bins_'+bin_tag+'.npy')

n_test_samples=100000
file_name_samples='/net/data_t2k/transformers-hep/JetClass/discretized/ALL_1M_val___10M_ALL_1M.h5'

jets,ptj,mj=LoadSGenamples(file_name_samples,pt_bins,eta_bins,phi_bins,n_test_samples)

discrete_truedata_filename='/net/data_t2k/transformers-hep/JetClass/discretized/ALL_1M_test___10M_ALL_1M.h5'

jets_true,ptj_true,mj_true=LoadTrue(discrete_truedata_filename,n_test_samples,pt_bins,eta_bins,phi_bins)


path_to_plots='test_plots_join_all_test_val_'+str(n_test_samples)
os.makedirs(path_to_plots,exist_ok=True)
Make_Plots(jets,pt_bins,eta_bins,phi_bins,mj,jets_true,ptj_true,mj_true,path_to_plots)
'''

def GetEvalDataJoined(file_dir):


    file_qcd=file_dir+'/results_test_eval_qcd_nsamples200000.npz'
    evalprob_qcd = np.load(file_qcd)
    
    file_top=file_dir+'/'+'/results_test_eval_top_nsamples200000.npz'
    evalprob_top = np.load(file_top)



    return evalprob_top['probs'],evalprob_qcd['probs']
    

def GetEvalDataTop(file):

    evalprob_top_best = np.load(file)

    return evalprob_top_best



def GetEvalDataQCD(file):

  
    evalprob_qcd_best = np.load(file)

    return evalprob_qcd_best
    
    
def plot_probs(evalprob_best,evalprob_joined,path_to_plots,tag,plot_title):

 plt.hist(evalprob_best,histtype='step',bins=50,density=True,color='blue',label='single')
 plt.hist(evalprob_joined,histtype='step',bins=50,density=True,color='blue',linestyle='--',label='joined')
 
 plt.xlabel('log(p)')
 plt.legend()
 plt.title(plot_title,loc='left')
 plt.savefig(path_to_plots+'/plot_probs_'+tag+'.png')
 plt.close()

 return

def BayesFactor(evalprob,evalprob_true):


    LR_statistic =  (np.sum(evalprob) /np.sum(evalprob_true))
    
    
    return LR_statistic


test_results_dir='/net/data_t2k/transformers-hep/JetClass/'

joined_file_dir=test_results_dir+'/TTBar_ZJetsToNuNu_models/'
qcd_file_name=test_results_dir+'ZJetsToNuNu_models/ZJetsToNuNu_run_testwall_10M_6/results_test_eval_nsamples200000.npz'
top_file_name=test_results_dir+'/TTBar_models/TTBar_run_testwall_10M_11/results_test_eval_nsamples200000.npz'



evalprob_top_best=GetEvalDataTop(qcd_file_name)['probs']
evalprob_qcd_best=GetEvalDataQCD(top_file_name)['probs']


joined_result_tag='TTBar_ZJetsToNuNu_run_test_joined_403030_'
joined_result_list=['0S1DG44','GGZNTEU']

for joined_result in joined_result_list:

    
    path=joined_file_dir+'/'+joined_result_tag+joined_result+'/'
    print(path)
    evalprob_top,evalprob_qcd=GetEvalDataJoined(path)
    
    
    bayes_factor=BayesFactor(evalprob_top,evalprob_top_best)
    print(bayes_factor)
    plot_probs(evalprob_top_best,evalprob_top,path,'TTBar','TTBar -- bayes_factor:'+str(bayes_factor))
    
    
    bayes_factor=BayesFactor(evalprob_qcd,evalprob_qcd_best)
    print(bayes_factor)
    plot_probs(evalprob_qcd_best,evalprob_qcd,path,'QCD','QCD -- bayes_factor:'+str(bayes_factor))
