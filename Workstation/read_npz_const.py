import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from data_eval_helpers import make_continues,Make_Plots,LoadTrue,LoadSGenamples,GetHighLevel,Wasserstein_distance

def GetDataEval(file_dir):
    file_name='results_test_eval_nconst_nsamples200000.npz'
    file_name_qcd='results_test_eval_nconst_other_nsamples200000.npz'
    
    file=file_dir+'/'+file_name
    evalprob = np.load(file)

    return evalprob




def plot_probs(evalprob,num_const,color):

    plt.hist(evalprob['probs'],histtype='step',bins=30,density=True,color=color,label=str(num_const))
    plt.legend(fontsize="large")

    return
 
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


def plot_multiplicity(path_to_plots):


    mask = jets[:, :, 0] != 0
    plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 100.5, 102),color=color,histtype='step',density=True,linestyle='dashed')
    
    plt.xlabel('Multiplicity')
    plt.legend()
    plt.savefig(path_to_plots+'/plot_mul.png')
    plt.close()



    return



mother_dir='/net/data_t2k/transformers-hep/JetClass/TTBar_models/test_const_dep/'
results_tag='TTBar_run_scan_const_1M_'

results_list=os.listdir(mother_dir)

colors=['blue','green','red', 'cyan','magenta','yellow','black']

for j  in range(len(results_list)):
    result=results_list[j]
    color=colors[j]

    file_dir=mother_dir+'/'+result+'/'
    try:
        arguments_file=read_file(file_dir+'arguments.txt')
        num_const=extract_value('num_const',arguments_file)
        print(num_const)
        evalprob=GetDataEval(file_dir)
        plot_probs(evalprob,num_const,color)
    except:
        continue
    

plt.xlabel('log(p)')
plt.legend(fontsize="large")
plt.savefig(mother_dir+'plot_probs_test_1.png')
plt.close()


######## PLOT SAMPLES #######



data_dict={'name_sufix':[],'dropout':[],'lr':[],'hidden_dim':[],'num_layers':[],'num_heads':[]}
data_dict_result={'w_distance_pt':[],'w_distance_mj':[] ,'w_distance_mul':[] }


bins_path_prefix='../preprocessing_bins/'
bin_tag='10M_TTBar'
pt_bins = np.load(bins_path_prefix+'pt_bins_'+bin_tag+'.npy')
eta_bins = np.load(bins_path_prefix+'eta_bins_'+bin_tag+'.npy')
phi_bins = np.load(bins_path_prefix+'phi_bins_'+bin_tag+'.npy')

n_test_samples=200000
discrete_truedata_filename='/net/data_t2k/transformers-hep/JetClass/discretized/TTBar_test___10M_TTBar.h5'
jets_true,ptj_true,mj_true=LoadTrue(discrete_truedata_filename,n_test_samples,pt_bins,eta_bins,phi_bins)
pt_true, eta_true,phi_true,mul_true=GetHighLevel(jets_true)

mask = jets_true[:, :, 0] != 0
plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 100.5, 102),color='black',histtype='step',density=True)

for j  in range(len(results_list)):
    result=results_list[j]
    color=colors[j]

    file_dir=mother_dir+'/'+result+'/'
    
    file_name_samples=mother_dir+'/'+result+'/samples__nsamples200000_trunc_5000.h5'
    #try:
    jets,ptj,mj=LoadSGenamples(file_name_samples,pt_bins,eta_bins,phi_bins)
    print('jets')
    pt, eta,phi,mul=GetHighLevel(jets)
    print('high level')
    plot_multiplicity(mother_dir)
    print('plot')
   # except:
   '     continue