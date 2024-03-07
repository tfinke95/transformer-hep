import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def GetDataFromQCDT():

    file_dir='test_results/full_test_qcd/_1/'
    file_name_qcd='results_test_2_nconst100_qcd.npz'
    file_name_top='results_test_2_nconst100_topfromqcd.npz'
    

    file_qcd=file_dir+'/'+file_name_qcd
    evalprob_qcdfromqcd = np.load(file_qcd)
    
    file_top=file_dir+'/'+file_name_top
    evalprob_topfromqcd = np.load(file_top)



    return evalprob_qcdfromqcd,evalprob_topfromqcd


def GetDataFromTopT():

    file_dir='test_results/full_test_top_2/'
    file_name_top='results_test_2_nconst100_top.npz'
    file_name_qcd='results_test_2_nconst100_qcdfromtop.npz'
    

    file_top=file_dir+'/'+file_name_top
    evalprob_topfromtop = np.load(file_top)
    
    file_qcd=file_dir+'/'+file_name_qcd
    evalprob_qcdfromtop = np.load(file_qcd)
    
    return evalprob_topfromtop,evalprob_qcdfromtop


def ExpectedProb(evalprob):
    exp_logp=np.log(1/(39402))*evalprob['n_const']
    return exp_logp

def ComputeLLR(evalprobT,evalprobF,type):

    s=evalprobT['probs']-evalprobF['probs']

    return s

def PlotLLR(s_qcd,s_top):
    bins = np.linspace(-100, 120, 40)
    plt.hist(s_qcd,histtype='step',bins=bins,density=True,color='blue',label='QCD')
    plt.hist(s_top,histtype='step',bins=bins,density=True,color='black',label='Top')
    plt.legend()
    plt.savefig('plot_LLR_test_1.png')
    plt.close()
    return


def ROCcurve(s_qcd,s_top):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top))
    print(roc_auc_score(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top)))
    print(fpr)
    plt.plot(fpr,tpr, label="Transformer", c="r")
    plt.savefig('plot_ROC_1.png')
    plt.close()
    
    plt.plot(tpr,1/fpr, label="Transformer", c="r")
    plt.ylim(1, 1e4)
    plt.xlim(0, 1)
    plt.xlabel(r"$\epsilon_{\rm{top}}$")
    plt.ylabel(r"$1 / \epsilon_{\rm{QCD}}$")
    plt.yscale('log')
    #plt.xscale('log')
    plt.savefig('plot_ROC2_1.png')
    plt.close()
    return


'''
def plot_probs(evalprob):

 plt.hist(evalprob['probs'],histtype='step',bins=30,density=True,color='blue')

 exp_logp=ExpectedProb(evalprob)

 plt.hist(exp_logp,histtype='step',bins=30,linestyle='--',density=True,color='blue')
 
 plt.savefig('plot_probs_test_1.png')
 plt.close()

 return
'''


def plot_probs(evalprob_qcdfromqcd,evalprob_topfromqcd,evalprob_topfromtop,evalprob_qcdfromtop):

 plt.hist(evalprob_qcdfromqcd['probs'],histtype='step',bins=30,density=True,color='blue',label='QCD')
 plt.hist(evalprob_qcdfromtop['probs'],histtype='step',bins=30,density=True,color='blue',linestyle='--')
 
 plt.hist(evalprob_topfromtop['probs'],histtype='step',bins=30,density=True,color='black',label='Top')
 plt.hist(evalprob_topfromqcd['probs'],histtype='step',bins=30,density=True,color='black',linestyle='--')
 plt.xlabel('log(p)')
 plt.legend()
 
 plt.savefig('plot_probs_test_1.png')
 plt.close()

 return



def plot_color(evalprob):

 plt.plot(evalprob['probs'],evalprob['n_const'],'.')
 plt.xlabel('n_{const}')
 plt.ylabel('log(p)')
 plt.savefig('plot_plot_test_1.png')
 plt.close()

 return
 
 
def plot_contour(evalprob_qcdfromqcd,evalprob_topfromtop):

 sns.kdeplot(data=evalprob_qcdfromqcd,x='probs',y='n_const',levels=4,color='blue', fill=True,alpha=.5)
 sns.kdeplot(data=evalprob_topfromtop,x='probs',y='n_const',levels=4,color='black', fill=True,alpha=.5)
 plt.plot(np.log(1. / 39402) * np.linspace(0, 100, 100), np.linspace(0, 100, 100), linestyle="--", color='grey')
 plt.ylabel('$n_{const}$')
 plt.xlabel('log(p)')
 plt.savefig('plot_contour_test_1.png')
 plt.close()

 return


evalprob_qcdfromqcd,evalprob_topfromqcd=GetDataFromQCDT()
evalprob_topfromtop,evalprob_qcdfromtop=GetDataFromTopT()

plot_probs(evalprob_qcdfromqcd,evalprob_topfromqcd,evalprob_topfromtop,evalprob_qcdfromtop)
#plot_contour(evalprob_qcdfromqcd,evalprob_topfromtop)


s_top=ComputeLLR(evalprob_topfromtop,evalprob_topfromqcd,'qcd')
s_qcd=ComputeLLR(evalprob_qcdfromtop,evalprob_qcdfromqcd,'qcd')
PlotLLR(s_qcd,s_top)
ROCcurve(s_qcd,s_top)
exit()

plot_color(evalprob)
plot_contour(evalprob)
