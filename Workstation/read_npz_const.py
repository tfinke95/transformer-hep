import numpy as np
import matplotlib.pyplot as plt
import os


def GetDataEval(file_dir):
    file_name='results_test_eval_nconst_nsamples200000.npz'
    file_name_qcd='results_test_eval_nconst_other_nsamples200000.npz'
    
    file=file_dir+'/'+file_name
    evalprob = np.load(file)

    return evalprob



def ReadArguments():
    return

def plot_probs(evalprob,num_const,color):

 plt.hist(evalprob['probs'],histtype='step',bins=30,density=True,color=color,label=num_const)


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
plt.legend()
plt.savefig(mother_dir+'plot_probs_test_1.png')
plt.close()



