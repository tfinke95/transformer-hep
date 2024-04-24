import os
import numpy as np
###make sufix a random number



def random_string():
    # initializing size of string
    N = 7
 
    # using random.choices()
    # generating random strings
    res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=N))
    # print result
    print("The generated random string : " + str(res))
    return str(res)
####Trainning parameters

#list_of_jets=['TTBar','ZJetsToNuNu','HToBB','HToCC','HToGG','HToWW2Q1L','HToWW4Q','TTBarLep','WToQ','ZToQQ']
list_of_jets=['TTBar']
num_const_list=[50]
num_epochs_list=[3]
lr_list=[.001,.0005]
lr_decay_list=[.000001]
num_events_list=[600]
dropout_list=[0.1]
num_heads_list=[4]
num_layers_list=[8]
num_bins_list=["41 31 31"]
weight_decay_list=[0.00001]
hidden_dim_list=[256]
batch_size_list=[100]

###Sampling parameters
num_samples_test_list=[200]
train_batch_size=100
num_const_test=100
trunc_test_list=[5000]
model_name='model_best.pt'

main_dir='/net/data_t2k/transformers-hep/JetClass/'
bins_path_prefix='preprocessing_bins/'
###Training stage
for jet in list_of_jets:

    mother_dir='/net/data_t2k/transformers-hep/JetClass/'+jet+'_models/'
    tag_oftrain=jet+'_run_testscan_600k'
    data_path='/net/data_t2k/transformers-hep/JetClass/discretized/'+jet+'_train___10M_'+jet+'.h5'
    model_path=mother_dir+'/'+tag_oftrain
    log_dir='/net/data_t2k/transformers-hep/JetClass/'+jet+'_models/'+tag_oftrain
    output='linear'
    
    ###for sampling
    

    test_dataset='/net/data_t2k/transformers-hep/JetClass/discretized/'+jet+'_test___10M_TTBar.h5'
    
    
    ###for plotting samples
    bin_tag='10M_'+str(jet)
    pt_bins = np.load(bins_path_prefix+'pt_bins_'+bin_tag+'.npy')
    eta_bins = np.load(bins_path_prefix+'eta_bins_'+bin_tag+'.npy')
    phi_bins = np.load(bins_path_prefix+'phi_bins_'+bin_tag+'.npy')

    n_samples=200000
    test_data_name='/discretized/'+jet+'_test___10M_'+jet+'.h5'
    discrete_truedata_filename=main_dir+test_data_name
    jets_true,ptj_true,mj_true=LoadTrue(discrete_truedata_filename,n_samples)
    print(mj_true)


    for num_events  in num_events_list:
        for num_const in num_const_list:
            for batch_size in batch_size_list:
                for num_bins in num_bins_list:
                    for num_epochs in num_epochs_list:
                        for lr_decay in lr_decay_list:
                            for weight_decay in weight_decay_list:
                                for num_layers in num_layers_list:
                                    for dropout in dropout_list:
                                        for num_heads in num_heads_list:
                                            for lr in lr_list:
                                                for hidden_dim in hidden_dim_list:
                                                
                                                
                                                    name_sufix=random_string()
                                                
                                                    os.system('python train_2.py --data_path '+str(data_path)+' --model_path '+str(model_path)+' --log_dir '+str(log_dir)+'  --output '+str(output)+' --num_const '+str(num_const)+' --num_epochs '+str(num_epochs)+'  --lr '+str(lr)+' --lr_decay '+str(lr_decay)+' --batch_size '+str(batch_size)+' --num_events '+str(num_events)+' --dropout '+str(dropout)+' --num_heads '+str(num_heads)+' --num_layers '+str(num_layers)+' --num_bins '+str(num_bins)+' --weight_decay '+str(weight_decay)+' --hidden_dim '+str(hidden_dim)+' --end_token --start_token '+' --name_sufix '+str(name_sufix))



                                                    name_sufix=random_string()
                                                    
                                                    model_path_curr=model_path+'_'+name_sufix
                                                    
                                                    command_sample= 'python sample_jets_0.py --model_dir '+model_path_curr+' --savetag '+str(tag_forsample)+' --num_samples '+str(num_samples)+' --num_const '+str(num_const)+' --trunc '+str(trunc)+' --batchsize '+str(train_batch_size)+' --model_name '+model_name
                                                    print(command_sample)
                                                    os.system(command_sample)
                                                    
                                                    
                                                    
                                                    ####plotting
                                                    filename=model_path_curr+'/samples_'+tag_forsample+'.h5'

                                                    path_to_plots=test_results_dir+'/'+tag_oftrain+'_'+str(j)+'/'+tag_forsample
                                                    os.makedirs(path_to_plots,exist_ok=True)

                                                    #filename = test_results_dir+"/ttbar_run_b_2_6/samples_test_sample_200k.h5"
                                                    tmp = pd.read_hdf(filename, key="discretized", stop=None)
                                                    tmp = tmp.to_numpy()[:, :300].reshape(len(tmp), -1, 3)
                                                    print(tmp.shape)

                                                    mask = tmp[:, :, 0] == -1
                                                    jets,ptj,mj = make_continues(tmp, mask,pt_bins,eta_bins,phi_bins, noise=False)

                                                    Make_Plots(jets,pt_bins,eta_bins,phi_bins,mj,jets_true,ptj_true,mj_true,path_to_plots)
