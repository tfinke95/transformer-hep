import os






mother_dir='/net/data_t2k/transformers-hep/JetClass/OptClass/TTBar_models/TTBar_run_test__part_pt_1Mfromeach_403030_test_2_343QU3V/'
model_type='model_best.pt'
tag_oftrain='1Mfromeach_403030_'
num_samples_list=[200]
train_batch_size=100
num_const_list=[128]
trunc_list=[5000]
###For test samples

bin_tag='1Mfromeach_403030'


models_list=os.listdir(mother_dir)

#for result in list_of_results:
#    model=tag_oftrain+'_'+str(result)
model_path=mother_dir

for model in models_list:
    
    if tag_oftrain not in model:
    #    if ('_6' not in model) and ('_10' not in model):
        continue

    for num_const in num_const_list:
        for num_samples in num_samples_list:
            for j in range(5):
            
                tag_forsample='samples_nsamples'+str(num_samples)+'_trunc_'+str(trunc)+'_'+str(j)
        
                command_sample= 'python sample_jets_1.py --model_dir '+model_path+' --savetag '+str(tag_forsample)+' --num_samples '+str(num_samples)+' --num_const '+str(num_const)+' --trunc '+str(trunc)+' --batchsize '+str(train_batch_size)+' --model_name '+model_name
                print(command_sample)
                os.system(command_sample)



