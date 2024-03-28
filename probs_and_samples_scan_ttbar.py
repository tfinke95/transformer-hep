import os





test_dataset='/net/data_t2k/transformers-hep/JetClass/discretized/TTBar_test_top_10M_ttbar.h5'
test_dataset_other='/net/data_t2k/transformers-hep/JetClass/discretized/ZJetsToNuNu_test_qcd_10M_zjettonunu.h5'
mother_dir='/net/data_t2k/transformers-hep/JetClass/ttbar_models/'
model_type='model_last.pt'
tag_oftrain='ttbar_run_b_2'
num_samples=200000
tag_foreval='test_eval_200k'
tag_foreval_other='test_eval_other_200k'
tag_forsample='test_sample_200k'
num_const=100

###For test samples
bg=test_dataset
bin_tag='10M_ttbar'
save_dir_tag='test_sample_test_0'
num_epochs_test=5

models_list=os.listdir(mother_dir)
for model in models_list:
    if tag_oftrain in model:
        if ('_6' not in model) and ('_5' not in model):
            continue
        model_path=mother_dir+'/'+model+'/'
        print(model)         
        
        command_eval='python evaluate_probabilities.py --model '+str(model_path)+str(model_type)+' --data '+str(test_dataset)+' --tag '+tag_foreval+' --num_const '+str(num_const)+' --num_events '+str(num_samples)
        
        
        os.system(command_eval)

        command_eval_other='python evaluate_probabilities.py --model '+str(model_path)+str(model_type)+' --data '+str(test_dataset_other)+' --tag '+tag_foreval_other+' --num_const '+str(num_const)+' --num_events '+str(num_samples)

        os.system(command_eval_other)


        command_sample= 'python sample_jets.py --model_dir '+model_path+' --savetag '+str(tag_forsample)+' --num_samples '+str(num_samples)+' --num_const '+str(num_const)
        os.system(command_sample)
        
        
        sg=model_path+'samples_'+tag_forsample+'.h5'
        save_dir=model_path+save_dir_tag
        command_test_sample='python test_samples.py --bg '+str(bg)+' --sig '+str(sg)+' --save_dir '+str(save_dir)+' --num_const '+str(num_const)+' --num_epochs '+str(num_epochs_test)+' -N '+str(num_samples)+' --bin_tag '+str(bin_tag)


        os.system(command_test_sample)

