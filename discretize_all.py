

#####CHECK WHICH ONE IS MISSING : IS ZToQQ !!!

list_of_jets=['TTBar','ZJetsToNuNu','HToBB','HToCC','HToGG','HToWW2Q1L','HToWW4Q','TTBarLep','WToQ','ZToQQ']

types_list=['train','test','validation']


for jet in list_of_jets:
    for type in types_list:

        command='python preprocess_jetclass.py --input_file  /net/data_t2k/transformers-hep/JetClass/'+type+'/'+jet+'_+'type'+.h5  --class_label 1 --nBins 40 30 30 --nJets 10000000 --tag 10M_'+jet
        os.system(command)
    
