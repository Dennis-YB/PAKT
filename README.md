# PAKT
The code for paper "A Prompt-Aware Knowledge-Tuning Framework for Histopathology Subtype Classification with Scarce Annotation"

In order to ensure seamless execution of our experiments:

1. One must configure the dataset paths within files such as 'dataset' and 'util'. Additionally, it is imperative to download the requisite SSL pre-trained models for the tasks at hand.
   
2. Subsequently, the 'get_feature.py' script should be employed to extract the features of the prompt, which will then be archived as .npy files for future utilization.
  
3. Finally, by executing the 'train_and_test.py' script, one can initiate the training and evaluation processes for the model.
