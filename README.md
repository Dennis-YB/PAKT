# PAKT
The code for paper "A Prompt-Aware Knowledge-Tuning Framework for Histopathology Subtype Classification with Scarce Annotation"

In order to ensure seamless execution of our experiments:

1. One must configure the dataset paths within files such as 'dataset' and 'util'. Additionally, it is imperative to download the requisite SSL pre-trained models for the tasks at hand.
   
2. Subsequently, the 'get_feature.py' script should be employed to extract the features of the prompt, which will then be archived as .npy files for future utilization.
  
3. Finally, by executing the 'train_and_test.py' script, one can initiate the training and evaluation processes for the model.

The address of TCGA dataset is here: https://portal.gdc.cancer.gov/repository?facetTab=cases&filters=%7B%22content%22%3A%5B%7B%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-KIRC%22%5D%7D%2C%22op%22%3A%22in%22%7D%2C%7B%22content%22%3A%7B%22field%22%3A%22files.experimental_strategy%22%2C%22value%22%3A%5B%22Tissue%20Slide%22%5D%7D%2C%22op%22%3A%22in%22%7D%5D%2C%22op%22%3A%22and%22%7D&searchTableTab=cases

The address of BRACS dataset is here: https://www.bracs.icar.cnr.it/
