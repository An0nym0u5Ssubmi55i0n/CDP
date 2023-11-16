# CC2Vec: Distributed Representations of Code Changes [[pdf](https://arxiv.org/pdf/2003.05620.pdf)]

## Contact
Questions and discussion are welcome: vdthoang.2016@smu.edu.sg

## Implementation Environment

Please install the neccessary libraries before running our tool:

- python==3.6.9
- torch==1.2.0
- tqdm==4.46.1
- nltk==3.4.5
- numpy==1.16.5
- scikit-learn==0.22.1

## Data & Pretrained models:

Please following the link below to download the data. 

- https://zenodo.org/record/3965149#.X2VeP5MzY1J


## Running and evalutation
##. Just-in-time defect prediction

- For each dataset in just-in-time defect prediction (qt or openstack), we create two variants: one for training code changes features ('.pkl'), the other one for training just-in-time defect prediction model (end with '_dextend.pkl'). 

- Please run this command to train the code changes features:

      $ python jit_cc2ftr.py -train -train_data [path of our training data] -test_data [path of our training data] -dictionary_data [path of our dictionary data]

- Similar to the second task, the command will create a folder snapshot used to save our model. To extract the code change features, please follow this command:

      $ python jit_cc2ftr.py -predict -predict_data [path of our data] -dictionary_data [path of our dictionary data] -load_model [path of our model] -name [name of our output file]
      
- To train the model for just-in-time defect prediction, please follow this command: 

      $ python jit_DExtended.py -train -train_data [path of our data] -train_data_cc2ftr [path of our code changes features extracted from training data] -dictionary_data [path of our dictionary data]
      
- To evaluate the model for just-in-time defect prediction, please follow this command:
      
       $ python jit_DExtended.py -predict -pred_data [path of our data] -pred_data_cc2ftr [path of our code changes features extracted from our data] -dictionary_data [path of our dictionary data] -load_model [path of our model]

## Contact

Questions and discussion are welcome: vdthoang.2016@smu.edu.sg
