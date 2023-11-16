# CDP: Conformal_defect_prediction

This repository complements the DeepJIT & CC2Vec techniques with conformal prediction (ICP)

# DeepJIT: An End-To-End Deep LearningFramework for Just-In-Time Defect Prediction [[pdf](https://posl.ait.kyushu-u.ac.jp/~kamei/publications/Thong_MSR2019.pdf)]
#  CC2Vec: Distributed Representations of Code Changes[[pdf](https://2020.icse-conferences.org/track/icse-2020-papers)]

## Implementation Environment

Please install the necessary libraries before running our tool:

- python==3.6.9
- torch==1.2.0
- tqdm==4.46.1
- nltk==3.4.5
- numpy==1.16.5
- scikit-learn==0.22.1

## Data:
- https://zenodo.org/record/3965246#.XyEDVnUzY5k

## Running and evaluation
      
- To train and evaluate ICP with DeepJIT, please follow this command: 

      $ python main.py -train -train_data [path of our train data] -dictionary_data [path of our dictionary data]
      
- To train and evaluate the ICP with CC2Vec, please follow this command:
      
       $ python run_CP_CC2Vec.py -train -train_data .[path of our train data] -dictionary_data [path of our dictionary data]

