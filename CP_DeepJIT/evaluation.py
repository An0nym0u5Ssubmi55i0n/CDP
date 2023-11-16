from model import DeepJIT
from utils import mini_batches_test
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch 
from tqdm import tqdm
import numpy as np


def binarize_label(x):
    return np.round(x)

def evaluation_model(data, params):
    pad_msg, pad_code, labels, dict_msg, dict_code = data
    batches = mini_batches_test(X_msg=pad_msg, X_code=pad_code, Y=labels)

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model = DeepJIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        all_predict, all_label = list(), list()
        for i, (batch) in enumerate(tqdm(batches)):
            pad_msg, pad_code, label = batch
            if torch.cuda.is_available():                
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(label)
            else:                
                pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()
            if torch.cuda.is_available():
                predict = model.forward(pad_msg, pad_code)
                predict = predict.cpu().detach().numpy().tolist()
            else:
                predict = model.forward(pad_msg, pad_code)
                predict = predict.detach().numpy().tolist()
            all_predict += predict
            all_label += labels.tolist()
        print(len(all_predict))
        print(len(all_label))
        print("___________________")
    binary_predictions = list()
    for prediction in all_predict:
        binary_predictions.append(binarize_label(prediction))
    print(len(binary_predictions))
    print(len(all_label))
    conf_matrix = confusion_matrix(all_label, binary_predictions)
    per_clas_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)
    print('Test data -- AUC score:', auc_score)
    file = open(params.save_file, "a")
    file.write("AUC score of DeepJIT= " + str(auc_score) + "\n")
    file.write("confusion_matrix scores:" + str(conf_matrix) + "\n")
    file.write("per-clas accuracy scores:" + str(per_clas_acc) + "\n")
    file.write("________________________________________________\n")
    file.close()