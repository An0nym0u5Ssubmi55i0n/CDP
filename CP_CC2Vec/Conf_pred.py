import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix
from jit_utils import mini_batches_DExtended
from jit_DExtended_model import DeepJITExtended
import pandas as pd

def train_calib_sets(data_size, calib_set_size):
    ids = np.arange(0, data_size)
    np.random.shuffle(ids)
    calib_set = ids[:calib_set_size]
    train_set = ids[calib_set_size:]
    return train_set, calib_set

def prepare_data(ids, msg, code, label):
    d_msg = []
    d_code = []
    d_label = []
    for id in ids:
        d_msg.append(msg[id])
        d_code.append(code[id])
        d_label.append(label[id])
    return d_msg, d_code, d_label

def prepare_ftr_data(ids, data):
    print("-----prepare_ftr_data-----------")
    print(data.shape)
    print(data[0].shape)
    print(data[0].dtype)


    ftr_id = []
    for id in ids:
        ftr_id.append(data[id])

    return ftr_id

def binarize_label(x):
    return np.round(x)

def calibrate(predictions, labels, alpha, n):
    # for each instance, make a prediction and get the softmax score of the true label of that instance
    class_1 = np.array(predictions)
    class_0 = 1 - class_1
    true_class_prob = []
    conformal_scores = []
    for i in range(len(predictions)):
        if labels[i] == 0:
            pred_score = class_0[i]
        else:
            pred_score = class_1[i]

        true_class_prob.append(pred_score)
        # compute conformal score: si = 1 − ˆf (Xi)Yi
        conformal_scores.append(1-pred_score)
    #compute quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(conformal_scores, q_level, interpolation='higher')
    print("quantile value: "+ str(qhat))
    return qhat


def calibrate_class_conditional(predictions, labels, alpha):
    # for each instance, make a prediction and get the softmax score of the true label of that instance
    class_1 = np.array(predictions)
    class_0 = 1 - class_1
    conformal_scores_0 = []
    conformal_scores_1 = []
    for i in range(len(predictions)):
        if labels[i] == 0:
            pred_score = class_0[i]
            conformal_scores_0.append(1-pred_score)
        else:
            pred_score = class_1[i]
            conformal_scores_1.append(1 - pred_score)

    n_0 = len(conformal_scores_0)
    n_1 = len(conformal_scores_1)

    #compute quantile
    q_level_0 = np.ceil((n_0 + 1) * (1 - alpha)) / n_0
    qhat_0 = np.quantile(conformal_scores_0, q_level_0, interpolation='higher')

    q_level_1 = np.ceil((n_1 + 1) * (1 - alpha)) / n_1
    qhat_1 = np.quantile(conformal_scores_1, q_level_1, interpolation='higher')
    print("quantile value for class 0: "+ str(qhat_0))
    print("quantile value for class 1: "+ str(qhat_1))

    return qhat_0, qhat_1

def get_classes_prob(prediction):
    # for each instance, make a prediction and get the softmax score of the true label of that instance
    class_1 = prediction
    class_0 = 1 - class_1
    return np.array([class_1, class_0])

def compute_conf_prediction_set(qhat, predict_scores):
    prediction_set = []
    if predict_scores[0] >= (1-qhat):
        prediction_set.append((predict_scores[0], 1))
    if predict_scores[1] >= (1-qhat):
        prediction_set.append((predict_scores[1], 0))
    return prediction_set, len(prediction_set)

def compute_class_cond_CP_set(qhat_0, qhat_1, predict_scores):
    prediction_set_0 = []
    prediction_set_1 = []

    if predict_scores[0] >= (1-qhat_1):
        prediction_set_1.append((predict_scores[0], 1))
    if predict_scores[1] >= (1-qhat_0):
        prediction_set_0.append((predict_scores[1], 0))
    return prediction_set_0, prediction_set_1

def normal_CP(i, t_predictions, t_true_labels, quantile, alpha):
    res = []
    for t in t_predictions:
        t_prob = get_classes_prob(t)
        set, size = compute_conf_prediction_set(quantile, t_prob)
        tl = t_true_labels[t_predictions.index(t)]
        correct = -1
        if size == 1:
            correct = set[0][1] == tl
        res.append((t_prob, size, set, tl, correct))

    df = pd.DataFrame(res, columns=['softmax', 'size', 'conf set', 'true label', 'Conf pred check'])
    df.to_excel('testing-ALPHA' + str(alpha) + '_' + str(i) + '_OS_res.xlsx')

def get_predictions(data, params):
    cc2ftr, pad_msg, pad_code, labels, dict_msg, dict_code = data
    batches = mini_batches_DExtended(X_ftr=cc2ftr, X_msg=pad_msg, X_code=pad_code, Y=labels)

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]
    params.embedding_ftr = cc2ftr.shape[1]
    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model = DeepJITExtended(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))
    model.eval()  # eval mode (batch norm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        all_predict, all_label = list(), list()
        for i, (batch) in enumerate(tqdm(batches)):
            ftr, pad_msg, pad_code, label = batch
            if torch.cuda.is_available():
                ftr = torch.tensor(ftr).cuda()
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(label)
            else:
                ftr = torch.tensor(ftr).long()
                pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()
            if torch.cuda.is_available():
                predict = model.forward(ftr, pad_msg, pad_code)
                predict = predict.cpu().detach().numpy().tolist()
            else:
                predict = model.forward(ftr, pad_msg, pad_code)
                predict = predict.detach().numpy().tolist()
            all_predict += predict
            all_label += labels.tolist()

    binary_predictions = list()
    for prediction in all_predict:
        binary_predictions.append(binarize_label(prediction))
    auc_score = roc_auc_score(y_true=all_label, y_score=all_predict)
    conf_matrix = confusion_matrix(all_label, binary_predictions)
    per_clas_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    print('Test data -- AUC score:', auc_score)
    print('Test data -- conf matrix:', conf_matrix)
    print('Test data -- per_clas_acc:', per_clas_acc)
    print("ended calib")
    file = open(params.save_file, "a")
    file.write("AUC score of extendedCC2Vec= " + str(auc_score) + "\n")
    file.write("confusion_matrix scores:" + str(conf_matrix) + "\n")
    file.write("per-clas accuracy scores:" + str(per_clas_acc) + "\n")
    file.write("________________________________________________\n")
    file.close()
    print("write to file calib")
    return all_predict, all_label

def class_cond_CP(i, t_predictions, t_true_labels, quantile_0, quantile_1, alpha):
    cc_res = []
    for tp in t_predictions:
        cc_t_prob = get_classes_prob(tp)
        set_0, set_1 = compute_class_cond_CP_set(quantile_0, quantile_1, cc_t_prob)
        tl = t_true_labels[t_predictions.index(tp)]
        correct_0 = -1
        correct_1 = -1
        if len(set_0) != 0 and tl == 0:
            correct_0 = True
        else:
            correct_0 = False

        if len(set_1) != 0 and tl == 1:
            correct_1 = True
        else:
            correct_1 = False
        cc_res.append((cc_t_prob, set_0, correct_0, set_1, correct_1, tl))

    cc_df = pd.DataFrame(cc_res, columns=['softmax', 'set_0', 'correct_0', 'set_1', 'correct_1', 'true label'])
    cc_df.to_excel('testing-ALPHA' + str(alpha) + '_' + str(i) + '_OS_res_class-conditional.xlsx')


