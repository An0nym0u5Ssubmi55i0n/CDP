import argparse
from padding import padding_data
import pickle
import numpy as np
import pandas as pd
from evaluation import evaluation_model
from calibration import get_predictions
from train import train_model
import os, datetime, time
from xlsxwriter import Workbook
from sklearn.feature_extraction.text import CountVectorizer
import torch
from sklearn.utils.random import sample_without_replacement


def read_args():
    parser = argparse.ArgumentParser()
    # Training our model
    parser.add_argument('-train', action='store_true', help='training DeepJIT model')
    parser.add_argument('-train_data', type=str, help='the directory of our training data')
    parser.add_argument('-dictionary_data', type=str, help='the directory of our dicitonary data')

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='predicting testing data')
    parser.add_argument('-pred_data', type=str, help='the directory of our testing data')

    # Predicting our data
    parser.add_argument('-load_model', type=str, help='loading our model')

    # Number of parameters for reformatting commits
    parser.add_argument('-msg_length', type=int, default=256, help='the length of the commit message')
    parser.add_argument('-code_line', type=int, default=10, help='the number of LOC in each hunk of commit code')
    parser.add_argument('-code_length', type=int, default=512, help='the length of each LOC of commit code')

    # Number of parameters for PatchNet model
    parser.add_argument('-embedding_dim', type=int, default=64, help='the dimension of embedding vector')
    parser.add_argument('-filter_sizes', type=str, default='1, 2, 3', help='the filter size of convolutional layers')
    parser.add_argument('-num_filters', type=int, default=64, help='the number of filters')
    parser.add_argument('-hidden_units', type=int, default=512, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training DeepJIT')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=25, help='the number of epochs')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    # CUDA
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')
    return parser


def train_calib_valid_sets(data_size, calib_set_size):
    ids = np.arange(0, data_size)
    np.random.shuffle(ids)
    percent_80 = int(len(ids) * 0.8)
    train_set = ids[:percent_80]
    calib_set = ids[percent_80: percent_80 + calib_set_size]
    valid_set = ids[percent_80 + calib_set_size:]
    return train_set, calib_set, valid_set


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

def get_classes_prob(prediction):
    # for each instance, make a prediction and get the softmax score of the true label of that instance
    class_1 = prediction
    class_0 = 1 - class_1
    return np.array([class_1, class_0])

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
    qhat = np.quantile(conformal_scores, q_level, method='higher')
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
    qhat_0 = np.quantile(conformal_scores_0, q_level_0, method='higher')

    q_level_1 = np.ceil((n_1 + 1) * (1 - alpha)) / n_1
    qhat_1 = np.quantile(conformal_scores_1, q_level_1, method='higher')
    print("quantile value for class 0: "+ str(qhat_0))
    print("quantile value for class 1: "+ str(qhat_1))

    return qhat_0, qhat_1

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

def normal_CP(t_predictions, t_true_labels, quantile, alpha):
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
    df.to_excel('testing-ALPHA' + str(alpha) + '_' + str(i) + '_res.xlsx')


def class_cond_CP(t_predictions, t_true_labels, quantile_0, quantile_1, alpha):
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
    cc_df.to_excel('testing-ALPHA' + str(alpha) + '_' + str(i) + '_res_class-conditional.xlsx')


if __name__ == '__main__':
    params = read_args().parse_args()
    # Problem setup
    calib_set_size = 1000  # number of calibration points
    alpha_0 = 0.05  # 1-alpha is the desired coverage
    alpha_1 = 0.10  # 1-alpha is the desired coverage
    alpha_2 = 0.15  # 1-alpha is the desired coverage


    if params.train is True:
        for i in range(100):
            params = read_args().parse_args()
            #  params = read_args().parse_args()
            start = time.time()
            dictionary = pickle.load(open(params.dictionary_data, 'rb'))
            dict_msg, dict_code = dictionary
            data = pickle.load(open(params.train_data, 'rb'))
            ids, labels, msgs, codes = data
            train_set_id, calib_set_id = train_calib_sets(len(ids), calib_set_size)
            # training data
            train_msg, train_code, train_label = prepare_data(train_set_id, msgs, codes, labels)
            train_label = np.array(train_label)
            pad_msg = padding_data(data=train_msg, dictionary=dict_msg, params=params, type='msg')
            pad_code = padding_data(data=train_code, dictionary=dict_code, params=params, type='code')
            data = (pad_msg, pad_code, train_label, dict_msg, dict_code)
            # calibration data
            calib_msg, calib_code, calib_label = prepare_data(calib_set_id, msgs, codes, labels)
            calib_label = np.array(calib_label)
            c_pad_msg = padding_data(data=calib_msg, dictionary=dict_msg, params=params, type='msg')
            c_pad_code = padding_data(data=calib_code, dictionary=dict_code, params=params, type='code')
            c_data = (c_pad_msg, c_pad_code, calib_label, dict_msg, dict_code)

            date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            params.save_dir = os.path.join(params.save_dir, date_time)
            params.save_file = './snapshot/' + date_time + '/results.txt'
            train_model(data=data, params=params)
            train_time = time.time()
            elapsed_train_time = train_time - start
            print("finished the training of DeepJit")
            print("starting calibration:...")
            calib_start_time = time.time()
            params.no_cuda = False
            params.filter_sizes = '1, 2, 3'
            params.load_model = './snapshot/' + date_time + '/epoch_25.pt'
            #params.load_model = './snapshot/2023-09-22_14-26-58/epoch_25.pt'
            predictions, true_labels = get_predictions(data=c_data, params=params)
            # normal calibration
            quantile_a0 = calibrate(predictions, true_labels, alpha_0, calib_set_size)
            quantile_a1 = calibrate(predictions, true_labels, alpha_1, calib_set_size)
            quantile_a2 = calibrate(predictions, true_labels, alpha_2, calib_set_size)
            #class- conditional calibration
            quantile_0_a0, quantile_1_a0 = calibrate_class_conditional(predictions, true_labels, alpha_0)
            quantile_0_a1, quantile_1_a1 = calibrate_class_conditional(predictions, true_labels, alpha_1)
            quantile_0_a2, quantile_1_a2 = calibrate_class_conditional(predictions, true_labels, alpha_2)

            calib_end_time = time.time()
            elapsed_calib_time = calib_end_time - calib_start_time
            print("end of calibration step")
            # elif params.predict is True:
            print("starting the evaluation step:...")
            params = read_args().parse_args()
            start_test = time.time()
            params.save_dir = os.path.join(params.save_dir, date_time)

            params.save_file = './snapshot/' + date_time + '/results.txt'
            #params.save_file = './snapshot/2023-09-22_14-26-58/results.txt'
            params.load_model = './snapshot/' + date_time + '/epoch_25.pt'
            #params.load_model = './snapshot/2023-09-22_14-26-58/epoch_25.pt'

            params.pred_data = './data/qt/qt_test.pkl'
            params.dictionary_data = './data/qt/qt_dict.pkl'
            #params.pred_data = './data/os/openstack_test.pkl'
            #params.dictionary_data = './data/os/openstack_dict.pkl'
            test_data = pickle.load(open(params.pred_data, 'rb'))
            test_ids, test_labels, test_msgs, test_codes = test_data
            test_labels = np.array(test_labels)

            dictionary = pickle.load(open(params.dictionary_data, 'rb'))
            test_dict_msg, test_dict_code = dictionary

            t_pad_msg = padding_data(data=test_msgs, dictionary=test_dict_msg, params=params, type='msg')
            t_pad_code = padding_data(data=test_codes, dictionary=test_dict_code, params=params, type='code')
            t_data = (t_pad_msg, t_pad_code, test_labels, test_dict_msg, test_dict_code)

            # evaluation_model(data=t_data, params=params)

            # perform validation
            t_predictions, t_true_labels = get_predictions(data=t_data, params=params)
            # normal calibration
            normal_CP(t_predictions, t_true_labels, quantile_a0, alpha_0)
            normal_CP(t_predictions, t_true_labels, quantile_a1, alpha_1)
            normal_CP(t_predictions, t_true_labels, quantile_a2, alpha_2)

            # class-conditional calibration
            class_cond_CP(t_predictions, t_true_labels, quantile_0_a0, quantile_1_a0, alpha_0)
            class_cond_CP(t_predictions, t_true_labels, quantile_0_a1, quantile_1_a1, alpha_1)
            class_cond_CP(t_predictions, t_true_labels, quantile_0_a2, quantile_1_a2, alpha_2)

            end_test = time.time()
            elapsed_test_time = end_test - start_test
            file = open(params.save_file, "a")
            file.write(" QT dataset experiment nr: " + str(i) + "\n")
            file.write(" Conformal set size: "+str(calib_set_size) + "\n")
            file.write(" qhat_0: {0:f}:, qhat_1: {1:f}:, qhat_2: {2:f}: ".format(quantile_a0, quantile_a1, quantile_a2) + "\n")
            file.write(" Alpha_0: {0:f}, Alpha_1: {1:f}, Alpha_2: {2:f} c: ".format(alpha_0, alpha_1, alpha_2) + "\n")
            file.write(" CC q-hat_0: {0:f}, q-hat_1: {1:f}: for alpha_0:".format(quantile_0_a0, quantile_1_a0) + "\n")
            file.write(" CC q-hat_0: {0:f}, q-hat_1: {1:f}: for alpha_1:".format(quantile_0_a1, quantile_1_a1) + "\n")
            file.write(" CC q-hat_0: {0:f}, q-hat_1: {1:f}: for alpha_2:".format(quantile_0_a2, quantile_1_a2) + "\n")

            file.write("DeepJit training time " + str(elapsed_train_time) + "\n")
            file.write("DeepJit calib time " + str(elapsed_calib_time) + "\n")
            file.write("DeepJit test time " + str(elapsed_test_time) + "\n")
            file.write("------------------------------------------------------------ \n")
            file.close()
            # else:
            print('--------------------------------------------------------------------------------')
            print('--------------------------Model training & testing done-------------------------')
            print('--------------------------------------------------------------------------------')

        exit()
