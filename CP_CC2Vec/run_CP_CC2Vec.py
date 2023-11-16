import argparse
import pickle
import time
import numpy as np
from jit_padding import padding_message, clean_and_reformat_code, padding_commit_code, mapping_dict_msg, mapping_dict_code, convert_msg_to_label
from jit_cc2ftr_train import train_model
from jit_DExtended import read_args as read_args_extended
from jit_cc2ftr import read_args
from jit_cc2ftr_extracted import extracted_cc2ftr
import os, datetime
from jit_DExtended_padding import padding_data
from jit_DExtended_eval import evaluation_model
from jit_DExtended_train import train_model as train_model_ext
from Conf_pred import *


def read_dict(msgs,codes):
    # read dictionary
    dictionary = pickle.load(open(dictionary_data_path, 'rb'))
    dict_msg, dict_code = dictionary

    pad_msg = padding_message(data=msgs, max_length=256)
    added_code, removed_code = clean_and_reformat_code(codes)
    pad_added_code = padding_commit_code(data=added_code, max_file=2, max_line=10, max_length=64)
    pad_removed_code = padding_commit_code(data=removed_code, max_file=2, max_line=10, max_length=64)
    pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dict_msg)
    pad_added_code = mapping_dict_code(pad_code=pad_added_code, dict_code=dict_code)
    pad_removed_code = mapping_dict_code(pad_code=pad_removed_code, dict_code=dict_code)
    pad_msg_labels = convert_msg_to_label(pad_msg=pad_msg, dict_msg=dict_msg)
    data = (pad_added_code, pad_removed_code, pad_msg_labels, dict_msg, dict_code)
    return data


def test(params, date, ext):
    params.model_name = date
    params.save_file = './snapshot/' + params.model_name + '/openstack-results.txt'
    if ext:
        params.load_model = './snapshot_ext/' + params.model_name + '/epoch_50.pt'
    else:
        params.load_model = './snapshot/' + params.model_name + '/epoch_50.pt'


def trainCC2ftr(data, params):
    train_model(data=data, params=params)
    print('----------------------------------------------------------------------------------------------------')
    print('-----------------------Finished the training of the code changes features---------------------------')
    print('----------------------------------------------------------------------------------------------------')
    end_1train_time = time.time()
    elapsed_train_time = end_1train_time - start_time
    file = open(params.save_file, "a")
    file.write("\n total time for training CC2Vec= " + str(elapsed_train_time) + "\n")
    file.close()


def testCC2ftr(data, params):
    params.load_model = './snapshot/'+params.model_name+'/epoch_50.pt'
    params.name = str(params.model_nr) +'_epoch_50.pt'
    file = open(params.name, "x")
    params.batch_size = 1
    extracted_cc2ftr(data=data, params=params)
    print('----------------------------------------------------------------------------------------------------')
    print('--------------------------Finish the extracting of the code change features-------------------------')
    print('----------------------------------------------------------------------------------------------------')
    end_test_time = time.time()
    elapsed_test_time = end_test_time - start_time
    file = open(params.save_file, "a")
    file.write("total time for training & testing CC2Vec= " + str(elapsed_test_time) + "\n")
    file.close()

def trainExtCC2ftr(train_set_id, train_ftr_set_id, params, save_file,dict_msg, dict_code):
    params.load_model = './snapshot_ext/' + params.model_name + '/epoch_50.pt'
    params.save_file = save_file
    train_data = pickle.load(open(train_data_path_ext, 'rb'))
    ids, labels, msgs, codes = train_data
    train_msg, train_code, train_label = prepare_data(train_set_id, msgs, codes, labels)
    tr_label = np.array(train_label)

    train_data_ftr = pickle.load(open(train_data_cc2ftr_path, 'rb'))
    #train_ftr = prepare_ftr_data(train_ftr_set_id, train_data_ftr)

    train_pad_msg = padding_data(data=train_msg, dictionary=dict_msg, params=params, type='msg')
    train_pad_code = padding_data(data=train_code, dictionary=dict_code, params=params, type='code')
    train_ext_data = (train_data_ftr, train_pad_msg, train_pad_code, tr_label, dict_msg, dict_code)

    train_model_ext(data=train_ext_data, params=params)
    end_trainCC2Vec_time = time.time()
    CC2Vec = end_trainCC2Vec_time - start_time
    file = open(params.save_file, "a")
    file.write("total time for training CC2Vec with features= " + str(CC2Vec) + "\n")
    file.close()
    print('-----------------------------------------------------------------------------------')
    print('-----------------------Finish the training of extended DeepJIT---------------------')
    print('-----------------------------------------------------------------------------------')

def get_calib_data(calib_set_id,calib_ftr_set_id, params, dict_msg, dict_code):
    # calibration data
    train_data = pickle.load(open(train_data_path_ext, 'rb'))
    ids, labels, msgs, codes = train_data
    calib_msg, calib_code, calib_label = prepare_data(calib_set_id, msgs, codes, labels)
    cal_label = np.array(calib_label)

    train_data_ftr = pickle.load(open(train_data_cc2ftr_path, 'rb'))
    print("inside get_calib_data")
    #calib_ftr = prepare_ftr_data(calib_ftr_set_id, train_data_ftr)

    c_pad_msg = padding_data(data=calib_msg, dictionary=dict_msg, params=params, type='msg')
    c_pad_code = padding_data(data=calib_code, dictionary=dict_code, params=params, type='code')
    c_data = (train_data_ftr, c_pad_msg, c_pad_code, cal_label, dict_msg, dict_code)
    return c_data

if __name__ == '__main__':
    start_time = time.time()
    # Problem setup
    calib_set_size = 1000  # number of calibration points
    alpha_0 = 0.05  # 1-alpha is the desired coverage
    alpha_1 = 0.10  # 1-alpha is the desired coverage
    alpha_2 = 0.15  # 1-alpha is the desired coverage
    params = read_args().parse_args()
    params_ext = read_args_extended().parse_args()
    train_data_path = './data/jit/openstack_train.pkl'
    test_data_path = './data/jit/openstack_test.pkl'
    dictionary_data_path = './data/jit/openstack_dict.pkl'
    train_data_path_ext = './data/jit/openstack_train_dextend.pkl'
    test_data_path_ext = './data/jit/openstack_test_dextend.pkl'
    train_data_cc2ftr_path = './data/jit/openstack_train_cc2ftr.pkl'
    test_data_cc2ftr_path = './data/jit/openstack_test_cc2ftr.pkl'
    # Training our model

    for i in range(100):
        # -----------prepare training of jit_cc2ftr-----------
        params = read_args().parse_args()
        # file = open("QT-output.txt", "x")
        date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        params.model_name = date_time
        params.model_nr = i
        params.save_file = './snapshot/'+params.model_name+'/openstack-results.txt'
        train_data = pickle.load(open(train_data_path, 'rb'))
        train_ids, train_labels, train_messages, train_codes = train_data

        test_data = pickle.load(open(test_data_path, 'rb'))
        test_ids, test_labels, test_messages, test_codes = test_data
        ids = train_ids + test_ids
        labels = list(train_labels) + list(test_labels)
        msgs = train_messages + test_messages
        codes = train_codes + test_codes
        data = read_dict(msgs,codes)
        #train CC2ftr
        trainCC2ftr(data, params)

        # ------------prepare training of CC2Vec + JIT------------
        params_ext = read_args_extended().parse_args()
        params_ext.name = str(i) + '_epoch_50_ext.pt'
        params_ext.model_name = params.model_name
        dictionary = pickle.load(open(dictionary_data_path, 'rb'))
        dict_msg, dict_code = dictionary

        # -----------split train/calib sets---------------------
        train_data = pickle.load(open(train_data_path_ext, 'rb'))
        train_data_ftr = pickle.load(open(train_data_cc2ftr_path, 'rb'))
        train_set_id, calib_set_id = train_calib_sets(len(train_data[0]), calib_set_size)
        train_ftr_set_id, calib_ftr_set_id = train_calib_sets(len(train_data_ftr), calib_set_size)

        #--------------train ext CC2VEC--------------------
        trainExtCC2ftr(train_set_id, train_ftr_set_id, params_ext, params.save_file, dict_msg, dict_code)

        # ----------calibration----------------
        print("starting calibration:...")
        calib_start_time = time.time()
        params_ext = read_args_extended().parse_args()
        params_ext.name = str(i) + '_epoch_50_ext.pt'
        params_ext.model_name = params.model_name
        params_ext.save_dir = os.path.join(params.save_dir, date_time)
        params_ext.save_file = params.save_file
        params_ext.no_cuda = False
        params_ext.filter_sizes = '1, 2, 3'
        params_ext.load_model = './snapshot_ext/' + params.model_name + '/epoch_50.pt'

        calib_data = get_calib_data(calib_set_id, calib_ftr_set_id, params_ext, dict_msg, dict_code)

        predictions, true_labels = get_predictions(data=calib_data, params=params_ext)
        # normal calibration
        quantile_a0 = calibrate(predictions, true_labels, alpha_0, calib_set_size)
        quantile_a1 = calibrate(predictions, true_labels, alpha_1, calib_set_size)
        quantile_a2 = calibrate(predictions, true_labels, alpha_2, calib_set_size)
        # class- conditional calibration
        quantile_0_a0, quantile_1_a0 = calibrate_class_conditional(predictions, true_labels, alpha_0)
        quantile_0_a1, quantile_1_a1 = calibrate_class_conditional(predictions, true_labels, alpha_1)
        quantile_0_a2, quantile_1_a2 = calibrate_class_conditional(predictions, true_labels, alpha_2)

        calib_end_time = time.time()
        elapsed_calib_time = calib_end_time - calib_start_time
        print("end of calibration step")

        # test CC2Vec + JIT
        start_test = time.time()
        test_data = pickle.load(open(test_data_path_ext, 'rb'))
        t_ids, t_labels, t_msgs, t_codes = test_data
        t_labels = np.array(t_labels)
        data_ftr_test = pickle.load(open(test_data_cc2ftr_path, 'rb'))
        params_ext.no_cuda = False
        params_ext.save_dir = os.path.join(params.save_dir, date_time)
        params_ext.save_file = params.save_file
        params_ext.filter_sizes = '1, 2, 3'
        test_pad_msg = padding_data(data=t_msgs, dictionary=dict_msg, params=params_ext, type='msg')
        test_pad_code = padding_data(data=t_codes, dictionary=dict_code, params=params_ext, type='code')
        tt_data = (data_ftr_test, test_pad_msg, test_pad_code, t_labels, dict_msg, dict_code)

        # perform validation
        t_predictions, t_true_labels = get_predictions(data=tt_data, params=params_ext)
        # normal calibration
        normal_CP(i, t_predictions, t_true_labels, quantile_a0, alpha_0)
        normal_CP(i, t_predictions, t_true_labels, quantile_a1, alpha_1)
        normal_CP(i, t_predictions, t_true_labels, quantile_a2, alpha_2)

        # class-conditional calibration
        class_cond_CP(i, t_predictions, t_true_labels, quantile_0_a0, quantile_1_a0, alpha_0)
        class_cond_CP(i, t_predictions, t_true_labels, quantile_0_a1, quantile_1_a1, alpha_1)
        class_cond_CP(i, t_predictions, t_true_labels, quantile_0_a2, quantile_1_a2, alpha_2)

        end_test = time.time()
        elapsed_test_time = end_test - start_test
        end_tot_time = time.time()
        elapsed_tot_time = end_tot_time -start_time
        file = open(params_ext.save_file, "a")
        file.write(" openstack dataset experiment nr: " + str(i) + "\n")
        file.write(" Conformal set size: " + str(calib_set_size) + "\n")
        file.write(" qhat_0: {0:f}:, qhat_1: {1:f}:, qhat_2: {2:f}: ".format(quantile_a0, quantile_a1, quantile_a2) + "\n")
        file.write(" Alpha_0: {0:f}, Alpha_1: {1:f}, Alpha_2: {2:f} c: ".format(alpha_0, alpha_1, alpha_2) + "\n")
        file.write(" CC q-hat_0: {0:f}, q-hat_1: {1:f}: for alpha_0:".format(quantile_0_a0, quantile_1_a0) + "\n")
        file.write(" CC q-hat_0: {0:f}, q-hat_1: {1:f}: for alpha_1:".format(quantile_0_a1, quantile_1_a1) + "\n")
        file.write(" CC q-hat_0: {0:f}, q-hat_1: {1:f}: for alpha_2:".format(quantile_0_a2, quantile_1_a2) + "\n")
        file.write("total time for training & testing CC2Vec with features= " + str(elapsed_tot_time)+"\n")
        file.write(" calib time " + str(elapsed_calib_time) + "\n")
        file.write(" test time " + str(elapsed_test_time) + "\n")
        file.write("------------------------------------------------------------ \n")
        file.write("end of model = " + str(i) + "\n")
        file.write("________________________________________________________ \n")
        file.close()
        print('--------------------------------------------------------------------------------')
        print('--------------------------Model training & testing done-------------------------')
        print('--------------------------------------------------------------------------------')


