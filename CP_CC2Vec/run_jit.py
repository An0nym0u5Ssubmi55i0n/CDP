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
    params.load_model = './snapshot/'+params.model_name+'/epoch_1.pt'
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

def trainExtCC2ftr(params, save_file,dict_msg, dict_code):
    params.load_model = './snapshot_ext/' + params.model_name + '/epoch_1.pt'
    params.save_file = save_file
    train_data = pickle.load(open(train_data_path_ext, 'rb'))
    train_ids, train_labels, train_msgs, train_codes = train_data
    train_labels_arr = np.array(train_labels)
    train_data_ftr = pickle.load(open(train_data_cc2ftr_path, 'rb'))
    train_pad_msg = padding_data(data=train_msgs, dictionary=dict_msg, params=params_ext, type='msg')
    train_pad_code = padding_data(data=train_codes, dictionary=dict_code, params=params_ext, type='code')
    train_ext_data = (train_data_ftr, train_pad_msg, train_pad_code, train_labels_arr, dict_msg, dict_code)
    train_model_ext(data=train_ext_data, params=params)
    end_trainCC2Vec_time = time.time()
    CC2Vec = end_trainCC2Vec_time - start_time
    file = open(params.save_file, "a")
    file.write("total time for training CC2Vec with features= " + str(CC2Vec) + "\n")
    file.close()
    print('-----------------------------------------------------------------------------------')
    print('-----------------------Finish the training of extended DeepJIT---------------------')
    print('-----------------------------------------------------------------------------------')


if __name__ == '__main__':
    start_time = time.time()
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

    i=0

    if params.train is True:
        # -----------train jit_cc2ftr-----------
        params = read_args().parse_args()
        # file = open("QT-output.txt", "x")
        params.model_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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
        print('-----------------------------------------------------------------------------------')
        print("parameters before training", params)
        print('-----------------------------------------------------------------------------------')
        trainCC2ftr(data, params)
        # ----------test jit_cc2ftr------------
        print('-----------------------------------------------------------------------------------')
        print("parameters before testing", params)
        print('-----------------------------------------------------------------------------------')
        #testCC2ftr(data, params)
        #ids, labels, msgs, codes = test_data
        # ------------train CC2Vec + JIT------------
        params_ext = read_args_extended().parse_args()
        params_ext.name = str(i) + '_epoch_1_ext.pt'
        params_ext.model_name = params.model_name
        dictionary = pickle.load(open(dictionary_data_path, 'rb'))
        dict_msg, dict_code = dictionary
        print('-----------------------------------------------------------------------------------')
        print("parameters before EXT training", params_ext)
        print('-----------------------------------------------------------------------------------')
        trainExtCC2ftr(params_ext, params.save_file, dict_msg, dict_code)
        # test CC2Vec + JIT
        test_data = pickle.load(open(test_data_path_ext, 'rb'))
        t_ids, t_labels, t_msgs, t_codes = test_data
        t_labels = np.array(t_labels)
        data_ftr_test = pickle.load(open(test_data_cc2ftr_path, 'rb'))
        params_ext.no_cuda = False
        params_ext.filter_sizes = '1, 2, 3'
        test_pad_msg = padding_data(data=t_msgs, dictionary=dict_msg, params=params_ext, type='msg')
        test_pad_code = padding_data(data=t_codes, dictionary=dict_code, params=params_ext, type='code')
        test_data = (data_ftr_test, test_pad_msg, test_pad_code, t_labels, dict_msg, dict_code)
        #test(params_ext, "2023-07-13_12-03-43", True)
        print('-----------------------------------------------------------------------------------')
        print("parameters before EXT testing", params_ext)
        print('-----------------------------------------------------------------------------------')
        evaluation_model(data=test_data, params=params_ext)
        print('-----------------------------------------------------------------------------------')
        print('-----------------------Evaluation of extended DeepJIT complete---------------------')
        print('-----------------------------------------------------------------------------------')
        end_tot_time = time.time()
        elapsed_tot_time = end_tot_time -start_time
        file = open(params_ext.save_file, "a")
        file.write("total time for training & testing CC2Vec with features= " + str(elapsed_tot_time)+"\n")
        file.write("end of model = " + str(i) + "\n")
        file.write("________________________________________________________ \n")
        file.close()


