#coding=utf-8
"""Main script to run things"""
import sys
import os
from data_utils import read_nmt_data, get_minibatch, read_config, hyperparam_string, mkdir
from model import Seq2Seq, Seq2SeqAttention, Seq2SeqFastAttention
from evaluate import evaluate_model, evaluate_accuracy, calc_test_accuracy, save_test_results
import numpy as np
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# import setproctitle
# setproctitle.setproctitle('mapmatching@zhaokai')

if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        help="gpu id",
        required=True
    )
    parser.add_argument(
        "--config",
        help="path to json config",
        required=True
    )
    args = parser.parse_args()
    config = read_config(args.config)

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set all the path
    experiment_name = 'test' + time.strftime("-%m%dT%H%M", time.localtime(time.time()))  # hyperparam_string(config)
    save_dir = config['data']['folder'] + config['data']['save_dir'] + experiment_name + '/'
    sample_dir = config['data']['folder'] + config['data']['sample_folder'] + experiment_name + '/'
    mkdir(sample_dir)  # 注意：文件夹名字不能过长
    mkdir(config['data']['folder'] + config['data']['log_folder'])

    # set logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=config['data']['folder'] + config['data']['log_folder'] + experiment_name + '_lr-decay.log',
        filemode='w'
    )
    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    # optional, set the logging level
    console.setLevel(logging.INFO)
    # set a format which is the same for console use
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # read data from files
    t_s = time.time()
    src, trg, src_loc_vocab_size, src_tim_vocab_size, trg_seg_vocab_size = read_nmt_data(dataset='train', config=config)
    src_valid, trg_valid, _, _, _ = read_nmt_data(dataset='valid', config=config)
    src_test, trg_test, _, _, _ = read_nmt_data(dataset='test', config=config)
    t_e = time.time()
    logging.info('Reading data cost %.2f seconds' % (t_e - t_s))

    batch_size = config['training']['batch_size']
    src_max_len = config['model']['max_src_length']
    trg_max_len = config['model']['max_trg_length']

    logging.info('------------------ model ------------------')
    logging.info('Model : %s ' % (config['model']['seq2seq']))
    logging.info('Source RNN Bidirectional  : %s' % (config['model']['bidirectional']))
    logging.info('Drop out : %f ' % config['model']['dropout'])
    logging.info('Encoder: Source RNN Depth : %d ' % (config['model']['n_layers_src']))
    logging.info('Encoder: Source RNN Hidden Dim  : %s' % (config['model']['src_hidden_dim']))
    logging.info('Decoder: Target RNN Depth : %d ' % (config['model']['n_layers_trg']))
    logging.info('Decoder: Target RNN Hidden Dim  : %s' % (config['model']['trg_hidden_dim']))
    logging.info('Source Loc Embedding Dim  : %s' % (config['model']['dim_loc_src']))
    logging.info('Source Tim OneEncoding Embedding Dim : %s' % (config['model']['dim_tim1_src']))
    logging.info('Source Tim TwoEncoding Embedding Dim : %s' % (config['model']['dim_tim2_1_src']))
    logging.info('Source Tim TwoEncoding Embedding Dim : %s' % (config['model']['dim_tim2_2_src']))
    logging.info('Target Seg Embedding Dim : %s' % (config['model']['dim_seg_trg']))
    logging.info('Time Encoding : %s' % (config['model']['time_encoding']))
    logging.info('Batch Size : %d' % batch_size)
    logging.info('Max source trace length : %d' % config['model']['max_src_length'])
    logging.info('Max target trace length : %d' % config['model']['max_trg_length'])
    logging.info('Loc num in src: %d' % src_loc_vocab_size)
    logging.info('Tim num in src by OneEncoding : %d' % src_tim_vocab_size[0])
    logging.info('Tim num in src by TwoEncoding : %d' % src_tim_vocab_size[1][0])
    logging.info('Tim num in src by TwoEncoding : %d' % src_tim_vocab_size[1][1])
    logging.info('Seg num in trg : %d : ' % trg_seg_vocab_size)
    logging.info('------------------ data ------------------')
    logging.info('Data folder : %s' % config['data']['folder'])
    logging.info('------------------ management ------------------')
    logging.info('Accuracy estimate traces num : %d' % config['management']['accuracy_estimate_num'])
    logging.info('------------------ training ------------------')
    logging.info('Optimizer : %s ' % (config['training']['optimizer']))
    logging.info('Learning Rate : %f ' % (config['training']['lrate']))

    weight_mask = torch.ones(trg_seg_vocab_size).cuda()
    weight_mask[trg['seg2id']['<pad>']] = 0
    loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()

    if config['model']['seq2seq'] == 'vanilla':
        model = Seq2Seq(
            src_loc_emb_dim=config['model']['dim_loc_src'],
            src_tim_emb_dim=[config['model']['dim_tim1_src'], [config['model']['dim_tim2_1_src'], config['model']['dim_tim2_2_src']]],
            trg_seg_emb_dim=config['model']['dim_seg_trg'],
            src_loc_vocab_size=src_loc_vocab_size,
            src_tim_vocab_size=src_tim_vocab_size,
            trg_seg_vocab_size=trg_seg_vocab_size,
            src_hidden_dim=config['model']['src_hidden_dim'],
            trg_hidden_dim=config['model']['trg_hidden_dim'],
            batch_size=batch_size,
            bidirectional=config['model']['bidirectional'],
            pad_token_src_loc=src['loc2id']['<pad>'],
            pad_token_src_tim1=src['tim2id_1']['<pad>'] if config['model']['time_encoding'] == 'OneEncoding' else None,
            pad_token_src_tim2=[src['tim2id_2_1']['<pad>'], src['tim2id_2_2']['<pad>']] if config['model']['time_encoding'] == 'TwoEncoding' else None,
            pad_token_trg=trg['seg2id']['<pad>'],
            nlayers_src=config['model']['n_layers_src'],
            nlayers_trg=config['model']['n_layers_trg'],
            dropout=config['model']['dropout'],
            time_encoding=config['model']['time_encoding'],
        ).cuda()
    elif config['model']['seq2seq'] == 'attention':
        model = Seq2SeqAttention(
            src_loc_emb_dim=config['model']['dim_loc_src'],
            src_tim_emb_dim=[config['model']['dim_tim1_src'], [config['model']['dim_tim2_1_src'], config['model']['dim_tim2_2_src']]],
            trg_seg_emb_dim=config['model']['dim_seg_trg'],
            src_loc_vocab_size=src_loc_vocab_size,
            src_tim_vocab_size=src_tim_vocab_size,
            trg_seg_vocab_size=trg_seg_vocab_size,
            src_hidden_dim=config['model']['src_hidden_dim'],
            trg_hidden_dim=config['model']['trg_hidden_dim'],
            # ctx_hidden_dim=0,
            # attention_mode='dot',
            batch_size=batch_size,
            bidirectional=config['model']['bidirectional'],
            pad_token_src_loc=src['loc2id']['<pad>'],
            pad_token_src_tim1=src['tim2id_1']['<pad>'] if config['model']['time_encoding'] == 'OneEncoding' else None,
            pad_token_src_tim2=[src['tim2id_2_1']['<pad>'], src['tim2id_2_2']['<pad>']] if config['model']['time_encoding'] == 'TwoEncoding' else None,
            pad_token_trg=trg['seg2id']['<pad>'],
            nlayers_src=config['model']['n_layers_src'],
            dropout=config['model']['dropout'],
            time_encoding=config['model']['time_encoding'],
        ).cuda()
    elif config['model']['seq2seq'] == 'fastattention':
        model = Seq2SeqFastAttention(
            src_emb_dim=config['model']['dim_loc_src'],
            trg_emb_dim=config['model']['dim_seg_trg'],
            src_vocab_size=src_loc_vocab_size,
            trg_vocab_size=trg_seg_vocab_size,
            src_hidden_dim=config['model']['src_hidden_dim'],
            trg_hidden_dim=config['model']['trg_hidden_dim'],
            batch_size=config['training']['batch_size'],
            bidirectional=config['model']['bidirectional'],
            pad_token_src=src['loc2id']['<pad>'],
            pad_token_trg=trg['seg2id']['<pad>'],
            nlayers=config['model']['n_layers_src'],
            nlayers_trg=config['model']['n_layers_trg'],
            dropout=config['model']['dropout'],
        ).cuda()

    # load_dir = config['data']['folder'] + config['data']['load_dir']
    model.load_state_dict(torch.load(open('D:/DeepMapMatching/data/tencent/seq2seq/combine/attention/model/attention-2_1-512_512-256_256-40_54-NoEncoding_16_0_0-0.50-adam_0.0010_128-0607T1451/final.model', 'rb')))

    for N in [1, 2, 3, 4]:
        with open('D:/DeepMapMatching/data/tencent/seq2seq/combine/attention/data/n_%d.pkl' % N, 'rb') as fin:
            data = pickle.load(fin)
        all_dist = []
        count = 0
        for seg in data:
            seg_dist = []
            for i in range(len(seg[0])):
                try:
                    embedding1 = np.array(model.src_embedding.weight[src['loc2id'][str(seg[0][i])]].tolist())
                    embedding2 = np.array(model.src_embedding.weight[src['loc2id'][str(seg[1][i])]].tolist())
                    # seg_dist.append(cosine_similarity([embedding1, embedding2])[0][1])
                    seg_dist.append(np.linalg.norm(embedding1 - embedding2))
                except:
                    count += 1
            all_dist.append(sum(seg_dist) / len(seg_dist))
        print(sum(all_dist) / len(all_dist))
