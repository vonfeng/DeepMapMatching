# coding=utf-8
"""Main script to run things"""
import sys
import random
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
import json
import mlflow
from mlflow.tracking import MlflowClient

import setproctitle

setproctitle.setproctitle('mapmatching@zhaokai')


def args_parser():
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0, required=True, help='CUDA training.')
    parser.add_argument('--config', type=str, required=True, help="path to json config")

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--experiment_name', type=str, default="default")

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--seq2seq', type=str, default="attention")
    parser.add_argument('--n_layers_src', type=int, default=2)
    parser.add_argument('--n_layers_trg', type=int, default=1)
    parser.add_argument('--src_hidden_dim', type=int, default=512)
    parser.add_argument('--trg_hidden_dim', type=int, default=512)
    parser.add_argument('--dim_loc_src', type=int, default=256)
    parser.add_argument('--dim_seg_trg', type=int, default=256)

    parser.add_argument('--rnn_type', type=str, default="LSTM")
    parser.add_argument('--attn_type', type=str, default="dot")

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    config = read_config(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    config["training"]["batch_size"] = args.batch_size
    # config["training"]["lrate"] = args.lr
    # config["management"]["max_epoch_num"] = args.epoch
    # config["model"]["seq2seq"] = args.seq2seq
    # config["model"]["drop"] = args.drop
    config["model"]["n_layers_src"] = args.n_layers_src
    # config["model"]["n_layers_trg"] = args.n_layers_trg
    # config["model"]["src_hidden_dim"] = args.src_hidden_dim
    # config["model"]["trg_hidden_dim"] = args.trg_hidden_dim
    # config["model"]["dim_loc_src"] = args.dim_loc_src
    # config["model"]["dim_seg_trg"] = args.dim_seg_trg
    config["model"]["rnn_type"] = args.rnn_type
    config["model"]["attn_type"] = args.attn_type

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    mlflow.set_tracking_uri("../../experiments/")
    client = MlflowClient()
    experiment_name_mlflow = args.experiment_name
    try:
        experiment_id = client.create_experiment(name=experiment_name_mlflow)
    except:
        experiments = client.get_experiment_by_name(experiment_name_mlflow)
        experiment_id = experiments.experiment_id
    # mlflow ui --port 5002 --backend-store-uri ./experiments

    # set all the path
    experiment_name = hyperparam_string(config) + time.strftime("-%m%dT%H%M", time.localtime(time.time()))  # hyperparam_string(config)
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
            src_tim_emb_dim=[config['model']['dim_tim1_src'],
                             [config['model']['dim_tim2_1_src'], config['model']['dim_tim2_2_src']]],
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
            pad_token_src_tim2=[src['tim2id_2_1']['<pad>'], src['tim2id_2_2']['<pad>']] if config['model'][
                                                                                               'time_encoding'] == 'TwoEncoding' else None,
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
            rnn_type=config["model"]["rnn_type"],
            attn_type=config["model"]['attn_type']
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

    if config['data']['load_dir']:
        load_dir = config['data']['folder'] + config['data']['load_dir']
        model.load_state_dict(torch.load(open(load_dir)))

    # __TODO__ Make this more flexible for other learning methods.
    if config['training']['optimizer'] == 'adam':
        lr = config['training']['lrate']
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif config['training']['optimizer'] == 'adadelta':
        optimizer = optim.Adadelta(model.parameters())
    elif config['training']['optimizer'] == 'sgd':
        lr = config['training']['lrate']
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError("Learning method not recommend for task")


    def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay_epoch):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = init_lr * (0.5 ** (epoch // lr_decay_epoch))
        lr = max(lr, init_lr / 8.0)

        if epoch % lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer


    # training
    best_valid_accuracy = 0
    worse_accu_count = 0
    with mlflow.start_run(experiment_id=experiment_id):
        archive_path = mlflow.get_artifact_uri()
        json.dump(vars(args), open(os.path.join(archive_path, "args.json"), "w"), indent=2)
        mlflow.log_params(vars(args))

        for i in range(config['management']['max_epoch_num']):
            # __TODO__ Make this more flexible for other learning methods.
            if config['training']['optimizer'] == 'adam':
                optimizer = exp_lr_scheduler(optimizer, i, init_lr=lr, lr_decay_epoch=5)
            elif config['training']['optimizer'] == 'adadelta':
                optimizer = optim.Adadelta(model.parameters())
            elif config['training']['optimizer'] == 'sgd':
                lr = config['training']['lrate']
                optimizer = optim.SGD(model.parameters(), lr=lr)
            else:
                raise NotImplementedError("Learning method not recommend for task")

            losses = []
            train_idx = 0
            accuracy_train_list = []
            logging.info('============================================')
            t_s_epoch = time.time()
            t_s_train = time.time()
            flag = False
            for j in range(0, len(src['traces_loc']), batch_size):
                input_lines_src, _, _, _ = get_minibatch(src['traces_loc'], src['loc2id'], j, batch_size, src_max_len,
                                                         add_start=True, add_end=True)
                input_lines_trg, output_lines_trg, _, _ = get_minibatch(trg['traces_seg'], trg['seg2id'], j, batch_size,
                                                                        trg_max_len, add_start=True, add_end=True)
                input_lines_src_time = None
                if config['model']['time_encoding'] == 'OneEncoding':
                    input_lines_src_time, _, _, _ = get_minibatch(src['traces_tim1'], src['tim2id_1'], j, batch_size,
                                                                  src_max_len, add_start=True, add_end=True)
                elif config['model']['time_encoding'] == 'TwoEncoding':
                    input_lines_src_time_1, _, _, _ = get_minibatch(src['traces_tim2_1'], src['tim2id_2_1'], j,
                                                                    batch_size, src_max_len, add_start=True,
                                                                    add_end=True)
                    input_lines_src_time_2, _, _, _ = get_minibatch(src['traces_tim2_2'], src['tim2id_2_2'], j,
                                                                    batch_size, src_max_len, add_start=True,
                                                                    add_end=True)
                    input_lines_src_time = [input_lines_src_time_1, input_lines_src_time_2]

                decoder_logit = model(input_lines_src, input_lines_trg, input_lines_src_time)
                optimizer.zero_grad()

                loss = loss_criterion(decoder_logit.contiguous().view(-1, trg_seg_vocab_size),
                                      output_lines_trg.view(-1))
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                if j % config['management']['monitor_loss'] == 0:
                    logging.info('Epoch : %-3d Minibatch : %-7d Loss : %.5f' % (i, j, sum(losses) / len(losses)))
                    losses = []

                # 每隔monitor_accuracy个epoch，计算一次train accuracy以及输出前sample_num条train轨迹样例
                if i % config['management']['monitor_accuracy'] == 0 and train_idx < config['management'][
                    'accuracy_estimate_num']:
                    flag = True
                    word_probs = model.decode(decoder_logit).data.cpu().numpy().argmax(axis=-1)
                    output_lines_trg = output_lines_trg.data.cpu().numpy()

                    for sentence_pred, sentence_real in zip(word_probs, output_lines_trg):
                        sentence_pred = [trg['id2seg'][x] for x in sentence_pred]
                        sentence_real = [trg['id2seg'][x] for x in sentence_real]

                        index_pred = sentence_pred.index('</s>') if '</s>' in sentence_pred else len(sentence_pred)
                        index_real = sentence_real.index('</s>') if '</s>' in sentence_real else len(sentence_real)
                        sentence_pred = sentence_pred[:index_pred]
                        sentence_real = sentence_real[:index_real]

                        # train accuracy
                        accuracy_train = evaluate_accuracy(sentence_pred[:], sentence_real[:])
                        accuracy_train_list.append(accuracy_train)

                        if train_idx < config['management']['sample_num']:
                            with open(sample_dir + 'train_epoch_%d.samp' % i, 'a') as fout:
                                if config['management']['verbose']:
                                    logging.info('Train_' + str(train_idx) + '_Pred : %s ' % (' '.join(sentence_pred)))
                                    logging.info('Train_' + str(train_idx) + '_Real : %s ' % (' '.join(sentence_real)))
                                fout.write(' '.join(sentence_pred) + '\n')
                                fout.write(' '.join(sentence_real) + '\n')
                        train_idx += 1

                if flag and train_idx > config['management']['accuracy_estimate_num']:
                    t_e_train = time.time()
                    flag = False

            # 每隔monitor_accuracy个epoch，计算一次valid accuracy以及输出前sample_num条valid轨迹样例
            if i % config['management']['monitor_accuracy'] == 0:
                t_s_valid = time.time()
                accuracy_valid_list, preds, ground_truth = calc_test_accuracy(model, src, src_valid, trg, trg_valid,
                                                                              config)
                valid_accuracy_now = sum(accuracy_valid_list) / len(accuracy_valid_list)
                # save/log valid samples
                with open(sample_dir + 'valid_epoch_%d.samp' % i, 'w') as fout_valid:
                    for valid_idx, (pred, real) in enumerate(zip(preds, ground_truth)):
                        if config['management']['verbose']:
                            logging.info('Test_' + str(valid_idx) + '_Pred: ' + ' '.join(pred))
                            logging.info('Test_' + str(valid_idx) + '_Real: ' + ' '.join(real))
                        fout_valid.write(' '.join(pred) + '\n')
                        fout_valid.write(' '.join(real) + '\n')
                t_e_valid = time.time()
                # train & valid accuracy info
                logging.info('---------------------------------------')
                train_acc_epoch = sum(accuracy_train_list) / len(accuracy_train_list)
                logging.info('Trace_num: %d, Train_accuracy: %.5f' % (len(accuracy_train_list), train_acc_epoch))
                logging.info('Trace_num: %d, Test_accuracy : %.5f' % (len(accuracy_valid_list), valid_accuracy_now))
                logging.info('train evaluation cost %.2f seconds' % (t_e_train - t_s_train))
                logging.info('valid  evaluation cost %.2f seconds' % (t_e_valid - t_s_valid))

                mlflow.log_metric(key="train_acc", value=train_acc_epoch, step=i)
                mlflow.log_metric(key="valid_acc", value=valid_accuracy_now, step=i)

                # early stop
                if valid_accuracy_now < best_valid_accuracy:
                    worse_accu_count += 1
                else:
                    best_valid_accuracy = valid_accuracy_now
                    worse_accu_count = 0
                if worse_accu_count > 3 and i > 10:
                    break

            t_e_epoch = time.time()
            logging.info('Epoch : %-3d cost %.2f seconds' % (i, t_e_epoch - t_s_epoch))

        # # valid
        # accuracy_valid_list, preds, ground_truth = save_test_results(model, src, src_valid, trg, trg_valid, config)
        # with open(sample_dir + 'valid_result.samp', 'w') as fout_valid:
        #     for valid_idx, (pred, real) in enumerate(zip(preds, ground_truth)):
        #         if config['management']['verbose']:
        #             logging.info('Test_' + str(valid_idx) + '_Pred: ' + ' '.join(pred))
        #             logging.info('Test_' + str(valid_idx) + '_Real: ' + ' '.join(real))
        #         fout_valid.write(' '.join(pred) + '\n')
        #         fout_valid.write(' '.join(real) + '\n')

        # test
        accuracy_test_list, preds, ground_truth = save_test_results(model, src, src_test, trg, trg_test, config)
        with open(sample_dir + 'test_result.samp', 'w') as fout_test:
            for test_idx, (pred, real) in enumerate(zip(preds, ground_truth)):
                if config['management']['verbose']:
                    logging.info('Test_' + str(test_idx) + '_Pred: ' + ' '.join(pred))
                    logging.info('Test_' + str(test_idx) + '_Real: ' + ' '.join(real))
                fout_test.write(' '.join(pred) + '\n')
                fout_test.write(' '.join(real) + '\n')
        test_acc_now = float(sum(accuracy_test_list)) / len(accuracy_test_list)
        logging.info('Test num: %d, Test accuracy: %f' % (len(accuracy_test_list), test_acc_now))
        mlflow.log_metric(key="test_acc", value=test_acc_now)

        # 暂时只保存最终的model
        mkdir(save_dir)
        torch.save(
            model.state_dict(),
            open(os.path.join(save_dir, 'final.model'), 'wb')
        )
