#coding=utf-8
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
fig_path = 'D:/DeepMapMatching/figs/'
lw = 5
ms = 14
tick_size = 40
label_size = 44


def plot_line(data, gap, xl, yl, xtics):
    plt.figure(figsize=(16, 9))
    plt.plot(xtics, data, '-o')
    plt.xlabel(xl, fontsize=24)
    if gap > 0:
        plt.xticks(range(len(data)), [str(i * gap) for i in range(len(data))], fontsize=20)
    else:
        plt.xticks(xtics, [str(item) for item in xtics], fontsize=20)
        plt.xscale('log')
    plt.yticks(fontsize=20)
    plt.ylabel(yl, fontsize=24)
    plt.show()


def plot_lines(data_dict, eval_gap, x_gap, xl, yl, title):
    '''
    :param data: dict
    :param gap:
    :param xl:
    :param yl:
    :return:
    '''
    plt.figure(figsize=(16, 9))
    max_len = 0
    les = []
    t = int(x_gap/eval_gap)
    for key in data_dict.keys():
        data = data_dict[key]
        plt.plot(data)
        max_len = max(max_len, len(data[::t]))
        les.append(key)
    plt.xlabel(xl, fontsize=24)
    plt.xticks([i * t for i in range(max_len)], [str(i * x_gap) for i in range(max_len)], fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title)
    plt.ylabel(yl, fontsize=24)
    plt.legend(les, fontsize=16)
    plt.show()


def tune_para(folder, para):
    log_folder = folder + para + '/'
    N = 3
    accuracys = {}
    result = []
    col_names = []
    for file in os.listdir(log_folder):
        with open(log_folder + file, 'r') as fin:
            lines = fin.readlines()
            # plot valid accuracy
            valid_acc = []
            paras = {}
            for line in lines:
                if 'Test_accuracy' in line:
                    valid_acc.append(float(line.strip().split(': ')[-1]))
                if 'Learning Rate : ' in line:
                    paras['lr'] = line.strip().split('Learning Rate : ')[-1]
                # if 'Model' in line:
                #     paras['model'] = line.strip().split('Model : ')[-1]
                if 'Encoder: Source RNN Depth' in line:
                    paras['src_layer'] = line.strip().split('Encoder: Source RNN Depth : ')[-1]
                if 'Decoder: Target RNN Depth' in line:
                    paras['trg_layer'] = line.strip().split('Decoder: Target RNN Depth : ')[-1]
                if 'Max source trace length : ' in line:
                    paras['max_src_len'] = line.strip().split('Max source trace length : ')[-1]
                if 'Max target trace length : ' in line:
                    paras['max_trg_len'] = line.strip().split('Max target trace length : ')[-1]
                if 'Encoder: Source RNN Hidden Dim' in line:
                    paras['hidden_dim'] = line.strip().split('Encoder: Source RNN Hidden Dim  : ')[-1]
                if 'Source Loc Embedding Dim ' in line:
                    paras['src_loc_emb'] = line.strip().split('Source Loc Embedding Dim  : ')[-1]
                if 'Source Tim OneEncoding Embedding Dim : ' in line:
                    paras['src_tim1_emb'] = line.strip().split('Source Tim OneEncoding Embedding Dim : ')[-1]
                if 'Source Tim TwoEncoding Embedding Dim : ' in line and 'src_tim21_emb' not in paras:
                    paras['src_tim21_emb'] = line.strip().split('Source Tim TwoEncoding Embedding Dim : ')[-1]
                if 'Source Tim TwoEncoding Embedding Dim : ' in line and 'src_tim21_emb' in paras:
                    paras['src_tim22_emb'] = line.strip().split('Source Tim TwoEncoding Embedding Dim : ')[-1]
                if 'Target Seg Embedding Dim' in line:
                    paras['trg_seg_emb'] = line.strip().split('Target Seg Embedding Dim : ')[-1]
                if 'Time Encoding' in line:
                    paras['time_encoding'] = line.strip().split('Time Encoding : ')[-1]
                # if 'Batch Size : ' in line:
                #     paras['batchsize'] = line.strip().split('Batch Size : ')[-1]
                # if 'Max source trace length' in line:
                #     paras['max_src_len'] = line.strip().split('Max source trace length : ')[-1]
                # if 'Max target trace length' in line:
                #     paras['max_trg_len'] = line.strip().split('Max target trace length : ')[-1]
                # if 'Data folder : ' in line:
                #     # paras['dataset'] = line.strip().split('Data folder : ')[-1].split('/')[5].replace('-', '_') + '_' + line.strip().split('Data folder : ')[-1].split('/')[6].replace('-', '_') #.split('-')[-1]
                #     paras['dataset'] = line.strip().split('Data folder : ')[-1].split('/')[1].replace('-', '_') + '_' + \
                #                        line.strip().split('Data folder : ')[-1].split('/')[2].replace('-', '_')  # .split('-')[-1]

                if 'Test accuracy: ' in line:
                    paras['test_accu'] = line.strip().split('Test accuracy: ')[-1]

            if len(valid_acc) > N and abs(valid_acc[-1] - valid_acc[-N]) < 0.1:
                paras['valid_accu'] = sum(valid_acc[-N:]) / N
            result.append([paras[key] for key in sorted(paras.keys())])
            col_names = sorted(paras.keys())

            # key = ''
            # for name in sorted(paras.keys()):
            #     key += paras[name] + '-'
            # accuracys[key[:-1]] = valid_acc
            # title = '-'.join(sorted(paras.keys()))
    # save result

    result = pd.DataFrame(result, columns=col_names)
    result = result.sort_values(by=['test_accu'])

    # for key in accuracys:
    #     if len(accuracys[key]) > N and abs(accuracys[key][-1] - accuracys[key][-N]) < 0.1:
    #         ave_accu[key] = sum(accuracys[key][-N:]) / N
    # result = pd.DataFrame([key.split('-') + [ave_accu[key]] for key in ave_accu], columns=title.split('-') + ['valid_accu'])
    # # result = result[['valid_accu', 'test_accu', 'lr', 'dataset', 'trg_layer', 'batchsize', 'max_src_len', 'max_trg_len', 'src_loc_emb']]
    # # result['dataset'] = result['dataset'].map(lambda x: int(x.split('_')[1]))
    # result = result.sort_values(by=[ 'test_accu', 'valid_accu'])
    result.to_csv(folder + para + '.csv', index=False)
    print(result)


def get_result(mode, folder):
    N = 3
    log_folder = folder + mode + '/'
    result = []
    for file in os.listdir(log_folder):
        print(file)
        with open(log_folder + file, 'r') as fin:
            lines = fin.readlines()
            # plot valid accuracy
            valid_acc = []
            paras = {}
            for line in lines:
                if 'Test_accuracy' in line:
                    valid_acc.append(float(line.strip().split(': ')[-1]))
                if 'Data folder : ' in line:
                    if mode == 'noise' or mode == 'time_gap':
                        paras['dataset'] = line.strip().split('Data folder : ')[-1].split('/')[1].replace('-', '_') + '_' + \
                                           line.strip().split('Data folder : ')[-1].split('/')[2].replace('-', '_')  # .split('-')[-1]
                    elif mode == 'dup_num':
                        paras['dataset'] = line.strip().split('Data folder : ')[-1].split('/')[2]
                    elif mode == 'gen_num':
                        paras['dataset'] = line.strip().split('Data folder : ')[-1].split('/')[-3]
                    else:
                        print('wrong mode')
                        return
                if 'Test accuracy: ' in line:
                    paras['test_accu'] = line.strip().split('Test accuracy: ')[-1]

            if len(valid_acc) > N and abs(valid_acc[-1] - valid_acc[-N]) < 0.1:
                paras['valid_accu'] = sum(valid_acc[-N:]) / N

            result.append([paras['test_accu'], paras['valid_accu'], paras['dataset']])
            # key = ''
            # for name in sorted(paras.keys()):
            #     key += paras[name] + '-'
            # valid_accu[key[:-1]] = valid_acc
            # title = '-'.join(sorted(paras.keys()))
    # save result
    # ave_accu = {}

    result = pd.DataFrame(result, columns=['test_accu', 'valid_accu', 'dataset'])
    result = result.sort_values(by=['test_accu', 'valid_accu', 'dataset'])
    result.to_csv(folder + mode + '.csv', index=False)
    print(result)


def get_curve(log_folder):
    result = []
    names = ['train_accu_real_gen', 'train_accu_real']
    for idx, file in enumerate(os.listdir(log_folder)):
        print(file)
        train_acc = []
        valid_acc = []
        with open(log_folder + file, 'r') as fin:
            lines = fin.readlines()
            # plot valid accuracy
            paras = {}
            for line in lines:
                if 'Train_accuracy' in line:
                    train_acc.append(float(line.strip().split(': ')[-1]))
                if 'Test_accuracy' in line:
                    valid_acc.append(float(line.strip().split(': ')[-1]))
                if 'Data folder : ' in line:
                    paras['dataset'] = line.strip().split('Data folder : ')[-1].split('/')[-3]
                if 'Test accuracy: ' in line:
                    paras['test_accu'] = line.strip().split('Test accuracy: ')[-1]

            train_acc = [0] + train_acc[1:]
            valid_acc = [0] + valid_acc[:-1]
            result.append([paras['test_accu'], paras['dataset'], train_acc, valid_acc])

            plt.figure(figsize=[11, 9])
            plt.plot(range(len(train_acc)), train_acc, '-', lw=lw, ms=ms)
            plt.plot(range(len(valid_acc)), valid_acc, '--', lw=lw)
            plt.plot(range(len(valid_acc)), [0.55]*len(valid_acc), '-.', lw=lw)

            plt.yticks(fontsize=tick_size)
            plt.xlabel('Epoch', fontsize=label_size)
            plt.ylabel('Accuracy', fontsize=label_size)
            plt.legend(['train', 'validation', 'HMM'], fontsize=tick_size, loc='best')
            plt.ylim([0, 1])
            if paras['dataset'] == 'trace_700000':
                plt.xticks([0, 3, 6, 9, 12, 15, 18], fontsize=tick_size)
                plt.arrow(19.8, 0.662, 0, 0.886 - 0.674, color='red', head_length=0.03, head_width=0.25, length_includes_head=True, linewidth=4)
                plt.arrow(19.8, 0.8, 0, 0.66 - 0.8, color='red', head_length=0.03, head_width=0.25, length_includes_head=True, linewidth=4)
                plt.text(17.5, 0.75, '0.23', fontsize=28)
            elif paras['dataset'] == 'trace_0':
                plt.xticks([0, 3, 6, 9, 12], fontsize=tick_size)
                plt.arrow(10.8, 0.584, 0, 0.982 - 0.594, color='red', head_length=0.03, head_width=0.2, length_includes_head=True, linewidth=4)
                plt.arrow(10.8, 0.9, 0, 0.584 - 0.9, color='red', head_length=0.03, head_width=0.2, length_includes_head=True, linewidth=4)
                plt.text(9.5, 0.75, '0.41', fontsize=28)
            plt.tight_layout()
            plt.savefig(fig_path + names[idx] + '.png')
            plt.show()

    # result = pd.DataFrame(result, columns=['test_accu', 'valid_accu', 'dataset'])
    # result = result.sort_values(by=['test_accu', 'valid_accu', 'dataset'])
    # result.to_csv(folder + mode + '.csv')
    print(result)


if __name__ == '__main__':
    get_curve(log_folder='D:/DeepMapMatching/data/tencent/seq2seq/combine/timegap-60_noise-gaussian_sigma-100/loss/')
    # get_result(mode='dup_num', folder='D:/DeepMapMatching/data/tencent/seq2seq/combine/timegap-60_noise-gaussian_sigma-100/attention/')
    # get_result(mode='time_gap', folder='D:/DeepMapMatching/data/tencent/seq2seq/combine_fixlen/')
    # get_result(mode='noise', folder='D:/DeepMapMatching/data/tencent/seq2seq/combine/attention/')
    # tune_para(folder='D:/DeepMapMatching/data/tencent/seq2seq/combine/timegap-60_noise-gaussian_sigma-100/dup-10_sl-100/trace_700000/', para='attention_hidden')
    # get_result(mode='gen_num', folder='D:/DeepMapMatching/data/tencent/seq2seq/combine/timegap-60_noise-gaussian_sigma-100/attention/')
    # get_result(mode='gen_num', folder='D:/DeepMapMatching/data/tencent/seq2seq/combine_shortest/')
