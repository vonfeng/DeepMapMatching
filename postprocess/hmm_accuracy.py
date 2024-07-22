#coding=utf-8
import os
from utils import haversine
from xml.dom.minidom import parse
import xml.dom.minidom
import os
import pandas as pd
from utils import *
import numpy as np

"""
计算hmm结果的accuracy
先用validation结果选出最佳参数，然后计算test数据的结果
"""


def read_match_result(folder, mode):
    paras = []
    for para in os.listdir(folder):
        if para[:5] == 'sigma':
            paras.append(para)

    result = {}
    for para in paras:
        print(para)
        traces = {}
        if mode == 'segments':
            folder_traces = folder + para + '/segments/'
            filenames = os.listdir(folder_traces)
            for filename in filenames:
                with open(folder_traces + filename, 'r') as fin:
                    lines = fin.readlines()
                    traces[filename] = [line.strip() for line in lines]
            result[para] = traces
        elif mode == 'gps':
            folder_traces = folder + para + '/gps/'
            filenames = os.listdir(folder_traces)
            for filename in filenames:
                traces[filename] = read_gpx(folder_traces + filename)
            result[para] = traces
        else:
            print('error')
            break
    return result


def evaluate_accuracy(preds, reals):
    preds = drop_dup(preds)
    reals = drop_dup(reals)

    list_same_address_preds = []
    list_same_address_reals = []
    for i in set(preds):
        address_index_preds = [x for x in range(len(preds)) if preds[x] == i]
        list_same_address_preds.append([i, address_index_preds])
    for i in range(len(list_same_address_preds)):
        for j in range(len(list_same_address_preds[i][1])):
            preds[list_same_address_preds[i][1][j]] = preds[list_same_address_preds[i][1][j]] + '_' + str(j)

    for i in set(reals):
        address_index_reals = [x for x in range(len(reals)) if reals[x] == i]
        list_same_address_reals.append([i, address_index_reals])
    for i in range(len(list_same_address_reals)):
        for j in range(len(list_same_address_reals[i][1])):
            reals[list_same_address_reals[i][1][j]] = reals[list_same_address_reals[i][1][j]] + '_' + str(j)
    interset = list(set(preds).intersection(set(reals)))
    accuracy = float(len(interset))/(max(len(preds), len(reals)))
    return accuracy


def evaluate(trace_ids, traces_dict, traces_gt_dict, MAX_LEN):
    accuracy = []
    pred_len = []
    real_len = []
    for trace_id in trace_ids:
        trace_real = drop_dup(traces_gt_dict[trace_id])
        trace_pred = drop_dup(traces_dict[trace_id])
        pred_len.append(len(trace_pred))
        real_len.append(len(trace_real))
        # print(filename)
        # print(trace_pred)
        # print(trace_real)
        trace_pred = drop_dup(trace_pred)
        trace_real = drop_dup(trace_real)
        trace_real = trace_real[:MAX_LEN]
        accuracy.append(evaluate_accuracy(trace_pred, trace_real))
    return accuracy, int(sum(pred_len) / float(len(pred_len))), int(sum(real_len) / float(len(real_len)))


# CTS metrics
def read_gpx(path):
    trace = []
    DOMTree = xml.dom.minidom.parse(path)
    collection = DOMTree.documentElement
    records = collection.getElementsByTagName("trkpt")
    for record in records:
        lat = float(record.getAttribute("lat").strip().replace(' ', ''))
        lon = float(record.getAttribute("lon").strip().replace(' ', ''))
        trace.append([lon, lat])
    return trace


def calc_trace_dist(trace):
    '''
    :param trace: [[lon, lat], ...]
    :return:
    '''
    dist = 0
    for i in range(1, len(trace)):
        dist += haversine(trace[i-1][0], trace[i-1][1], trace[i][0], trace[i][1])
    return dist


def distance_accuracy(traces_dict, folder_real, algo):
    if algo == 'CTS':
        real_filenames = [item[:-12] for item in os.listdir(folder_real)]
    elif algo == 'HMM':
        real_filenames = os.listdir(folder_real)
    delta_dist_sum = 0
    real_dist_sum = 0
    for filename in traces_dict.keys():
        if filename in real_filenames:
            if algo == 'CTS':
                trace_real = read_gpx(folder_real + filename + '.matched.gpx')
            elif algo == 'HMM':
                trace_real = read_gpx(folder_real + filename)
            trace_pred = traces_dict[filename]
            # print(filename)
            # print(trace_pred)
            # print(trace_real)
            pred_dist = calc_trace_dist(trace_pred)
            real_dist = calc_trace_dist(trace_real)
            delta_dist_sum += abs(pred_dist - real_dist)
            real_dist_sum += real_dist
    return 1 - delta_dist_sum / real_dist_sum


def nors_accuracy(trace_ids, traces_dict, gt_result_seg):  # number of road segment
    delta_segnum_sum = 0
    real_segnum_sum = 0
    for trace_id in trace_ids:
        trace_real = gt_result_seg[trace_id]
        trace_pred = traces_dict[trace_id]
        delta_segnum_sum += abs(len(trace_pred) - len(trace_real))
        real_segnum_sum += len(trace_real)
    return 1 - float(delta_segnum_sum) / real_segnum_sum


def tune_para(folder_hmm_result, path_real_seg, time_gap, noise, filenames, mode):
    para_num = len(filenames)
    HMM_result = pd.DataFrame(np.zeros((para_num, 4)), columns=['sigma', 'beta', 'as', 'accu'])
    para_list = []

    # gt
    gt_result_seg = read_seg_trace_data(path_real_seg)

    # HMM
    hmm_result_seg = {}
    MAX_LEN = 0
    max_len_dict = {
        (60, 0): (40, 54),
        (30, 100): (71, 54),
        (40, 100): (57, 54),
        (80, 100): (31, 54),
        (100, 100): (25, 55),
        (120, 100): (21, 56),
        (200, 100): (14, 60),
    }
    for para in filenames:
        if para[:5] == 'sigma':
            if time_gap == 60:
                MAX_LEN = max_len_dict[(60, 0)][1]
            else:
                MAX_LEN = max_len_dict[(time_gap, noise)][1]
            # MAX_LEN = int(para.split('_')[-1])
            hmm_result_seg[para] = read_seg_trace_data(
                folder_hmm_result + 'tune_para/' + para + '/segments/seg_matched_noise_real_%s.trace' % mode)

    for idx, para in enumerate(hmm_result_seg.keys()):
        para_list.append(para)
        HMM_result.iloc[idx]['sigma'] = float(para.split('_')[1])
        HMM_result.iloc[idx]['beta'] = float(para.split('_')[-1])

    for idx, para in enumerate(hmm_result_seg.keys()):
        trace_ids = list(set(hmm_result_seg[para].keys()).intersection(set(gt_result_seg.keys())))
        accu, pred_len, real_len = evaluate(trace_ids, hmm_result_seg[para], gt_result_seg, MAX_LEN=MAX_LEN)
        num_accu = nors_accuracy(trace_ids, hmm_result_seg[para], gt_result_seg)
        HMM_result.iloc[idx]['accu'] = sum(accu) / len(accu)
        HMM_result.iloc[idx]['as'] = num_accu
        print('HMM: ' + para + ', trace num: ' + str(len(accu)) + ', pred_len: ' + str(pred_len) + ', real_len: ' + str(
            real_len) + ', average accuracy: ' + str(sum(accu) / len(accu)) + ', num accuracy: ' + str(num_accu))

    # hmm_result_gps = read_match_result(folder_hmm_result + 'tune_para/', 'gps')
    # for idx, para in enumerate(hmm_result_gps.keys()):
    #     dist_accu = distance_accuracy(hmm_result_gps[para], folder_real_gps, algo='HMM')
    #     HMM_result.iloc[idx]['ad'] = dist_accu
    #     print('HMM: ' + para + ', dist accuracy: ' + str(dist_accu))

    HMM_result = pd.concat([pd.DataFrame(para_list, columns=['para']), HMM_result], axis=1)
    return HMM_result


def hmm_valid(tt, ss):
    if tt == 'noise':
        total_result = []
        folder = 'D:/DeepMapMatching/data/tencent/'
        hmm_out_file = folder + 'HMM_result/%s/valid/%s.csv' % (ss, tt)
        for time_gap in [60]:  # 30, 40, 60, 80, 100, 120
            for noise in [10, 20, 40, 60, 80, 100, 120]:  # 0, 10, 20, 40, 60, 80, 100, 120
                print(time_gap, noise)
                folder_hmm_result = folder + 'HMM_result/%s/valid/timegap-%d_noise-gaussian_sigma-%d_dup-20/dup-10_sl-100/' % (ss, time_gap, noise)
                path_real_seg = folder + 'HMM_result/step3_real_split/run_all/sigma_60_beta_0.01/segments/seg_matched_real_valid.trace'
                HMM_result = tune_para(folder_hmm_result, path_real_seg, time_gap, noise, os.listdir(folder_hmm_result + 'tune_para/'), 'valid')
                total_result.append([time_gap, noise, HMM_result.loc[:, 'accu'].max(), HMM_result.iloc[HMM_result.loc[:, 'accu'].idxmax()]['para']])
        total_result = pd.DataFrame(total_result, columns=['time_gap', 'noise', 'accu', 'para'])
        total_result.sort_values(by=[tt]).to_csv(hmm_out_file, index=False)
    elif tt == 'time_gap':
        total_result = []
        folder = 'D:/DeepMapMatching/data/tencent/'
        hmm_out_file = folder + 'HMM_result/%s/valid/%s.csv' % (ss, tt)
        for time_gap in [30, 40, 60, 80, 100, 120]:
            for noise in [100]:
                print(time_gap, noise)
                folder_hmm_result = folder + 'HMM_result/%s/valid/timegap-%d_noise-gaussian_sigma-%d_dup-20/dup-10_sl-100/' % (ss, time_gap, noise)
                path_real_seg = folder + 'HMM_result/step3_real_split/run_all/sigma_60_beta_0.01/segments/seg_matched_real_valid.trace'
                HMM_result = tune_para(folder_hmm_result, path_real_seg, time_gap, noise, os.listdir(folder_hmm_result + 'tune_para/'), 'valid')
                total_result.append([time_gap, noise, HMM_result.loc[:, 'accu'].max(), HMM_result.iloc[HMM_result.loc[:, 'accu'].idxmax()]['para']])
        total_result = pd.DataFrame(total_result, columns=['time_gap', 'noise', 'accu', 'para'])
        total_result.sort_values(by=[tt]).to_csv(hmm_out_file, index=False)
    else:
        return


def hmm_test(tt, ss):
    if tt == 'noise':
        total_result = []
        folder = 'D:/DeepMapMatching/data/tencent/'
        hmm_out_file = folder + 'HMM_result/%s/test/%s.csv' % (ss, tt)
        best_paras = pd.read_csv(folder + 'HMM_result/%s/valid/noise.csv' % ss)
        para_dict = dict(zip(best_paras['noise'], best_paras['para']))
        for time_gap in [60]:  # 30, 40, 60, 80, 100, 120
            for noise in [10, 20, 40, 60, 80, 100, 120]:  # 0, 10, 20, 40, 60, 80, 100, 120
                print(time_gap, noise)
                folder_hmm_result = folder + 'HMM_result/%s/test/timegap-%d_noise-gaussian_sigma-%d_dup-20/dup-10_sl-100/' % (ss, time_gap, noise)
                path_real_seg = folder + 'HMM_result/step3_real_split/run_all/sigma_60_beta_0.01/segments/seg_matched_real_test.trace'
                HMM_result = tune_para(folder_hmm_result, path_real_seg, time_gap, noise, [para_dict[noise]], 'test')
                total_result.append([time_gap, noise, HMM_result.loc[:, 'accu'].max(), HMM_result.iloc[HMM_result.loc[:, 'accu'].idxmax()]['para']])
        total_result = pd.DataFrame(total_result, columns=['time_gap', 'noise', 'accu', 'para'])
        total_result.sort_values(by=[tt]).to_csv(hmm_out_file, index=False)
    elif tt == 'time_gap':
        total_result = []
        folder = 'D:/DeepMapMatching/data/tencent/'
        hmm_out_file = folder + 'HMM_result/%s/test/%s.csv' % (ss, tt)
        best_paras = pd.read_csv(folder + 'HMM_result/%s/valid/time_gap.csv' % ss)
        para_dict = dict(zip(best_paras['time_gap'], best_paras['para']))
        for time_gap in [30, 40, 60, 80, 100, 120]:
            for noise in [100]:
                print(time_gap, noise)
                folder_hmm_result = folder + 'HMM_result/%s/test/timegap-%d_noise-gaussian_sigma-%d_dup-20/dup-10_sl-100/' % (ss, time_gap, noise)
                path_real_seg = folder + 'HMM_result/step3_real_split/run_all/sigma_60_beta_0.01/segments/seg_matched_real_test.trace'
                HMM_result = tune_para(folder_hmm_result, path_real_seg, time_gap, noise, [para_dict[time_gap]], 'test')
                total_result.append([time_gap, noise, HMM_result.loc[:, 'accu'].max(), HMM_result.iloc[HMM_result.loc[:, 'accu'].idxmax()]['para']])
        total_result = pd.DataFrame(total_result, columns=['time_gap', 'noise', 'accu', 'para'])
        total_result.sort_values(by=[tt]).to_csv(hmm_out_file, index=False)
    else:
        return


if __name__ == '__main__':
    hmm_valid('noise', 'test_data')
    hmm_test('time_gap', 'test_data')

