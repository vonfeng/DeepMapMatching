#coding=utf-8
from utils import *
import pandas as pd
"""
计算spatial skewing
"""


def calc_trace_dist(trace):
    '''
    :param trace: [[lon, lat], ...]
    :return:
    '''
    dist = 0
    for i in range(1, len(trace)):
        dist += haversine(trace[i-1][0], trace[i-1][1], trace[i][0], trace[i][1])
    return dist


def calc_trace2trace_dist(trace1, trace2):
    '''
    :param trace: [[lon, lat], ...]
    :return:
    '''
    dist = 0
    for rec1 in trace1:
        p2p_dist = []
        for rec2 in trace2:
            p2p_dist.append(haversine(rec1[0], rec1[1], rec2[0], rec2[1]))
        dist += min(p2p_dist)
    return dist


def distance_accuracy(folder, data, filename):
    delta_dist_sum = 0
    gt_dist_sum = 0
    trace_gt = read_gpx(folder + 'gt_gps/gt_' + filename, timestamp=False)
    trace_pred = read_gpx(folder + data + '/' + data + '_' + filename, timestamp=False)
    gt_dist = calc_trace_dist(trace_gt)
    pred_dist = calc_trace_dist(trace_pred)
    delta_dist_sum += abs(pred_dist - gt_dist)
    gt_dist_sum += gt_dist
    return 1 - delta_dist_sum / gt_dist_sum


def main():
    result = []
    mode = 'time_gap'
    accu_type = 'trace2trace_dist'
    if mode == 'time_gap':
        t_list = [30, 40, 60, 80, 100, 120]
        n_list = [100]
    else:
        t_list = [60]
        n_list = [10, 20, 40, 60, 80, 100, 120]  #  [10, 20, 40, 60, 80, 100, 120]
    for time_gap in t_list:  # [30, 40, 60, 80, 100, 120]
        for noise in n_list:  # [10, 20, 40, 60, 80, 120]
            print(time_gap, noise)
            folder = 'D:/DeepMapMatching/data/tencent/gpx/result/%s/timegap-%d_noise-gaussian_sigma-%d_dup-20/' % (mode, time_gap, noise)
            gt_filenames = [item.split('gt_')[-1] for item in os.listdir(folder + 'gt/')]
            hmm_filenames = [item.split('hmm_')[-1] for item in os.listdir(folder + 'hmm/')]
            if time_gap == 60 and noise == 100:
                cts_filenames = [item.split('noise_')[-1] for item in os.listdir(folder + 'cts/')]
            else:
                cts_filenames = [item.split('cts_')[-1] for item in os.listdir(folder + 'cts/')]
            seq2seq_filenames_0 = [item.split('seq2seq_')[-1] for item in os.listdir(folder + 'seq2seq_0/')]
            seq2seq_filenames_1 = [item.split('seq2seq_')[-1] for item in os.listdir(folder + 'seq2seq_1/')]
            seq2seq_filenames_2 = [item.split('seq2seq_')[-1] for item in os.listdir(folder + 'seq2seq_2/')]

            filenames = set(gt_filenames).intersection(set(hmm_filenames)).intersection(set(seq2seq_filenames_0)).intersection(set(seq2seq_filenames_1)).intersection(set(seq2seq_filenames_2))

            hmm_accu = []
            cts_accu = []
            s2s_accu = [[], [], []]
            max_len_dict = {
                (60, 0): (40, 54),
                (30, 100): (71, 54),
                (40, 100): (57, 54),
                (80, 100): (31, 54),
                (100, 100): (25, 55),
                (120, 100): (21, 56),
                (200, 100): (14, 60),
            }
            if time_gap == 60:
                max_len = max_len_dict[(60, 0)]  # (1000, 1000)
            else:
                max_len = max_len_dict[(time_gap, noise)]
            for file_idx, filename in enumerate(filenames):
                if file_idx % 100 == 0:
                    print(file_idx)
                # read data
                trace_gt = read_gpx(folder + 'gt/gt_' + filename, timestamp=False)[:max_len[1]]
                trace_hmm = read_gpx(folder + 'hmm/hmm_' + filename, timestamp=False)
                hmm_accu.append(calc_trace2trace_dist(trace_gt, trace_hmm) / len(trace_gt))

                for i in range(3):
                    trace_seq2seq = read_gpx(folder + 'seq2seq_%d/seq2seq_' % i + filename, timestamp=False)
                    s2s_accu[i].append(calc_trace2trace_dist(trace_gt, trace_seq2seq) / len(trace_gt))

            for filename in cts_filenames:
                trace_gt = read_gpx(folder + 'gt/gt_' + filename, timestamp=False)[:max_len[1]]
                if time_gap == 60 and noise == 100:
                    trace_cts = read_gpx(folder + 'cts/noise_' + filename, timestamp=False)
                else:
                    trace_cts = read_gpx(folder + 'cts/cts_' + filename, timestamp=False)
                cts_accu.append(calc_trace2trace_dist(trace_gt, trace_cts) / len(trace_gt))

                # print(filename, hmm_accu[-1], s2s_accu[-1])
            print([time_gap, noise, sum(hmm_accu) / len(hmm_accu), sum(cts_accu) / len(cts_accu), sum(s2s_accu[0]) / len(s2s_accu[0]), sum(s2s_accu[1]) / len(s2s_accu[1]), sum(s2s_accu[2]) / len(s2s_accu[2])])
            result.append([time_gap, noise, sum(hmm_accu) / len(hmm_accu), sum(cts_accu) / len(cts_accu), sum(s2s_accu[0]) / len(s2s_accu[0]), sum(s2s_accu[1]) / len(s2s_accu[1]), sum(s2s_accu[2]) / len(s2s_accu[2])])
    pd.DataFrame(result, columns=['time_gap', 'noise', 'hmm', 'cts', 'seq2seq_0', 'seq2seq_1', 'seq2seq_2']).to_csv('D:/DeepMapMatching/data/tencent/gpx/result/' + mode + '_' + accu_type + '.csv', index=False)


if __name__ == '__main__':
    main()
