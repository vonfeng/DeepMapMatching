# -*- coding: UTF-8 -*-
import random
from utils import *
import json

"""
添加噪声及降采样
"""


def add_random_noise(lat, lon, error=0.005):
    # 纬度1度距离： pi*R/180 = 111321.37574886571
    # 经度1度距离： 2*pi*R*cos(39.916642 / 180 * pi)/360 = 85381.13579579219
    lat += (random.random() - 0.5) * error
    lon += (random.random() - 0.5) * error
    return lat, lon


def add_gaussian_noise(lat, lon, lon_mu, lon_sigma, lat_mu, lat_sigma):
    # sigma: meter
    # 纬度1度距离： pi*R/180 = 111321.37574886571
    # 经度1度距离： 2*pi*R*cos(39.916642 / 180 * pi)/360 = 85381.13579579219
    lon += random.gauss(lon_mu, lon_sigma) / 85381.13579579219
    lat += random.gauss(lat_mu, lat_sigma) / 111321.37574886571
    return lat, lon


def preprocess_step4_addnoise(paras, folder_in, folder_out, file_out_stat):
    TIME_GAP = paras['TIME_GAP']
    NOISE_TYPE = paras['NOISE_TYPE']
    MIN_POINT_NUM = paras['MIN_POINT_NUM']
    DUP_NUM = paras['DUP_NUM']  # 数据增广的倍数

    for dup in range(DUP_NUM):
        if os.path.exists(folder_out + '/dup_' + str(dup) + '/'):
            print(folder_out + '/dup_' + str(dup) + ' exists')
            return
        mkdir(folder_out + '/dup_' + str(dup) + '/')

    filenames = os.listdir(folder_in)
    # 降采样加噪声
    trace_len = []
    out_time_gap_ave = []

    for dup in range(DUP_NUM):
        for file_idx, filename in enumerate(filenames):
            # read data
            with open(folder_in + filename, 'r') as fin:
                lines = fin.readlines()
                traces = []
                for line in lines:
                    trace = [[], []]
                    line = line.split(':')
                    trace[0] = line[0]
                    records = line[1].split(',')
                    for record in records:
                        record = record.split(' ')
                        trace[1].append([float(record[0]), float(record[1]), int(float(record[2]))])
                    traces.append(trace)
                # traces = pickle.load(fin)

            traces_filtered = []
            # one trace
            for idx, trace in enumerate(traces):
                if idx % 100000 == 0:
                    print(dup, idx)

                time_gap = []
                for i in range(1, len(trace[1])):
                    time_gap.append(trace[1][i][2]-trace[1][i-1][2])
                ave_time_gap = sum(time_gap)/len(time_gap)

                trace_ds = trace[1][:]
                # down sampling
                if ave_time_gap < TIME_GAP:
                    DOWN_SAMPLING_RATE = ave_time_gap / TIME_GAP
                    list_from = list(range(len(trace_ds)))
                    samp_num = int(len(trace_ds) * DOWN_SAMPLING_RATE)
                    trace_ds = [trace_ds[i] for i in sorted(random.sample(list_from, samp_num))]

                # add noise
                noisy_trace = []
                for record in trace_ds:
                    if NOISE_TYPE == 'gaussian':
                        lat, lon = add_gaussian_noise(record[1], record[0], paras['lon_MU'], paras['lon_SIGMA'], paras['lat_MU'], paras['lat_SIGMA'])
                    elif NOISE_TYPE == 'random':
                        lat, lon = add_random_noise(record[1], record[0], paras['SIGMA'])
                    elif NOISE_TYPE == 'no':
                        lat, lon = record[1], record[0]
                    else:
                        print('noise type error')
                        return 0
                    noisy_trace.append([lon, lat, record[2]])

                if len(noisy_trace) > MIN_POINT_NUM:
                    trace_len.append(len(noisy_trace))
                    time_gap = []
                    for i in range(1, len(noisy_trace)):
                        time_gap.append(noisy_trace[i][2] - noisy_trace[i-1][2])
                    out_time_gap_ave.append(sum(time_gap) / len(time_gap))

                    noisy_trace = ','.join([' '.join([str(x) for x in record]) for record in noisy_trace])
                    traces_filtered.append(trace[0] + ':' + noisy_trace)

            with open(folder_out + '/dup_' + str(dup) + '/noisy_' + str(file_idx) + '.trace', 'w') as fout:
                for item in traces_filtered:
                    fout.write(item + '\n')

    with open(file_out_stat, 'w') as fout:
        fout.write('TIME_GAP: ' + str(TIME_GAP) + '\n')
        fout.write('NOISE_TYPE: ' + NOISE_TYPE + '\n')
        fout.write('NOISE: lon_mu: %f, lon_sigma: %f, lat_mu: %f, lat_sigma: %f' % (paras['lon_MU'], paras['lon_SIGMA'], paras['lat_MU'], paras['lat_SIGMA']) + '\n')
        fout.write('MIN_POINT_NUM: ' + str(MIN_POINT_NUM) + '\n')
        fout.write('DUP_NUM: ' + str(DUP_NUM) + '\n')
        fout.write('output trace number: ' + str(len(trace_len)) + '\n')
        fout.write('output average time gap: ' + str(sum(out_time_gap_ave) / len(out_time_gap_ave)) + '\n')
        fout.write('output average trace point number: ' + str(sum(trace_len) / len(trace_len)) + '\n')


def main():
    # 参数配置，不用修改
    # para
    # time_gap = 60
    # noise = 50
    for time_gap in [60]:  # 30s, 40s, 60s, 80s, 90, 100s, 120s
        for noise in [100]:  # 10, 20, 30, 40, 50, 60, 80, 100, 120, 150
            para_path = '../../data/tencent/preprocessed/step4_real_addnoise/timegap-%d_noise-gaussian_sigma-%d_dup-20/gen_parameter.json' % (time_gap, noise)
            with open(para_path, 'r') as fin:
                paras = json.load(fin)
            # TIME_GAP = paras['TIME_GAP']
            # NOISE_TYPE = paras['NOISE_TYPE']
            # MIN_POINT_NUM = paras['MIN_POINT_NUM']
            # DUP_NUM = paras['DUP_NUM']
            # paras['lon_MU'], paras['lon_SIGMA'], paras['lat_MU'], paras['lat_SIGMA']

            # paras = {"TIME_GAP": 60, "NOISE_TYPE": "gaussian", "MIN_POINT_NUM": 5, "DUP_NUM": 10,
            #          "lon_MU": -0.16, "lon_SIGMA": 7.18, "lat_MU": -0.53, "lat_SIGMA": 6.27}

            output_name = 'timegap-%d_noise-gaussian_sigma-%d' % (time_gap, noise)
            folder_in = '../../data/tencent/preprocessed/step2_gen/'
            folder_out = '../../data/tencent/preprocessed/step4_gen_addnoise/%s/' % output_name

            # add noise
            preprocess_step4_addnoise(paras, folder_in, folder_out, folder_out + 'stat_gen.txt')


if __name__ == '__main__':
    main()
