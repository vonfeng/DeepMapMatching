# -*- coding: UTF-8 -*-
import random
from utils import *
from scipy.stats import norm
import json

"""
statistic-based data augmentation, subsampling and adding nosie
"""


def add_random_noise(lat, lon, error=0.005):
    # 纬度1度距离： pi*R/180 = 111321.37574886571
    # 经度1度距离： 2*pi*R*cos(39.916642 / 180 * pi)/360 = 85381.13579579219
    lat += (random.random() - 0.5) * error
    lon += (random.random() - 0.5) * error
    return lat, lon


def add_gaussian_noise(lat, lon, sigma):
    # sigma: meter
    # 纬度1度距离： pi*R/180 = 111321.37574886571
    # 经度1度距离： 2*pi*R*cos(39.916642 / 180 * pi)/360 = 85381.13579579219
    lat += random.gauss(0, sigma) / 111321.37574886571
    lon += random.gauss(0, sigma) / 85381.13579579219
    return lat, lon


def preprocess_step4_addnoise(paras, path_in, path_out, file_out_stat):
    TIME_GAP = paras['TIME_GAP']
    NOISE_TYPE = paras['NOISE_TYPE']
    SIGMA = paras['SIGMA']
    MIN_POINT_NUM = paras['MIN_POINT_NUM']

    # 降采样加噪声
    trace_len = []
    out_time_gap_ave = []

    # read data
    with open(path_in, 'r') as fin:
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
        if idx % 1000 == 0:
            print(idx)

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
                lat, lon = add_gaussian_noise(record[1], record[0], SIGMA)
            elif NOISE_TYPE == 'random':
                lat, lon = add_random_noise(record[1], record[0], SIGMA)
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

    with open(path_out, 'w') as fout:
        for item in traces_filtered:
            fout.write(item + '\n')

    with open(file_out_stat, 'w') as fout:
        fout.write('TIME_GAP: ' + str(TIME_GAP) + '\n')
        fout.write('NOISE_TYPE: ' + NOISE_TYPE + '\n')
        fout.write('SIGMA: ' + str(SIGMA) + '\n')
        fout.write('output trace number: ' + str(len(trace_len)) + '\n')
        fout.write('output average time gap: ' + str(sum(out_time_gap_ave) / len(out_time_gap_ave)) + '\n')
        fout.write('output average trace point number: ' + str(sum(trace_len) / len(trace_len)) + '\n')


def get_gaussian_para(time_gap, noise, dup_num):
    '''
    :param time_gap:
    :param noise:
    :return:
    '''
    output_name = 'timegap-%d_noise-gaussian_sigma-%d_dup-%d' % (time_gap, noise, dup_num)
    out_path = '../../data/tencent/preprocessed/step4_real_addnoise/%s/gen_parameter.json' % output_name
    path_noise = '../../data/tencent/preprocessed/step4_real_addnoise/%s/noise_real_train_dup_0.trace' % output_name

    path_real = '../../data/tencent/preprocessed/step3_real_split/real_train.trace'
    traces_real = read_gps_trace_data(path_real)
    traces_noise = read_gps_trace_data(path_noise)

    time_gap = []
    for trace in traces_noise.values():
        for i in range(1, len(trace)):
            time_gap.append(trace[i][2] - trace[i-1][2])
    ave_tim_gap = sum(time_gap)/len(time_gap)
    print('average time gap: ' + str(ave_tim_gap))

    noise_lon = []
    noise_lat = []
    for idx, key in enumerate(traces_real):
        trace_real = traces_real[key]
        if idx % 1000 == 0:
            print(idx)
        try:
            trace_noise = traces_noise[key]
        except:
            continue

        for rec_noise in trace_noise:
            min_t = float('inf')
            min_rec_real = trace_real[0]
            for rec_real in trace_real:
                if abs(rec_real[-1] - rec_noise[-1]) < min_t:
                    min_t = abs(rec_real[-1] - rec_noise[-1])
                    min_rec_real = rec_real
            noise_lon.append(85381.13579579219 * (min_rec_real[0] - rec_noise[0]))
            noise_lat.append(111321.37574886571 * (min_rec_real[1] - rec_noise[1]))

    mean_lon, std_lon = norm.fit(noise_lon)
    mean_lat, std_lat = norm.fit(noise_lat)
    print('lon: ', mean_lon, std_lon)
    print('lat: ', mean_lat, std_lat)
    info_json = {'TIME_GAP': ave_tim_gap,
                 'NOISE_TYPE': 'gaussian',  # gaussian, random, no
                 'lon_MU': mean_lon,
                 'lon_SIGMA': std_lon,
                 'lat_MU': mean_lat,
                 'lat_SIGMA': std_lat,  # meter
                 'DUP_NUM': 1,   # 数据增广的倍数，生成数据不用数据增广，所以置位1即可
                 'MIN_POINT_NUM': 5
                 }
    with open(out_path, 'w') as fout:
        json.dump(info_json, fout)

    noise_lon_np = np.array(noise_lon)
    noise_lat_np = np.array(noise_lat)
    print('lon: ', np.mean(noise_lon_np), np.std(noise_lon_np))
    print('lat: ', np.mean(noise_lat_np), np.std(noise_lat_np))


def main(tim_gap, sigma):
    # statistic-based data augmentation, subsampling and adding nosie
    # 参数配置
    paras = {'TIME_GAP': tim_gap,
             'NOISE_TYPE': 'gaussian',
             'SIGMA': sigma,  # meter
             'DUP_NUM': 10,   # 数据增广的倍数
             'MIN_POINT_NUM': 5
             }

    output_name = 'timegap-%d_noise-%s_sigma-%d_dup-%d' % (paras['TIME_GAP'], paras['NOISE_TYPE'], paras['SIGMA'], paras['DUP_NUM'])
    folder_in = '../../data/tencent/preprocessed/step3_real_split/'
    folder_out = '../../data/tencent/preprocessed/step4_real_addnoise/%s/' % output_name

    if os.path.exists(folder_out):
        print(folder_out + ' exists')
        return
    else:
        os.mkdir(folder_out)

    # train add noise
    for dup in range(paras['DUP_NUM']):
        preprocess_step4_addnoise(paras, folder_in + 'real_train.trace', folder_out + 'noise_real_train_dup_' + str(dup) + '.trace', folder_out + 'stat_train_dup_' + str(dup) + '.txt')
    # validation
    preprocess_step4_addnoise(paras, folder_in + 'real_valid.trace', folder_out + 'noise_real_valid.trace', folder_out + 'stat_valid.txt')
    # test
    preprocess_step4_addnoise(paras, folder_in + 'real_test.trace', folder_out + 'noise_real_test.trace', folder_out + 'stat_test.txt')


if __name__ == '__main__':
    for time_gap in [60]:  # [30, 90, 120]: 30s, 40s, 60s, 80s, 100s, 120s
        for sigma in [100]:  # [10, 30, 50, 100, 150]  20, 40, 60, 80, 100, 120
            main(time_gap, sigma)
            get_gaussian_para(time_gap, sigma, dup_num=10)
