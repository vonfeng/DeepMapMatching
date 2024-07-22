#coding=utf-8
import time
import math
import numpy as np
import pandas as pd
from utils import *
import os
import sys

"""
1. 截取区域
2. 滤除轨迹中的移动非常小或者位置没有动的点。即Hidden Markov map matching through noise and sparseness[C]中的预处理方法
3. 筛选轨迹：
    1. 平均时间间隔
    2. 轨迹点数
4. 保存成一行一条轨迹的格式
"""


def filter_2sigma(trace, sigma=0.001):  # sigma=50米
    """
    滤除轨迹中的移动非常小或者位置没有动的点。
    :param trace: 过滤前的轨迹
    :param sigma: 过滤的阈值
    :return: 过滤后的轨迹
    """
    if trace.shape[0] > 1:
        idx = [0]
        for i in range(1, trace.shape[0]):
            lat_dis = abs(trace.iloc[i]['lat'] - trace.iloc[idx[-1]]['lat'])
            lon_dis = abs(trace.iloc[i]['lon'] - trace.iloc[idx[-1]]['lon'])
            if lat_dis + lon_dis > sigma:
                idx.append(i)
        trace = trace.iloc[idx, :]
    return trace


def preprocess_2_filter(filenames, path, path_out, file_out_stat, paras):
    MAX_TIME_GAP = paras['MAX_TIME_GAP']
    MAX_AVE_TIME_GAP = paras['MAX_AVE_TIME_GAP']
    MIN_DIST = paras['MIN_DIST']
    MAX_GAP = paras['MAX_GAP']

    # 筛选
    filecount = 0
    gen_count = 0
    ori_count = 0
    points_list = []  # 每条生成轨迹的点数
    dist_list = []  # 每条生成轨迹的长度
    list_average_time_gap = []  # 每条生成轨迹的平均采样间隔
    traces = []
    for filename in filenames:
        print(filename)

        data = pd.read_csv(path + filename, header=None, names=['id', 'sub_idx', 'ts', 'lat', 'lon'], dtype={'id': str})
        for group_name, group in data.groupby(['id', 'sub_idx']):
            ori_count += 1
            if ori_count % 1000 == 0:
                print(ori_count)

            trace = group

            # 1.区域
            if paras['REGION'] == 'beijing-part':
                lon_min = 116.3612830071808
                lon_max = 116.46
                lat_min = 39.8921412898523
                lat_max = 39.95556678758318
                trace = trace[(trace['lat'] < lat_max) & (trace['lon'] < lon_max) & (trace['lat'] > lat_min) & (
                            trace['lon'] > lon_min)]
            elif paras['REGION'] == "beijing-south":
                lon_min = 116.269
                lon_max = 116.498
                lat_min = 39.827
                lat_max = 39.885
                trace = trace[(trace['lat'] < lat_max) & (trace['lon'] < lon_max) & (trace['lat'] > lat_min) & (
                        trace['lon'] > lon_min)]

            trace = filter_2sigma(trace, sigma=0.0002)
            if trace.shape[0] > 1:
                time_gaps = np.array(trace.iloc[1:]['ts']) - np.array(trace.iloc[:-1]['ts'])
                max_time_gap = np.max(time_gaps)
                average_time_gap = np.mean(time_gaps)
                if max_time_gap < MAX_TIME_GAP and average_time_gap < MAX_AVE_TIME_GAP:  # 1. 最大时间间隔， 2.平均时间间隔
                    trace_gps = pd.DataFrame(np.concatenate([np.array(trace.iloc[:-1][['lat', 'lon']]),
                                                             np.array(trace.iloc[1:][['lat', 'lon']])], axis=1),
                                             columns=['lat1', 'lon1', 'lat2', 'lon2'])
                    gap_dists = trace_gps.apply(lambda x: haversine(x['lon1'], x['lat1'], x['lon2'], x['lat2']), axis=1)
                    dist = gap_dists.sum()
                    max_gap = gap_dists.max()
                    if dist > MIN_DIST and max_gap < MAX_GAP:  # 3.轨迹内最大空间间隔， 4.轨迹长度
                        points_list.append(trace.shape[0])
                        dist_list.append(dist)
                        list_average_time_gap.append(average_time_gap)
                        trace = np.array(trace[['lon', 'lat', 'ts']]).tolist()
                        trace = ','.join([' '.join([str(x) for x in record]) for record in trace])
                        traces.append(group_name[0] + '_' + str(group_name[1]) + ':' + trace)

                        # 每*条轨迹保存一个文件
                        if gen_count > 0 and gen_count % 100000 == 0:
                            with open(path_out + 'filtered_' + str(filecount) + '.trace', 'w') as fout:
                                for item in traces:
                                    fout.write(item + '\n')
                                traces = []
                                filecount += 1
                            print(ori_count, gen_count, gen_count / ori_count)
                            # info = psutil.virtual_memory()
                            # print(u'内存使：', psutil.Process(os.getpid()).memory_info().rss)
                            # print(u'总内存：', info.total)
                        gen_count += 1

    with open(path_out + 'filtered_' + str(filecount) + '.trace', 'w') as fout:
        for item in traces:
            fout.write(item + '\n')

    with open(file_out_stat, 'w') as fout:
        fout.write('%dmin_%dmin: ' % (MAX_TIME_GAP//60, MAX_AVE_TIME_GAP//60) + '\n')
        fout.write('original trace number: ' + str(ori_count) + '\n')
        fout.write('filtered trace number: ' + str(len(points_list)) + '\n')
        fout.write('filtered average time gap: ' + str(sum(list_average_time_gap) / len(list_average_time_gap)) + '\n')
        fout.write('filtered average trace point number: ' + str(sum(points_list) / len(points_list)) + '\n')


def main():
    # 原始数据范围(四环及以内):  116.269684, 39.826995;               116.49877843024292, 39.993743
    # beijing-part:           116.3612830071808,39.8921412898523;  116.46,39.95556678758318

    # 筛选参数不用修改
    paras = {'MAX_TIME_GAP': 10 * 60,
             'MAX_AVE_TIME_GAP': 2 * 60,
             'MIN_DIST': 2000,
             'MAX_GAP': 1000,
             'REGION': 'beijing-south'
             }

    # 配置路径
    path = '../../data/tencent/preprocessed/step1_1800_5_300_2/'
    path_out = '../../data/tencent/preprocessed/step2_real_filtered/'
    path_out_stat = '../../data/tencent/preprocessed/step2_statistics_real.txt'
    filenames = os.listdir(path)

    if os.path.exists(path_out):
        print(path_out + ' exists')
        return
    else:
        os.mkdir(path_out)

    if os.path.exists(path_out_stat):
        os.remove(path_out_stat)
        # print(path_out_stat + ' exists')
        # return

    # do filter
    preprocess_2_filter(filenames, path, path_out, path_out_stat, paras)


if __name__ == '__main__':
    main()
