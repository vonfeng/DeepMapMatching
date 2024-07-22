#coding=utf-8
import pandas as pd
from utils import *
import os

"""
1.按时间间隔切割轨迹，
2.轨迹筛选：滤除太短的轨迹(5个点以下，5分钟以内)，滤除在200m（大致，按照经纬度0.002）范围内的移动轨迹（认为静止）
2.转换坐标系：火星坐标系 ——> 84坐标系
3.筛选数据日期：2018-10-07（基本都在这一天，有一些特例）
"""


def read_data(path, nrow, skiprows):
    data = pd.read_csv(path, skiprows=skiprows, nrows=nrow, sep='|', names=['id', 'time', 'mod', 'loc'], dtype={'id': str})
    data['lat'] = data['loc'].map(lambda x: float(x.split(',')[1]))
    data['lon'] = data['loc'].map(lambda x: float(x.split(',')[2]))
    data['date'] = data['time'].map(lambda x: x[:10])
    data['ts'] = data['loc'].map(lambda x: float(x.split(',')[-1])/1000)
    data = data[['id', 'date', 'ts', 'lat', 'lon']]
    return data


def trace_cut(ori_trace, TIME_THRESHOLD):
    '''
    根据时间间隔切割轨迹，停留超过TIME_THRESHOLD的地方切断
    :param ori_trace:原始轨迹
    :param TIME_THRESHOLD: 时间间隔阈值
    :return:
    '''
    sub_traces = []
    pre_idx = 0
    i = 0
    for i in range(1, ori_trace.shape[0]):
        delta_t = ori_trace.iloc[i]['ts'] - ori_trace.iloc[i-1]['ts']
        if delta_t > TIME_THRESHOLD:
            sub_traces.append(ori_trace.iloc[pre_idx:i, :])
            pre_idx = i
    sub_traces.append(ori_trace.iloc[pre_idx:i+1, :])
    return sub_traces


def filter_subtraces(sub_traces, POINT_NUM_THRESHOLD, TIME_SPAN_THRESHOLD, MIN_RANGE):
    '''
    筛选子轨迹并添加编号
    :param sub_traces: list of 切割后的轨迹
    :param POINT_NUM_THRESHOLD: 子轨迹的最少轨迹点数
    :param TIME_SPAN_THRESHOLD: 子轨迹的最短时间长度
    :return:
    '''
    result = []
    idx = 0
    for item in sub_traces:
        if item.shape[0] > POINT_NUM_THRESHOLD and item.iloc[-1]['ts'] - item.iloc[0]['ts'] > TIME_SPAN_THRESHOLD:
            if item['lon'].max() - item['lon'].min() > MIN_RANGE or item['lat'].max() - item['lat'].min() > MIN_RANGE:
                item['sub_idx'] = idx
                idx += 1
                result.append(item)
    return result


if __name__ == '__main__':
    # 参数设置
    LINES = 63126241  # 原始文件中轨迹总条数
    gap = 200000      # 由于原始文件过大，每次读取gap条轨迹处理
    # 用于筛选的参数
    TIME_THRESHOLD = 30 * 60
    POINT_NUM_THRESHOLD = 5
    TIME_SPAN_THRESHOLD = 5 * 60
    MIN_RANGE = 0.002  # 经度：170m, 纬度：222m
    DATE = '2018-10-07'  # 数据主要是这一天的，所以仅筛选这一天的数据

    # 路径设置
    file_in = '../../data/tencent/raw/lbs_track_points.log'
    file_out = '../../data/tencent/preprocessed/n-step1_%d_%d_%d_%d/' % (TIME_THRESHOLD, POINT_NUM_THRESHOLD, TIME_SPAN_THRESHOLD, MIN_RANGE*1000)
    file_out_stat = '../../data/tencent/preprocessed/n-step1_statistics_%d_%d_%d_%d.txt' % (TIME_THRESHOLD, POINT_NUM_THRESHOLD, TIME_SPAN_THRESHOLD, MIN_RANGE*1000)

    if os.path.exists(file_out):
        pass  # os.rmdir(file_out)
    else:
        os.mkdir(file_out)
    if os.path.exists(file_out_stat):
        os.remove(file_out_stat)

    # 筛选过程
    num = 0
    sub_trace_num = []
    sorted_sub_trace_num = []
    print(LINES/gap)
    # border = [180, 90, 0, 0]  # min_lon, min_lat, max_lon, max_lat
    while num < LINES/gap:
        concat_traces = []
        traces = read_data(file_in, skiprows=num*gap, nrow=gap)
        # border[0] = min(traces['lon'].min(), border[0])
        # border[1] = min(traces['lat'].min(), border[1])
        # border[2] = max(traces['lon'].max(), border[2])
        # border[3] = max(traces['lat'].max(), border[3])
        print(num)
        print('original user number: ' + str(len(sub_trace_num)))
        print('filtered user number: ' + str(len(sorted_sub_trace_num)))
        num += 1
        trace_len = []
        for id, group in traces.groupby(['id', 'date']):
            if id[1] == DATE:  # 筛选日期
                ori_trace = group.drop_duplicates().sort_values('ts')
                # cut
                sub_traces = trace_cut(ori_trace, TIME_THRESHOLD=TIME_THRESHOLD)
                sub_trace_num.append(len(sub_traces))
                # filter并添加编号
                sub_traces = filter_subtraces(sub_traces, POINT_NUM_THRESHOLD=POINT_NUM_THRESHOLD, TIME_SPAN_THRESHOLD=TIME_SPAN_THRESHOLD, MIN_RANGE=MIN_RANGE)
                # save
                if len(sub_traces) > 0:
                    sorted_sub_trace_num.append(len(sub_traces))
                    concat_trace = pd.concat(sub_traces)
                    # GPS坐标系转换
                    concat_trace['84gps'] = concat_trace.apply(lambda x: gcj02_to_wgs84(x['lon'], x['lat']), axis=1)
                    concat_trace['lon'] = concat_trace['84gps'].apply(lambda x: x[0])
                    concat_trace['lat'] = concat_trace['84gps'].apply(lambda x: x[1])
                    concat_traces.append(concat_trace[['id', 'sub_idx', 'ts', 'lat', 'lon']])
                    trace_len.append(concat_trace.shape[0])
        if len(concat_traces) > 0:
            pd.concat(concat_traces).to_csv(file_out + 'track_points_part_' + str(num//63) + '.csv', mode='a', header=False, index=False)

    print('original user number: ' + str(len(sub_trace_num)) + '\n')
    print('original average traces per user : ' + str(sum(sub_trace_num) / len(sub_trace_num)) + '\n')
    print('filtered user number: ' + str(len(sorted_sub_trace_num)) + '\n')
    print('filtered average traces per user : ' + str(sum(sorted_sub_trace_num) / len(sorted_sub_trace_num)) + '\n')

    with open(file_out_stat, 'w') as fout:
        # fout.write('border: ' + ' '.join([str(item) for item in border]) + '\n')
        fout.write('original user number: ' + str(len(sub_trace_num)) + '\n')
        fout.write('original average traces per user : ' + str(sum(sub_trace_num)/len(sub_trace_num)) + '\n')
        fout.write('filtered user number: ' + str(len(sorted_sub_trace_num)) + '\n')
        fout.write('filtered average traces per user : ' + str(sum(sorted_sub_trace_num) / len(sorted_sub_trace_num)) + '\n')
