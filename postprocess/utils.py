#coding=utf-8
import math
import time
import os
from math import radians, cos, sin, asin, sqrt
from matplotlib import pyplot as plt
x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 偏心率平方
import numpy as np
from xml.dom.minidom import parse
import xml.dom.minidom


def generate_gpx(trace, output_path, ts):
    '''
    generate .gpx files for map-matching
    :param trace: list, [[lon, lat, unix timestamp], ...]
    :param output_path:
    '''
    with open(output_path, 'w') as fout:
        fout.write('<gpx><trk><trkseg>\n')
        for item in trace:
            if ts == 'unix':
                fout.write('<trkpt lat="%f" lon="%f"><time>%s</time></trkpt>\n' % (item[1], item[0], time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(item[2]))))
            elif ts == 'str_T':
                fout.write('<trkpt lat="%f" lon="%f"><time>%s</time></trkpt>\n' % (item[1], item[0], item[2]))
            else:
                fout.write('<trkpt lat="%f" lon="%f"><time>%s</time></trkpt>\n' % (item[1], item[0], item[2].replace(' ', 'T')))
        fout.write('</trkseg></trk></gpx>')


def read_gpx(path, timestamp):
    DOMTree = xml.dom.minidom.parse(path)
    collection = DOMTree.documentElement
    records = collection.getElementsByTagName("trkpt")

    trace = []
    time_gap = []
    prev_time = 0
    for idx, record in enumerate(records):
        lat = float(record.getAttribute("lat").strip().replace(' ', ''))
        lon = float(record.getAttribute("lon").strip().replace(' ', ''))
        if timestamp:
            ts = record.getElementsByTagName('time')[0].childNodes[0].data.strip().replace(' ', '')
            time_unix = time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S"))
            if idx > 0:
                time_gap.append(time_unix - prev_time)
            prev_time = time_unix
            trace.append([lon, lat, time_unix])
        else:
            trace.append([lon, lat])
    return trace


def out_of_china(lng, lat):
    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)


def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def gcj02_to_wgs84(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    if out_of_china(lng, lat):
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def wgs84_to_gcj02(lng, lat):
    """
    WGS84转GCJ02(火星坐标系)
    :param lng:WGS84坐标系的经度
    :param lat:WGS84坐标系的纬度
    :return:
    """
    if out_of_china(lng, lat):  # 判断是否在国内
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [mglng, mglat]


def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000


def plot_line(data, gap, xl, yl):
    plt.figure(figsize=(16, 9))
    plt.plot(data)
    plt.xlabel(xl, fontsize=24)
    plt.xticks(range(len(data)), [str(i * gap) for i in range(len(data))], fontsize=20)
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


def plot_cdf(data, xl):
    sorted_data = np.sort(data)
    sorted_data = sorted_data[:int(len(sorted_data)*0.99)]
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    plt.plot(sorted_data, yvals)
    plt.xlabel(xl)
    # axes = plt.gca()
    # axes.set_xlim([0, 400])
    plt.ylabel('CDF')
    plt.show()


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def drop_dup(lst):
    unq = []
    for item in lst:
        if not unq or unq[-1] != item:
            unq.append(item)
    return unq


def evaluate_accuracy(preds, reals):
    # 先去除连续的重复，比如： 112223344511 → 123451
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
    accuracy = float(len(interset))/(max(len(preds),len(reals)))
    return accuracy


def read_seq2seq_trace(test_filenames, folder_seq2seq):
    """
    :return:
    """
    trace_dict = {}
    # seq2seq and real
    with open(folder_seq2seq + 'test_result.samp', 'r') as fin:
        lines = fin.readlines()
        lines = [line.replace('</s>', '') for line in lines]
        lines = [line.replace('<unk>', '') for line in lines]
        lines = [line.replace('<s>', '') for line in lines]
        for j in range(len(test_filenames)):
            trace_dict[test_filenames[j]] = lines[2 * j].strip().split()
            # trace_dict['real_seg'][test_filenames[j]] = lines[2 * j + 1].strip().split()
    return trace_dict


def read_trace(test_filenames, folder_seq2seq, folder_hmm, path_real_gps, folder_noise):
    """
    读取轨迹数据
    :param test_filenames:
    :param folder_seq2seq:
    :param folder_hmm:
    :return:
    """
    trace_dict = {'real_seg': {}, 'real_gps': {}, 'noise_gps': {}, 'seq2seq_seg': {}, 'hmm_seg': {}, 'hmm_gps': {}}

    # seq2seq and real
    with open(folder_seq2seq + 'test_result.samp', 'r') as fin:
        lines = fin.readlines()
        lines = [line.replace('</s>', '') for line in lines]
        lines = [line.replace('<unk>', '') for line in lines]
        lines = [line.replace('<s>', '') for line in lines]
        for j in range(int(len(lines) / 2)):
            trace_dict['seq2seq_seg'][test_filenames[j]] = lines[2 * j].strip().split()
            trace_dict['real_seg'][test_filenames[j]] = lines[2 * j + 1].strip().split()

    trace_dict['real_gps'] = read_gps_trace_data(path_real_gps)
    trace_dict['real_seg'] = read_gps_trace_data(path_real_gps)
    # for idx, name in enumerate(test_filenames):
    #     # TODO: 加入hmm结果
    #     # # hmm
    #     # try:
    #     #     with open(folder_hmm + name + '.seg', 'r') as fin:
    #     #         trace_dict['hmm_seg'][name] = drop_dup([item.strip() for item in fin.readlines()])
    #     # except:
    #     #     print('no hmm_seg: ' + name)
    #
    #     # real
    #     trace_dict['real_gps'][name] = read_gpx(folder_real + '/gps/' + name, timestamp=False)
    #     with open(folder_real + '/segments/' + name + '.seg', 'r') as fin:
    #         trace_dict['real_seg'][name] = drop_dup([item.strip() for item in fin.readlines()])

    return trace_dict


def read_gps_trace_data(path_in):
    with open(path_in, 'r') as fin:
        lines = fin.readlines()
        traces = {}
        for line in lines:
            trace = [[], []]
            line = line.strip().split(':')
            trace[0] = line[0]
            records = line[1].strip().split(',')
            for record in records:
                record = record.strip().split(' ')
                trace[1].append([float(record[0]), float(record[1]), int(float(record[2]))])
            traces[trace[0]] = trace[1]
    return traces


def read_seg_trace_data(path_in):
    with open(path_in, 'r') as fin:
        lines = fin.readlines()
        traces = {}
        for line in lines:
            line = line.strip().split(':')
            traces[line[0]] = drop_dup(line[1].strip().split(' '))
    return traces
