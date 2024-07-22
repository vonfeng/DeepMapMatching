# coding=utf-8
import math
import time
import os
from math import radians, cos, sin, asin, sqrt
from matplotlib import pyplot as plt
from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np

x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 偏心率平方


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
            traces[line[0]] = line[1].strip().split(' ')
    return traces


def read_gpx(path):
    trace = []
    DOMTree = xml.dom.minidom.parse(path)
    collection = DOMTree.documentElement
    records = collection.getElementsByTagName("trkpt")
    for record in records:
        lat = float(record.getAttribute("lat").strip().replace(' ', ''))
        lon = float(record.getAttribute("lon").strip().replace(' ', ''))
        ts = record.getElementsByTagName('time')[0].childNodes[0].data
        trace.append([lon, lat, ts])
    return trace


def map_block(lat, lon, REGION, SIDE_LENGTH):
    # beijing: 	    116.24293854155346,39.80983695513636;
    #               116.50201886530907,39.99556678758318
    # 1/4 beijing: 	116.36128300718087,39.89214128985238;
    #               116.50201886530907,39.99556678758318

    if REGION == 'beijing':
        lon_min = 116.269684
        lon_max = 116.49877843024292
        lat_min = 39.826995
        lat_max = 39.993743
        # max_width = haversine(lon_min, lat_min, lon_max, lat_min)  # lon1, lat1, lon2, lat2
        # max_height = haversine(lon_min, lat_min, lon_min, lat_max)  # lon1, lat1, lon2, lat2
        col_num = 196  # round(max_width / 100)
        row_num = 185  # round(max_height / 100)
        lon_gap = 0.0011688491338924687  # (lon_max - lon_min) / col_num
        lat_gap = 0.00090134054054057  # (lat_max - lat_min) / row_num
        # print(col_num, row_num, lon_gap, lat_gap)

    elif REGION == 'beijing-part':
        lon_min = 116.3612830071808
        lon_max = 116.46
        lat_min = 39.8921412898523
        lat_max = 39.95556678758318

        max_width = 8422.005763509706  # haversine(lon_min, lat_min, lon_max, lat_min)  # lon1, lat1, lon2, lat2
        max_height = 7052.5935675800565  # haversine(lon_min, lat_min, lon_min, lat_max)  # lon1, lat1, lon2, lat2
        col_num = round(max_width / SIDE_LENGTH)
        row_num = round(max_height / SIDE_LENGTH)
        lon_gap = (lon_max - lon_min) / col_num
        lat_gap = (lat_max - lat_min) / row_num
    elif REGION == "beijing-south":
        lon_min = 116.269
        lon_max = 116.498
        lat_min = 39.827
        lat_max = 39.885

        max_width = haversine(lon_min, lat_min, lon_max, lat_min)  # lon1, lat1, lon2, lat2
        max_height = haversine(lon_min, lat_min, lon_min, lat_max)  # lon1, lat1, lon2, lat2
        col_num = round(max_width / SIDE_LENGTH)
        row_num = round(max_height / SIDE_LENGTH)
        lon_gap = (lon_max - lon_min) / col_num
        lat_gap = (lat_max - lat_min) / row_num

    else:
        print('loc map error')
        return

    row_idx = min(max(int((lat - lat_min) / lat_gap), 0), row_num)
    col_idx = min(max(int((lon - lon_min) / lon_gap), 0), col_num)
    idx = row_idx * col_num + col_idx
    return idx
