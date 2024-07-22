# coding=utf8
import os
from utils import *
from scipy.stats import norm, pearsonr
import numpy as np
import matplotlib.pyplot as plt

'''
analyze the osm trajectory dataset (Matěj Kubička, et al. Dataset for testing and training of map-matching algorithms. In 2015 IEEE Intelligent Vehicles Symposium (IV). IEEE, 1088–1093.)
1. trajectory length
2. noise distribution
'''


def osm_traj_len(path_raw):
    '''
    average length of raw trajectories and ground truth trajectories
    :return:
    '''
    mt_len = []
    rt_len = []
    # extract_data
    for i in range(100):
        name = str(i)
        while len(name) < 8:
            name = '0' + name
        arcs = {}
        with open(path_raw + name + '/' + name + '.arcs') as fin:
            lines = [item.strip().split('\t') for item in fin.readlines()]
            lines = [[int(item[0]), int(item[1])] for item in lines]
            for i in range(len(lines)):
                arcs[i] = lines[i]
        nodes = {}
        with open(path_raw + name + '/' + name + '.nodes') as fin:
            lines = [item.strip().split('\t') for item in fin.readlines()]
            lines = [[float(item[0]), float(item[1])] for item in lines]
            for i in range(len(lines)):
                nodes[i] = lines[i]

        # matched trace
        route = []
        with open(path_raw + name + '/' + name + '.route') as fin:
            lines = [item.strip() for item in fin.readlines()]
            for i in range(len(lines)):
                route.append(int(lines[i]))
        route_gps = []
        for rec in route:
            start = nodes[arcs[rec][0]]
            end = nodes[arcs[rec][1]]
            route_gps.append(start + [0])
        mt_len.append(len(route_gps))

        # raw trace
        track = []
        with open(path_raw + name + '/' + name + '.track') as fin:
            lines = [item.strip().split('\t') for item in fin.readlines()]
            lines = [[float(item[0]), float(item[1]), int(float(item[2]))] for item in lines]
            for i in range(len(lines)):
                track.append(lines[i])
        rt_len.append(len(track))
    print(sum(mt_len) / float(len(mt_len)))
    print(sum(rt_len) / float(len(rt_len)))


def noise_distribution_based_on_dist(traces_real, traces_noise):
    noise = []
    lon_noise = []
    lat_noise = []
    for idx, key in enumerate(traces_real):
        print(idx)
        trace_real = traces_real[key]
        trace_noise = traces_noise[key]

        for rec_real in trace_real:
            dist = []
            for rec_noise in trace_noise:
                dist.append(haversine(rec_real[0], rec_real[1], rec_noise[0], rec_noise[1]))
            noise.append(min(dist))
            rec_noise = trace_noise[dist.index(min(dist))]
            lon_noise.append(85381.13579579219 * (rec_noise[0] - rec_real[0]))
            lat_noise.append(111321.37574886571 * (rec_noise[1] - rec_real[1]))

    mean = np.mean(np.array(noise), axis=0)
    cov = np.cov(np.array(noise), rowvar=0)
    print('point num: ', len(noise))
    print('mean:', mean)
    print('cov: ', cov)

    return noise, lon_noise, lat_noise


def osm2gpx(path_raw, out_path):
    # extract_data
    for i in range(100):
        name = str(i)
        while len(name) < 8:
            name = '0' + name
        arcs = {}
        with open(path_raw + name + '/' + name + '.arcs') as fin:
            lines = [item.strip().split('\t') for item in fin.readlines()]
            lines = [[int(item[0]), int(item[1])] for item in lines]
            for i in range(len(lines)):
                arcs[i] = lines[i]
        nodes = {}
        with open(path_raw + name + '/' + name + '.nodes') as fin:
            lines = [item.strip().split('\t') for item in fin.readlines()]
            lines = [[float(item[0]), float(item[1])] for item in lines]
            for i in range(len(lines)):
                nodes[i] = lines[i]

        # matched trace
        route = []
        with open(path_raw + name + '/' + name + '.route') as fin:
            lines = [item.strip() for item in fin.readlines()]
            for i in range(len(lines)):
                route.append(int(lines[i]))
        route_gps = []
        for rec in route:
            start = nodes[arcs[rec][0]]
            end = nodes[arcs[rec][1]]
            route_gps.append(start + [0])

        generate_gpx(route_gps, out_path + 'groundtruth/' + name + '.gpx', ts='unix')

        # raw trace
        track = []
        with open(path_raw + name + '/' + name + '.track') as fin:
            lines = [item.strip().split('\t') for item in fin.readlines()]
            lines = [[float(item[0]), float(item[1]), int(float(item[2]))] for item in lines]
            for i in range(len(lines)):
                track.append(lines[i])
        generate_gpx(track, out_path + '/noise/' + name + '.gpx', ts='unix')


def osm_noise_dist(fig_path, path_real, path_noise):
    '''
    noise distribution
    :param fig_path: save sigure
    :param path_real: ground truth trajectories
    :param path_noise: noisy trajectories
    :return:
    '''
    # read data
    traces_real = {}
    for name in os.listdir(path_real):
        trace = []
        for rec in read_gpx(path_real + name):
            rec = [rec[0], rec[1], time.mktime(time.strptime(rec[2], "%Y-%m-%dT%H:%M:%S"))]
            trace.append(rec)
        traces_real[name] = trace

    traces_noise = {}
    for name in os.listdir(path_noise):
        trace = []
        for rec in read_gpx(path_noise + name):
            rec = [rec[0], rec[1], time.mktime(time.strptime(rec[2], "%Y-%m-%dT%H:%M:%S"))]
            trace.append(rec)
        traces_noise[name] = trace

    noise, lon_noise, lat_noise = noise_distribution_based_on_dist(traces_real, traces_noise)

    lon_noise_limit = []
    lat_noise_limit = []
    for item1, item2 in zip(lon_noise, lat_noise):
        if abs(item1) < 50 and abs(item2) < 50:
            lon_noise_limit.append(item1)
            lat_noise_limit.append(item2)
    print('point num: ' + str(len(lon_noise_limit)))

    print('pearsonr: ', pearsonr(lon_noise, lat_noise))
    mean = np.mean(np.array(list(zip(lon_noise, lat_noise))), axis=0)
    cov = np.cov(np.array(list(zip(lon_noise, lat_noise))), rowvar=0)
    print('mean:' + str(mean))
    print('cov: ' + str(cov))

    # gaussian 拟合
    mean_lon, std_lon = norm.fit(lon_noise)
    mean_lat, std_lat = norm.fit(lat_noise)
    print('fitting')
    print('lon: ', mean_lon, std_lon)
    print('lat: ', mean_lat, std_lat)

    # ---------------- lon ------------------
    plt.figure(figsize=[12, 9])
    # 数据分布曲线
    plt.hist(lon_noise, bins=100, normed=True)
    # 拟合曲线
    x = np.arange(min(lon_noise), max(lon_noise), 0.1)
    mean, std = norm.fit(lon_noise)
    y = norm.pdf(x, mean, std)
    plt.plot(x, y, lw=3)
    plt.xlabel('Spatial Noise (meter)', fontsize=40)
    plt.ylabel('Probability', fontsize=40)
    plt.tick_params(labelsize=32)
    plt.tight_layout()
    plt.savefig(fig_path + 'real_noise_lon_dist.png')
    plt.show()

    # ---------------- lat ------------------
    plt.figure(figsize=[12, 9])
    # 数据分布曲线
    plt.hist(lat_noise, bins=100, normed=True)
    # 拟合曲线
    x = np.arange(min(lat_noise), max(lat_noise), 0.1)
    mean, std = norm.fit(lat_noise)
    y = norm.pdf(x, mean, std)
    plt.plot(x, y, lw=3)
    plt.xlabel('Spatial Noise (meter)', fontsize=40)
    plt.ylabel('Probability', fontsize=40)
    plt.tick_params(labelsize=32)
    plt.tight_layout()
    plt.savefig(fig_path + 'real_noise_lat_dist.png')
    plt.show()


if __name__ == '__main__':
    # trajectory length
    osm_traj_len('D:/DeepMapMatching/data/map-matching-dataset/raw/')

    # transfer to gpx format
    osm2gpx('D:/DeepMapMatching/data/map-matching-dataset/raw/', 'D:/DeepMapMatching/data/map-matching-dataset/')

    # noise distribution
    osm_noise_dist('D:/DeepMapMatching/figs/', 'D:/DeepMapMatching/data/map-matching-dataset/groundtruth/', 'D:/DeepMapMatching/data/map-matching-dataset/noise/')

