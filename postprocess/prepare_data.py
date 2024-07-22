# coding=utf-8
import json
import random
import pickle
"""
为说明embedding域距离增长慢准备数据
"""


def map_block(lat, lon):
    # beijing: 	    116.24293854155346,39.80983695513636;
    #               116.50201886530907,39.99556678758318
    # 1/4 beijing: 	116.36128300718087,39.89214128985238;
    #               116.50201886530907,39.99556678758318
    lon_min = 116.3612830071808
    lon_max = 116.46
    lat_min = 39.8921412898523
    lat_max = 39.95556678758318

    max_width = 8422.005763509706  # haversine(lon_min, lat_min, lon_max, lat_min)  # lon1, lat1, lon2, lat2
    max_height = 7052.5935675800565  # haversine(lon_min, lat_min, lon_min, lat_max)  # lon1, lat1, lon2, lat2
    col_num = round(max_width / 100)  # 84
    row_num = round(max_height / 100)  # 71
    lon_gap = (lon_max - lon_min) / col_num
    lat_gap = (lat_max - lat_min) / row_num

    row_idx = min(max(int((lat - lat_min) / lat_gap), 0), row_num)
    col_idx = min(max(int((lon - lon_min) / lon_gap), 0), col_num)
    idx = row_idx * col_num + col_idx
    return idx


def main():
    # 读入seg_id与gps对应关系
    with open('../../data/map/road2gps/beijing_5thring.json', 'r') as fin:
        seg2gps = json.load(fin)

    all_segs = []
    for noise in [10]:  #, 20, 40, 60, 80, 100, 120]:
        path = 'D:/DeepMapMatching/data/tencent/seq2seq/combine/timegap-60_noise-gaussian_sigma-%d/dup-10_sl-100/trace_700000/with_real_train/train.seg' % noise
        with open(path, 'r') as fin:
            lines = [line.strip().split() for line in fin.readlines()]
            lines = [item for sublist in lines for item in sublist]
            segs = set(lines)  # segs.union(set(line.strip().split()))
        print(len(segs))
        all_segs.append(segs)
    inter_segs = all_segs[0]
    for i in range(1, len(all_segs)):
        inter_segs.intersection(all_segs[i])
    print('all: ', len(inter_segs))

    seg2blocks = {}
    count = 0
    for seg in inter_segs:
        try:
            blocks = []
            for item in seg2gps[seg]:
                blocks.append(int(map_block(item[0], item[1])))
            seg2blocks[seg] = list(set(blocks))
        except:
            count += 1
    print(len(inter_segs) - count)
    with open('../../data/map/seg2blocks.json', 'w') as fout:
        json.dump(seg2blocks, fout)


def construct_data():
    with open('../../data/map/seg2blocks.json', 'r') as fin:
        seg2blocks = json.load(fin)
    for N in [1, 2, 3, 4]:
        data = []
        for seg in seg2blocks:
            blocks = seg2blocks[seg]
            if len(blocks) > 3:
                blocks_skewed = []
                flag = 0
                skew_idx = random.sample(list(range(len(blocks))), N)
                for idx in skew_idx:
                    new_block = blocks[idx] + random.sample([-1, 1, -84, 84], 1)[0]
                    if 0 <= new_block <= 6048:
                        blocks_skewed.append(new_block)
                    else:
                        flag = 1
                        break
                for idx in set(range(len(blocks))) - set(skew_idx):
                    blocks_skewed.append(blocks[idx])
                if flag == 0:
                    data.append([sorted(blocks), sorted(blocks_skewed)])
        print(N, len(data))
        with open('D:/DeepMapMatching/data/tencent/seq2seq/combine/attention/data/n_%d.pkl' % N, 'w') as fout:
            pickle.dump(data, fout)


if __name__ == '__main__':
    # main()
    construct_data()
