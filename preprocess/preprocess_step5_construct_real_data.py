import os
from xml.dom.minidom import parse
import xml.dom.minidom
from utils import haversine, mkdir, read_gps_trace_data, read_seg_trace_data, map_block
import random
import shutil

"""
生成seq2seq训练数据
"""


def drop_dup(lst):
    unq = []
    for item in lst:
        if not unq or unq[-1] != item:
            unq.append(item)
    return unq


def map_time(ts, ADD_TIME, init_ts):
    start_ts = 1538841600  # 2018-10-07-0:0:0
    if ADD_TIME == 'OneEncoding':
        return str(int((ts-start_ts)/60))
    elif ADD_TIME == 'TwoEncoding':
        return str(int((ts-start_ts)/60)//30) + '-' + str(int((ts - init_ts)/60))
    else:
        print('time encoding error')


def construct_data(noise_traces, gt_traces, REGION, SIDE_LENGTH):
    """
    :param filenames:
    :param gpx_folder: 加噪声轨迹所在文件夹
    :param dup_num:
    :param seg_folder: groundtruth所在文件夹
    :param REGION: 区域
    :return:
    """
    blocks_all = []
    tims1_all = []
    tims2_all = []
    segs_all = []

    trace_ids = list(set(noise_traces.keys()).intersection(set(gt_traces.keys())))
    print('trace num: ' + str(len(trace_ids)))
    count = 0
    for trace_id in trace_ids:  # TODO: sort
        if count % 1000 == 0:
            print(count)
        count += 1

        noise_trace = noise_traces[trace_id]
        gt_trace = gt_traces[trace_id]

        noise_trace = [[str(map_block(item[1], item[0], REGION, SIDE_LENGTH)), item[2]] for item in noise_trace]
        init_ts = noise_trace[0][1]
        noise_trace = drop_dup([[item[0], str(map_time(item[1], 'OneEncoding', init_ts)), str(map_time(item[1], 'TwoEncoding', init_ts))] for item in noise_trace])

        # source
        blocks = [item[0] for item in noise_trace]
        tims1 = [item[1] for item in noise_trace]
        tims2 = [item[2] for item in noise_trace]
        blocks_all.append(blocks)
        tims1_all.append(tims1)
        tims2_all.append(tims2)

        # read ground truth (segment trace)
        segs = drop_dup(gt_trace)
        segs_all.append(segs)

    return blocks_all, tims1_all, tims2_all, segs_all, trace_ids


def save_files(out_folder, blocks, segs, trace_ids, tims1, tims2, data_type):
    with open(out_folder + data_type + '.block', 'w') as fout_block, open(out_folder + data_type + '.seg', 'w') as fout_seg, \
            open(out_folder + data_type + '_trace_ids.txt', 'w') as fout_ids, open(out_folder + data_type + '.time1', 'w') as fout_time1, \
            open(out_folder + data_type + '.time2', 'w') as fout_time2:
        for i in range(len(blocks)):
            fout_block.write(' '.join(blocks[i]) + '\n')
        for i in range(len(segs)):
            fout_seg.write(' '.join(segs[i]).replace('\n', '') + '\n')
        fout_ids.write(','.join(trace_ids))

        for i in range(len(tims1)):
            fout_time1.write(' '.join(tims1[i]) + '\n')
        for i in range(len(tims2)):
            fout_time2.write(' '.join(tims2[i]) + '\n')


def real_train_real_test(paras, noise_folder, seg_folder, out_folder, train_size):
    '''
    :param paras:  参数
    :param noise_folder:  噪声轨迹所在文件夹
    :param seg_folder:  segment-based 轨迹所在文件夹
    :param out_folder: 输出文件夹
    :param train_size: 训练集的大小（主要用于控制不同time interval时，训练数据相同）
    :return:
    '''
    DUP_NUM = paras['DUP_NUM']
    REGION = paras['REGION']
    SIDE_LENGTH = paras['SIDE_LENGTH']

    noise_valid_traces = read_gps_trace_data(noise_folder + 'noise_real_valid.trace')
    noise_test_traces = read_gps_trace_data(noise_folder + 'noise_real_test.trace')

    gt_valid_traces = read_seg_trace_data(seg_folder + 'seg_matched_real_valid.trace')
    gt_test_traces = read_seg_trace_data(seg_folder + 'seg_matched_real_test.trace')

    # valid data
    print('validation data:')
    blocks_valid, tims1_valid, tims2_valid, segs_valid, trace_ids_valid = construct_data(noise_valid_traces, gt_valid_traces, REGION, SIDE_LENGTH)
    save_files(out_folder, blocks_valid, segs_valid, trace_ids_valid, tims1_valid, tims2_valid, 'valid')

    # test data
    print('test data:')
    blocks_test, tims1_test, tims2_test, segs_test, trace_ids_test = construct_data(noise_test_traces, gt_test_traces, REGION, SIDE_LENGTH)
    save_files(out_folder, blocks_test, segs_test, trace_ids_test, tims1_test, tims2_test, 'test')

    # train
    print('training data:')
    blocks_train = []
    tims1_train = []
    tims2_train = []
    segs_train = []
    trace_ids_train = []
    gt_train_traces = read_seg_trace_data(seg_folder + 'seg_matched_real_train.trace')
    for dup in range(DUP_NUM):
        print('dup: ' + str(dup))
        noise_train_traces = read_gps_trace_data(noise_folder + 'noise_real_train_dup_%d.trace' % dup)

        blocks_train_temp, tims1_train_temp, tims2_train_temp, segs_train_temp, trace_ids_train_temp = construct_data(noise_train_traces, gt_train_traces, REGION, SIDE_LENGTH)
        blocks_train += blocks_train_temp[:train_size]
        tims1_train += tims1_train_temp[:train_size]
        tims2_train += tims2_train_temp[:train_size]
        segs_train += segs_train_temp[:train_size]
        trace_ids_train += trace_ids_train_temp[:train_size]

    save_files(out_folder, blocks_train, segs_train, trace_ids_train, tims1_train, tims2_train, 'train')

    blocks_train_len = [len(item) for item in blocks_train]
    segs_train_len = [len(item) for item in segs_train]

    with open(out_folder + 'statistics.txt', 'w') as fout:
        fout.write('train trace number: ' + str(len(blocks_train)) + '\n')
        fout.write('train trace duplicate number: ' + str(DUP_NUM) + '\n')
        fout.write('validation trace number:  ' + str(len(trace_ids_valid)) + '\n')
        fout.write('test trace number:  ' + str(len(trace_ids_test)) + '\n')
        fout.write('average block number of a train trace:   ' + str(sum(blocks_train_len) / len(blocks_train_len)) + '\n')
        fout.write('average segment number of a train trace: ' + str(sum(segs_train_len) / len(segs_train_len)) + '\n')

        trace_num = len(blocks_train_len)
        blocks_train_len = sorted(blocks_train_len)
        segs_train_len = sorted(segs_train_len)
        for i in range(1, 10):
            fout.write('trace num: %d, block   trace length at percentage %d: %d \n' % (trace_num, i * 10, blocks_train_len[int(trace_num * i / 10)]))
            fout.write('trace num: %d, segment trace length at percentage %d: %d \n' % (trace_num, i * 10, segs_train_len[int(trace_num * i / 10)]))


def main():
    # 参数配置
    paras = {'DUP_NUM': 10,
             'REGION': 'beijing-south',
             'SIDE_LENGTH': 100}

    local_folder = '../../data/tencent/'

    for time_gap in [60]:  # 30, 40, 60, 80, 100, 120
        for noise in [100]:  # 10, 20, 40, 60, 80, 100, 120
            exp_name = 'timegap-%d_noise-gaussian_sigma-%d_dup-%d' % (time_gap, noise, paras["DUP_NUM"])
            noise_folder = local_folder + 'preprocessed/step4_real_addnoise/%s/' % exp_name  # 加噪声的轨迹
            seg_folder = local_folder + 'HMM_result/step3_real_split/run_all/sigma_60_beta_0.01/segments/'  # 真实轨迹HMM匹配后的轨迹
            out_folder = local_folder + 'seq2seq/real_fixlen/%s/dup-%d_sl-%d/' % (exp_name, paras['DUP_NUM'], paras['SIDE_LENGTH'])

            if os.path.exists(out_folder):
                print(out_folder + ' exists')
                return
            else:
                os.mkdir(out_folder)
            real_train_real_test(paras, noise_folder, seg_folder, out_folder, 6112)


if __name__ == '__main__':
    main()

