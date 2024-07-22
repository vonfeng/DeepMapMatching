import os
from utils import drop_dup, mkdir, read_gps_trace_data, read_seg_trace_data, map_block


def map_time(ts, ADD_TIME, init_ts):
    start_ts = 1538841600  # 2018-10-07-0:0:0
    if ADD_TIME == 'OneEncoding':
        return str(int((ts-start_ts)/60))
    elif ADD_TIME == 'TwoEncoding':
        return str(int((ts-start_ts)/60)//30) + '-' + str(int((ts - init_ts)/60))
    else:
        print('time encoding error')


def construct_data(noise_traces, gt_traces, REGION, SIDE_LENGTH):
    '''
    :param noise_traces: noisy trajectory
    :param gt_traces: ground truth trajectory
    :param REGION: beijing or beijing-part
    :param SIDE_LENGTH: location block size
    :return:
    '''
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


def seq2seq_gen(paras, noise_folder, seg_folder, out_folder, num, train_size):
    '''
    :param paras: 参数
    :param noise_folder: 噪声轨迹所在文件夹
    :param seg_folder:  segment-based 轨迹所在文件夹
    :param out_folder: 输出文件夹
    :param num: 使用生成轨迹的数量，num*100000
    :param train_size: 训练集的大小（主要用于控制不同time interval时训练数据相同）
    :return:
    '''
    REGION = paras['REGION']
    SIDE_LENGTH = paras['SIDE_LENGTH']

    out_folder = out_folder + 'trace_%d00000/' % num
    if os.path.exists(out_folder):
        print(out_folder + ' exists')
        return
    mkdir(out_folder)

    noise_traces = {}
    gt_traces = {}
    for i in range(num):
        noise_traces_tmp = read_gps_trace_data(noise_folder + 'noisy_%d.trace' % i)
        noise_traces = dict(noise_traces, **noise_traces_tmp)
    for i in range(num):
        gt_traces_tmp = read_seg_trace_data(seg_folder + 'seg_matched_filtered_%d.trace' % i)
        gt_traces = dict(gt_traces, **gt_traces_tmp)

    blocks, tims1, tims2, segs, trace_ids = construct_data(noise_traces, gt_traces, REGION, SIDE_LENGTH)
    blocks, tims1, tims2, segs, trace_ids = blocks[:train_size], tims1[:train_size], tims2[:train_size], segs[:train_size], trace_ids[:train_size]
    save_files(out_folder, blocks, segs, trace_ids, tims1, tims2, 'train')

    blocks_len = [len(item) for item in blocks]
    segs_len = [len(item) for item in segs]

    with open(out_folder + 'statistics.txt', 'w') as fout:
        fout.write('train trace number: ' + str(len(blocks)) + '\n')
        fout.write('average block number of a train trace:   ' + str(sum(blocks_len) / len(blocks_len)) + '\n')
        fout.write('average segment number of a train trace: ' + str(sum(segs_len) / len(segs_len)) + '\n')

        trace_num = len(blocks_len)
        blocks_len = sorted(blocks_len)
        segs_len = sorted(segs_len)
        for i in range(1, 10):
            fout.write('trace num: %d, block   trace length at percentage %d: %d \n' % (trace_num, i * 10, blocks_len[int(trace_num * i / 10)]))
            fout.write('trace num: %d, segment trace length at percentage %d: %d \n' % (trace_num, i * 10, segs_len[int(trace_num * i / 10)]))


def main():
    # 参数配置，不用修改
    paras = {'REGION': 'beijing-south',
             'SIDE_LENGTH': 100
             }
    local_folder = 'D:/DeepMapMatching/data/tencent/'

    for time_gap in [30, 40, 60, 80, 100, 120]:  # [30, 90, 120]: 30, 40, 60, 80, 100, 120
        for noise in [100]:  # 10, 20, 40, 60, 80, 100, 120
            exp_name = 'timegap-%d_noise-gaussian_sigma-%d' % (time_gap, noise)
            noise_folder = local_folder + 'preprocessed/step4_gen_addnoise/%s/dup_0/' % exp_name  # 加噪声的轨迹
            seg_folder = local_folder + 'HMM_result/step2_gen/run_all/sigma_20_beta_0.01/segments/'  # 真实轨迹HMM匹配后的轨迹
            out_folder = local_folder + 'seq2seq/generate_fixlen/%s/sl-%d/' % (exp_name, paras['SIDE_LENGTH'])

            seq2seq_gen(paras, noise_folder, seg_folder, out_folder, 7, 649881)


if __name__ == '__main__':
    main()

