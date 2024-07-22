# coding=utf-8
from utils import *
import json
import shutil


def map_seg2gps(seg2gps, seg_trace, noise_trace):
    """
    将seg序列构成的轨迹转换为gps形式的轨迹
    注意：一个路段有两种走向，所以每个路段都需要判断方向，走向判断利用路段端点之间的距离，距离近的端点拼接起来
    :param seg2gps: dict，每一个seg到gps的对应关系，一个seg可能有好多段
    :param seg_trace: list，seg序列形式的轨迹
    :return:
    """
    gps_trace = []
    order = []
    first_point = noise_trace[0]
    last_point = noise_trace[-1]
    for i in range(len(seg_trace)-1):
        min_dist = 1000000
        min_p = 0
        min_q = 0
        for idx_p, p in enumerate(seg2gps[seg_trace[i]]):
            for idx_q, q in enumerate(seg2gps[seg_trace[i+1]]):
                temp_dist = haversine(p[1], p[0], q[1], q[0])
                if temp_dist < min_dist:
                    min_p = idx_p
                    min_q = idx_q
                    min_dist = temp_dist
        order.append([min_p, min_q])
    # first point
    dist_list = []
    l = len(seg2gps[seg_trace[0]])
    for idx_p, p in enumerate(seg2gps[seg_trace[0]]):
        dist_list.append(haversine(p[1], p[0], first_point[0], first_point[1]))
    idx = dist_list.index(min(dist_list))
    if idx < order[0][0]:
        gps_trace = gps_trace + seg2gps[seg_trace[0]][idx:order[0][0]+1]
    else:
        gps_trace = gps_trace + seg2gps[seg_trace[0]][::-1][l-idx-1:l-order[0][0]]

    for i in range(1, len(order)):
        x = order[i-1][1]
        y = order[i][0]
        l = len(seg2gps[seg_trace[i]])
        if x < y:
            x, y = x, y + 1
            seg_gps = seg2gps[seg_trace[i]][x:y]
        else:
            x, y = l - x - 1, l - y
            seg_gps = seg2gps[seg_trace[i]][::-1][x:y]
        gps_trace += seg_gps

    # last point
    dist_list = []
    l = len(seg2gps[seg_trace[-1]])
    for idx_p, p in enumerate(seg2gps[seg_trace[-1]]):
        dist_list.append(haversine(p[1], p[0], last_point[0], last_point[1]))
    idx = dist_list.index(min(dist_list))
    if idx < order[-1][1]:
        gps_trace = gps_trace + seg2gps[seg_trace[-1]][idx:order[-1][1] + 1]
    else:
        gps_trace = gps_trace + seg2gps[seg_trace[-1]][::-1][l - idx - 1:l - order[-1][1]]

    return gps_trace


def sigma_para_tune_real():
    """
    给groundtruth选一个合适的sigma
    :return:
    """
    sigma = 60
    folder_out = 'D:/DeepMapMatching/data/tencent/gpx/real/hmm_sigma_%d/' % sigma
    for name in ['ori/', 'gt_gps/', 'gt_seg/']:
        if os.path.exists(folder_out + name):
            print(folder_out + name + ' exists')
            return
        mkdir(folder_out + name)

    # 读入seg_id与gps对应关系
    with open('../../data/map/road2gps/beijing_5thring.json', 'r') as fin:
        seg2gps = json.load(fin)

    # real
    ori_gps = read_gps_trace_data('D:/DeepMapMatching/data/tencent/preprocessed/step3_real_split/real_test.trace')
    gt_gps = read_gps_trace_data('D:/DeepMapMatching/data/tencent/HMM_result/step3_real_split/run_all/sigma_%d_beta_0.01/gps/gps_matched_real_test.trace' % sigma)
    gt_seg = read_seg_trace_data('D:/DeepMapMatching/data/tencent/HMM_result/step3_real_split/run_all/sigma_%d_beta_0.01/segments/seg_matched_real_test.trace' % sigma)

    test_trace_ids = list(set(ori_gps.keys()).intersection(set(gt_gps.keys())).intersection(gt_seg.keys()))

    # ori
    for trace_id in test_trace_ids:
        generate_gpx(ori_gps[trace_id], folder_out + 'ori/ori_gps_%s.gpx' % trace_id, ts='unix')

    # gt_gps
    for trace_id in test_trace_ids:
        generate_gpx(gt_gps[trace_id], folder_out + 'gt_gps/gt_gps_%s.gpx' % trace_id, ts='unix')

    # gt_seg_recover
    for trace_id in test_trace_ids:
        gps_list = drop_dup(map_seg2gps(seg2gps, gt_seg[trace_id]))
        gps_list = [[item[1], item[0], 0] for item in gps_list]
        generate_gpx(gps_list, folder_out + 'gt_seg/gt_seg_recover_%s.gpx' % trace_id, ts='unix')


def sigma_para_tune_gen():
    """
    给groundtruth选一个合适的sigma
    :return:
    """
    sigma = 40
    folder_out = 'D:/DeepMapMatching/data/tencent/gpx/generate/hmm_sigma_%d/' % sigma
    for name in ['ori/', 'gt_gps/', 'gt_seg/']:
        if os.path.exists(folder_out + name):
            print(folder_out + name + ' exists')
            return
        mkdir(folder_out + name)

    # 读入seg_id与gps对应关系
    with open('../../data/map/road2gps/beijing_5thring.json', 'r') as fin:
        seg2gps = json.load(fin)

    # real
    ori_gps = read_gps_trace_data('D:/DeepMapMatching/data/tencent/preprocessed/step2_gen_test/filtered_0.trace')
    gt_gps = read_gps_trace_data('D:/DeepMapMatching/data/tencent/HMM_result/step2_gen_test/tune_para/sigma_%d_beta_0.01_100/gps/gps_matched_filtered_0.trace' % sigma)
    gt_seg = read_seg_trace_data('D:/DeepMapMatching/data/tencent/HMM_result/step2_gen_test/tune_para/sigma_%d_beta_0.01_100/segments/seg_matched_filtered_0.trace' % sigma)

    test_trace_ids = list(set(ori_gps.keys()).intersection(set(gt_gps.keys())).intersection(gt_seg.keys()))

    # ori
    for trace_id in test_trace_ids:
        generate_gpx(ori_gps[trace_id], folder_out + 'ori/ori_gps_%s.gpx' % trace_id, ts='unix')

    # gt_gps
    for trace_id in test_trace_ids:
        generate_gpx(gt_gps[trace_id], folder_out + 'gt_gps/gt_gps_%s.gpx' % trace_id, ts='unix')

    # gt_seg_recover
    for trace_id in test_trace_ids:
        gps_list = drop_dup(map_seg2gps(seg2gps, gt_seg[trace_id]))
        gps_list = [[item[1], item[0], 0] for item in gps_list]
        generate_gpx(gps_list, folder_out + 'gt_seg/gt_seg_recover_%s.gpx' % trace_id, ts='unix')


def main_real():
    mode = 'noise'
    folder = 'D:/DeepMapMatching/data/tencent/gpx/result/' + mode + '/' # timegap-60_noise-gaussian_sigma-100_dup-10

    # 读入seg_id与gps对应关系
    with open('../../data/map/road2gps/beijing_5thring.json', 'r') as fin:
        seg2gps = json.load(fin)

    for time_gap in [60]:  # [30, 40, 60, 80, 100, 120]
        for noise in [100]:  # [10, 20, 40, 60, 80, 100, 120]:
            print('time gap: %d, noise: %d' % (time_gap, noise))
            filename = 'timegap-%d_noise-gaussian_sigma-%d_dup-20' % (time_gap, noise)
            folder_out = folder + filename + '/'

            for name in ['gt/', 'noise/', 'hmm/', 'cts/', 'seq2seq_0/', 'seq2seq_1/', 'seq2seq_2/']:
                if os.path.exists(folder_out + name):
                    print(folder_out + name + ' exists')
                    return
                mkdir(folder_out + name)

            best_para = {
                (60, 10): 'sigma_60_beta_0.01_40',
                (60, 20): 'sigma_60_beta_0.01_40',
                (60, 40): 'sigma_100_beta_0.01_40',
                (60, 60): 'sigma_140_beta_0.01_40',
                (60, 80): 'sigma_180_beta_0.01_40',
                (60, 120): 'sigma_240_beta_0.01_40',

                (30, 100): 'sigma_240_beta_0.01_71',
                (40, 100): 'sigma_240_beta_0.01_57',
                (60, 100): 'sigma_240_beta_0.01_40',
                (80, 100): 'sigma_200_beta_0.01_31',
                (100, 100): 'sigma_220_beta_0.01_25',
                (120, 100): 'sigma_200_beta_0.01_21',
            }

            # test_trace_ids
            if mode == 'time_gap':
                with open('D:/DeepMapMatching/data/tencent/seq2seq/real_fixlen/%s/dup-10_sl-100/test_trace_ids.txt' % filename, 'r') as fin:
                    test_trace_ids = fin.readlines()[0].strip().split(',')
            else:
                with open('D:/DeepMapMatching/data/tencent/seq2seq/real/%s/dup-10_sl-100/test_trace_ids.txt' % filename, 'r') as fin:
                    test_trace_ids = fin.readlines()[0].strip().split(',')

            # gt_seg
            gt_seg = read_seg_trace_data('D:/DeepMapMatching/data/tencent/HMM_result/step3_real_split/run_all/sigma_60_beta_0.01/segments/seg_matched_real_test.trace')
            with open(folder_out + 'gt_segments.trace', 'w') as fout:
                for trace_id in test_trace_ids:
                    fout.write(trace_id + ':' + ' '.join(gt_seg[trace_id]) + '\n')

            # gt_gps
            gt_gps = read_gps_trace_data('D:/DeepMapMatching/data/tencent/HMM_result/step3_real_split/run_all/sigma_60_beta_0.01/gps/gps_matched_real_test.trace')
            for trace_id in test_trace_ids:
                generate_gpx(gt_gps[trace_id], folder_out + 'gt/gt_%s.gpx' % trace_id, ts='unix')

            # noise
            noise_gps = read_gps_trace_data('D:/DeepMapMatching/data/tencent/preprocessed/step4_real_addnoise/%s/noise_real_test.trace' % filename)
            for trace_id in test_trace_ids:
                generate_gpx(noise_gps[trace_id], folder_out + 'noise/noise_%s.gpx' % trace_id, ts='unix')

            # hmm_gps
            hmm_gps = read_gps_trace_data('D:/DeepMapMatching/data/tencent/HMM_result/test_data/test/%s/dup-10_sl-100/tune_para/%s/gps/gps_matched_noise_real_test.trace' % (filename, best_para[(time_gap, noise)]))
            for trace_id in test_trace_ids:
                try:
                    generate_gpx(hmm_gps[trace_id], folder_out + 'hmm/hmm_%s.gpx' % trace_id, ts='unix')
                except:
                    print('no hmm result')
            # hmm_seg
            hmm_seg = read_seg_trace_data('D:/DeepMapMatching/data/tencent/HMM_result/test_data/test/%s/dup-10_sl-100/tune_para/%s/segments/seg_matched_noise_real_test.trace' % (filename, best_para[(time_gap, noise)]))
            with open(folder_out + 'hmm_segments.trace', 'w') as fout:
                for trace_id in test_trace_ids:
                    try:
                        fout.write(trace_id + ':' + ' '.join(hmm_seg[trace_id]) + '\n')
                    except:
                        print('no hmm seg result')

            # cts
            fn = os.listdir('D:/DeepMapMatching/data/tencent/test_data/cts/%s/' % (filename))[0]
            for trace in os.listdir('D:/DeepMapMatching/data/tencent/test_data/cts/%s/%s/gps/' % (filename, fn)):
                shutil.copyfile('D:/DeepMapMatching/data/tencent/test_data/cts/%s/%s/gps/' % (filename, fn) + trace, folder_out + 'cts/' + trace.replace('noise', 'cts'))
            traces_seg = []
            for trace in os.listdir('D:/DeepMapMatching/data/tencent/test_data/cts/%s/%s/segments/' % (filename, fn)):
                with open('D:/DeepMapMatching/data/tencent/test_data/cts/%s/%s/segments/' % (filename, fn) + trace) as fin:
                    lines = drop_dup([line.strip() for line in fin.readlines()])
                    traces_seg.append(trace[6:-8] + ':' + ' '.join(lines))
            with open(folder_out + 'cts_segments.trace', 'w') as fout:
                for trace in traces_seg:
                    fout.write(trace + '\n')

            # seq2seq
            for idx, fn in enumerate(os.listdir('D:/DeepMapMatching/data/tencent/test_data/%s/%s/dup-10_sl-100/seq2seq/' % (mode, filename))):
                seq2seq_seg_path = 'D:/DeepMapMatching/data/tencent/test_data/%s/%s/dup-10_sl-100/seq2seq/%s/test_result.samp' % (mode, filename, fn)
                seq2seq_seg = {}
                with open(seq2seq_seg_path, 'r') as fin:
                    lines = fin.readlines()
                    lines = [line.replace('</s>', '') for line in lines]
                    lines = [line.replace('<unk>', '') for line in lines]
                    lines = [line.replace('<s>', '') for line in lines]
                    for j in range(int(len(lines) / 2)):
                        seq2seq_seg[test_trace_ids[j]] = lines[2 * j].strip().split()
                # seg
                with open(folder_out + 'seq2seq_segments_%d.trace' % idx, 'w') as fout:
                    for trace_id in test_trace_ids:
                        try:
                            fout.write(trace_id + ':' + ' '.join(seq2seq_seg[trace_id]) + '\n')
                        except:
                            print('no seg')
                # gps recover
                for trace_id in test_trace_ids:
                    try:
                        gps_list = drop_dup(map_seg2gps(seg2gps, seq2seq_seg[trace_id], noise_gps[trace_id]))
                        gps_list = [[item[1], item[0], 0]for item in gps_list]
                        generate_gpx(gps_list, folder_out + 'seq2seq_%d/seq2seq_%s.gpx' % (idx, trace_id), ts='unix')
                    except:
                        print('no seg')


if __name__ == '__main__':
    # sigma_para_tune_real()
    # sigma_para_tune_gen()

    main_real()
