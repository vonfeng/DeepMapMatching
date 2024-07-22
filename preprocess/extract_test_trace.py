import shutil
import os
from utils import *


def main():
    """
    把测试集中的gps和segment轨迹提取出来
    :return:
    """
    sl = 100
    data_group = 'test'
    data_type = ''  # '_reserve_od', ''
    for time_gap in [30, 40, 60, 80, 100, 120]:  # 30, 40, 80, 100, 120
        for noise in [100]:  # 10, 20, 40, 60, 80, 100, 120
            filename = 'timegap-%d_noise-gaussian_sigma-%d_dup-20' % (time_gap, noise)
            filename2 = 'dup-10_sl-%d' % sl

            src_folder = 'D:/DeepMapMatching/data/tencent/HMM_result/step3_real_split/run_all/sigma_60_beta_0.01/'
            src_noise_folder = 'D:/DeepMapMatching/data/tencent/preprocessed/step4_real_addnoise%s/%s/' % (data_type, filename)
            trg_folder = 'D:/DeepMapMatching/data/tencent/test_data%s/%s/%s/%s/' % (data_type, data_group, filename, filename2)

            # test trace ids
            if data_group == 'test':
                with open('D:/DeepMapMatching/data/tencent/seq2seq/real%s/%s/%s/test_trace_ids.txt' % (data_type, filename, filename2), 'r') as fin:
                    trace_ids = fin.readlines()[0].strip().split(',')
            elif data_group == 'valid':
                trace_ids = read_gps_trace_data(src_folder + 'segments/seg_matched_real_valid.trace').keys()
            else:
                return

            for folder in ['gt/gps/', 'gt/segments/', 'noise/']:
                if os.path.exists(trg_folder + folder):
                    print(trg_folder + folder + ' exists')
                    return
                mkdir(trg_folder + folder)
            # gt
            with open(src_folder + 'gps/gps_matched_real_%s.trace' % data_group, 'r') as fin, open(trg_folder + 'gt/gps/gps_matched_real_%s.trace' % data_group, 'w') as fout:
                src_lines = fin.readlines()
                src_dict = {}
                for line in src_lines:
                    src_dict[line.split(':')[0]] = line
                for id in trace_ids:
                    fout.write(src_dict[id])
            with open(src_folder + 'segments/seg_matched_real_%s.trace' % data_group, 'r') as fin, open(trg_folder + 'gt/segments/seg_matched_real_%s.trace' % data_group, 'w') as fout:
                src_lines = fin.readlines()
                src_dict = {}
                for line in src_lines:
                    src_dict[line.split(':')[0]] = line
                for id in trace_ids:
                    fout.write(src_dict[id])

            # noise
            with open(src_noise_folder + 'noise_real_%s.trace' % data_group, 'r') as fin, open(trg_folder + 'noise/noise_real_%s.trace' % data_group, 'w') as fout:
                src_lines = fin.readlines()
                src_dict = {}
                for line in src_lines:
                    src_dict[line.split(':')[0]] = line
                for id in trace_ids:
                    try:
                        fout.write(src_dict[id])
                    except:
                        print('no noise trace')

            # for test_filename in test_filenames:
            #     shutil.copyfile(src_folder + 'segments/' + test_filename + '.seg', trg_folder + 'segments/' + test_filename + '.seg')
            #     shutil.copyfile(src_folder + 'gps/' + test_filename, trg_folder + 'gps/' + test_filename)


if __name__ == '__main__':
    main()
