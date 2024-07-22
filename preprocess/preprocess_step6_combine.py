#coding=utf-8
import random
from utils import mkdir


def merge_gen_real(addRealTrain, real_folder, gene_folder, out_folder, gen_trace_num):
    """
    合并real和generate训练数据，使用real的测试数据
    :param addRealTrain: 是否加入real数据
    :param real_folder:
    :param gene_folder:
    :param out_folder:
    :param gen_trace_num: 加入generate轨迹的数量
    :return:
    """
    if addRealTrain:
        folder_name = 'with_real_train'
    else:
        folder_name = 'without_real_train'
    mkdir(out_folder + folder_name + '/')
    # train
    for filename in ['train.block', 'train.time1', 'train.time2', 'train.seg']:
        with open(real_folder + filename, 'r') as freal, open(gene_folder + filename, 'r') as fgene, \
                open(out_folder + folder_name + '/' + filename, 'w') as fout:
            train_real = freal.readlines()
            train_gene = fgene.readlines()
            if addRealTrain:
                train_out = train_real + train_gene[:gen_trace_num]
            else:
                train_out = train_gene
            random.seed(0)
            random.shuffle(train_out)
            fout.writelines(train_out)
    # valid
    for filename in ['valid.block', 'valid.time1', 'valid.time2', 'valid.seg']:
        with open(real_folder + filename, 'r') as freal, open(out_folder + folder_name + '/' + filename, 'w') as fout:
            test_real = freal.readlines()
            fout.writelines(test_real)

    # test
    for filename in ['test.block', 'test.time1', 'test.time2', 'test.seg']:
        with open(real_folder + filename, 'r') as freal, open(out_folder + folder_name + '/' + filename, 'w') as fout:
            test_real = freal.readlines()
            fout.writelines(test_real)


if __name__ == '__main__':
    sl = 100
    dup = 10
    # real number = 8548 * 10
    trace_num = 700000  # (8548 * 10) * 1
    for time_gap in [30, 40, 60, 80, 100, 120]:  # 30, 40, 60, 80, 100, 120
        for noise in [100]:  # 10, 20, 40, 60, 80, 100, 120
            real_folder = '../../data/tencent/seq2seq/real_fixlen/timegap-%d_noise-gaussian_sigma-%d_dup-20/dup-%d_sl-%d/' % (time_gap, noise, dup, sl)
            gene_folder = '../../data/tencent/seq2seq/generate_fixlen/timegap-%d_noise-gaussian_sigma-%d/sl-%d/trace_700000/' % (time_gap, noise, sl)
            out_folder = '../../data/tencent/seq2seq/combine_fixlen/timegap-%d_noise-gaussian_sigma-%d/dup-%d_sl-%d/trace_%d/' % (time_gap, noise, dup, sl, trace_num)
            merge_gen_real(addRealTrain=True, real_folder=real_folder, gene_folder=gene_folder, out_folder=out_folder, gen_trace_num=trace_num)
