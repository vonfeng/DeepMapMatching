# coding=utf-8
import time
import math
import sys
import pandas as pd
from utils import *
import os
import random

'''
划分训练集、验证集和测试集
'''


def main():
    path_in = '../../data/tencent/preprocessed/step2_real_filtered/filtered_0.trace'
    path_out = '../../data/tencent/preprocessed/step3_real_split/'
    VALID_NUM = 1500
    TEST_NUM = 1500

    if os.path.exists(path_out):
        print(path_out + ' exists')
        return
    else:
        os.mkdir(path_out)

    with open(path_in, 'r') as fin:
        data = fin.readlines()

    random.seed(0)
    valid_test_traces = random.sample(data, VALID_NUM + TEST_NUM)
    train_traces = list(set(data) - set(valid_test_traces))
    random.seed(1)
    valid_traces = random.sample(valid_test_traces, VALID_NUM)
    test_traces = list(set(valid_test_traces) - set(valid_traces))

    random.shuffle(train_traces)
    random.shuffle(valid_traces)
    random.shuffle(test_traces)

    with open(path_out + 'real_train.trace', 'w') as fout1, open(path_out + 'real_valid.trace', 'w') as fout2, open(path_out + 'real_test.trace', 'w') as fout3:
        fout1.writelines(train_traces)
        fout2.writelines(valid_traces)
        fout3.writelines(test_traces)


if __name__ == '__main__':
    main()
