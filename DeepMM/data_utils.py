#coding=utf-8
"""Data utilities."""
import torch
from torch.autograd import Variable
import operator
import json
import os


def hyperparam_string(config):
    """Hyerparam string."""
    exp_name = ''
    exp_name += '%s-' % (config['model']['seq2seq'])
    exp_name += '%s_%s-' % (config['model']['n_layers_src'], config['model']['n_layers_trg'])
    exp_name += '%s_%s-' % (config['model']['src_hidden_dim'], config['model']['trg_hidden_dim'])
    exp_name += '%s_%s-' % (config['model']['dim_loc_src'], config['model']['dim_seg_trg'])
    exp_name += '%s_%s-' % (config['model']['max_src_length'], config['model']['max_trg_length'])
    exp_name += '%s_%s_%s_%s-' % (config['model']['time_encoding'], config['model']['dim_tim1_src'], config['model']['dim_tim2_1_src'], config['model']['dim_tim2_2_src'])
    exp_name += '%.2f-' % (config['model']['dropout'])
    exp_name += '%s_%.4f_%s' % (config['training']['optimizer'], config['training']['lrate'], config['training']['batch_size'])

    return exp_name


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def construct_vocab(lines):
    """Construct a vocabulary from tokenized lines."""
    vocab = {}
    for line in lines:
        for word in line:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    # Discard start, end, pad and unk tokens if already present
    if '<s>' in vocab:
        del vocab['<s>']
    if '<pad>' in vocab:
        del vocab['<pad>']
    if '</s>' in vocab:
        del vocab['</s>']
    if '<unk>' in vocab:
        del vocab['<unk>']

    word2id = {
        '<s>': 0,
        '<pad>': 1,
        '</s>': 2,
        '<unk>': 3,
    }

    id2word = {
        0: '<s>',
        1: '<pad>',
        2: '</s>',
        3: '<unk>',
    }

    sorted_word2id = sorted(
        vocab.items(),
        key=operator.itemgetter(1),
        reverse=True
    )

    # sorted_words = [x[0] for x in sorted_word2id[:vocab_size]]
    sorted_words = [x[0] for x in sorted_word2id]

    for ind, word in enumerate(sorted_words):
        word2id[word] = ind + 4

    for ind, word in enumerate(sorted_words):
        id2word[ind + 4] = word

    return word2id, id2word


def read_dialog_summarization_data(src, config, trg):
    """Read data from files."""
    print('Reading source data ...')
    src_lines = []
    with open(src, 'r') as f:
        for ind, line in enumerate(f):
            src_lines.append(line.strip().split())

    print('Reading target data ...')
    trg_lines = []
    with open(trg, 'r') as f:
        for line in f:
            trg_lines.append(line.strip().split())

    print('Constructing common vocabulary ...')
    word2id, id2word = construct_vocab(
        src_lines + trg_lines, config['data']['n_words_src']
    )

    src = {'data': src_lines, 'word2id': word2id, 'id2word': id2word}
    trg = {'data': trg_lines, 'word2id': word2id, 'id2word': id2word}

    return src, trg


def read_nmt_data(dataset, config):
    """Read data from files."""
    # ************************ source ******************************
    src_tim_vocab_size = [0, [0, 0]]
    # location
    with open(config['data']['folder'] + config['data'][dataset]['src_loc'], 'r') as f:  # , encoding='utf-8') as f:
        src_loc_lines = [line.strip().split() for line in f]
    src_loc2id, src_id2loc = construct_vocab(src_loc_lines)
    src = {'traces_loc': src_loc_lines, 'loc2id': src_loc2id, 'id2loc': src_id2loc}
    del src_loc_lines
    # time
    if config['model']['time_encoding'] == 'OneEncoding':
        with open(config['data']['folder'] + config['data'][dataset]['src_tim1'], 'r') as f:  # , encoding='utf-8') as f:
            src_tim_lines = [line.strip().split() for line in f]
        src_time2id, src_id2time = construct_vocab(src_tim_lines)
        src['traces_tim1'] = src_tim_lines
        src['tim2id_1'] = src_time2id
        src['id2tim_1'] = src_id2time
        del src_tim_lines
        src_tim_vocab_size[0] = len(src['tim2id_1'])
    elif config['model']['time_encoding'] == 'TwoEncoding':
        with open(config['data']['folder'] + config['data'][dataset]['src_tim2'], 'r') as f:  # , encoding='utf-8') as f:
            src_tim_lines = [[tim.split('-') for tim in line.strip().split()] for line in f]
            src_tim_lines_1 = [[tim[0] for tim in line] for line in src_tim_lines]
            src_tim_lines_2 = [[tim[1] for tim in line] for line in src_tim_lines]
        src['traces_tim2_1'] = src_tim_lines_1
        src['traces_tim2_2'] = src_tim_lines_2
        src['tim2id_2_1'], src['id2tim_2_1'] = construct_vocab(src_tim_lines_1)
        src['tim2id_2_2'], src['id2tim_2_2'] = construct_vocab(src_tim_lines_2)
        del src_tim_lines, src_tim_lines_1, src_tim_lines_2
        src_tim_vocab_size[1] = len(src['tim2id_2_1']), len(src['tim2id_2_2'])

    # ************************ target ******************************
    with open(config['data']['folder'] + config['data'][dataset]['trg_seg'], 'r') as f:
        trg_seg_lines = [line.strip().split() for line in f]
    trg_seg2id, trg_id2seg = construct_vocab(trg_seg_lines)

    trg = {'traces_seg': trg_seg_lines, 'seg2id': trg_seg2id, 'id2seg': trg_id2seg}

    return src, trg, len(src['loc2id']), src_tim_vocab_size, len(trg['seg2id'])


# def read_nmt_data(src, config, time_encoding, trg=None, time=None):
#     """Read data from files."""
#     # print('Reading source data ...')
#     # ************************ source ******************************
#     src_loc_lines = []
#     with open(src, 'r') as f:  # , encoding='utf-8') as f:
#         for ind, line in enumerate(f):
#             src_loc_lines.append(line.strip().split())
#     # print('Constructing source vocabulary ...')
#     src_word2id, src_id2word = construct_vocab(src_loc_lines, config['data']['n_locs_src'])
#     src = {'traces_loc': src_loc_lines, 'loc2id': src_word2id, 'id2loc': src_id2word}
#
#     if time_encoding == 'OneEncoding':
#         # add time according to time_encoding
#         src_time_lines = []
#         with open(time, 'r') as f:  # , encoding='utf-8') as f:
#             for ind, line in enumerate(f):
#                 src_time_lines.append(line.strip().split())
#         src_time2id, src_id2time = construct_vocab(src_time_lines, config['data']['n_tims_src'])
#         src['traces_tim'] = src_time_lines
#         src['tim2id'] = src_time2id
#         src['id2tim'] = src_id2time
#         del src_time_lines
#
#     del src_loc_lines
#
#     # ************************ target ******************************
#     if trg is not None:
#         # print('Reading target data ...')
#         trg_lines = []
#         with open(trg, 'r') as f:
#             for line in f:
#                 trg_lines.append(line.strip().split())
#
#         # print('Constructing target vocabulary ...')
#         trg_word2id, trg_id2word = construct_vocab(trg_lines, config['data']['n_segs_trg'])
#
#         trg = {'traces_seg': trg_lines, 'seg2id': trg_word2id, 'id2seg': trg_id2word}
#     else:
#         trg = None
#
#     return src, trg


def read_summarization_data(src, trg):
    """Read data from files."""
    src_lines = [line.strip().split() for line in open(src, 'r')]
    trg_lines = [line.strip().split() for line in open(trg, 'r')]
    word2id, id2word = construct_vocab(src_lines + trg_lines, 30000)
    src = {'data': src_lines, 'word2id': word2id, 'id2word': id2word}
    trg = {'data': trg_lines, 'word2id': word2id, 'id2word': id2word}

    return src, trg


def get_minibatch(lines, word2ind, index, batch_size, max_len, add_start=True, add_end=True):
    """Prepare minibatch."""
    if add_start and add_end:
        lines = [
            ['<s>'] + line + ['</s>']
            for line in lines[index:index + batch_size]
        ]
    elif add_start and not add_end:
        lines = [
            ['<s>'] + line
            for line in lines[index:index + batch_size]
        ]
    elif not add_start and add_end:
        lines = [
            line + ['</s>']
            for line in lines[index:index + batch_size]
        ]
    elif not add_start and not add_end:
        lines = [
            line
            for line in lines[index:index + batch_size]
        ]
    lines = [line[:max_len] for line in lines]

    lens = [len(line) for line in lines]
    max_len = max(lens)

    input_lines = [
        [word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[:-1]] +
        [word2ind['<pad>']] * (max_len - len(line))
        for line in lines
    ]

    output_lines = [
        [word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[1:]] +
        [word2ind['<pad>']] * (max_len - len(line))
        for line in lines
    ]

    mask = [
        ([1] * (l - 1)) + ([0] * (max_len - l))
        for l in lens
    ]

    input_lines = Variable(torch.LongTensor(input_lines)).cuda()
    output_lines = Variable(torch.LongTensor(output_lines)).cuda()
    mask = Variable(torch.FloatTensor(mask)).cuda()

    return input_lines, output_lines, lens, mask


def get_autoencode_minibatch(
    lines, word2ind, index, batch_size,
    max_len, add_start=True, add_end=True
):
    """Prepare minibatch."""
    if add_start and add_end:
        lines = [
            ['<s>'] + line + ['</s>']
            for line in lines[index:index + batch_size]
        ]
    elif add_start and not add_end:
        lines = [
            ['<s>'] + line
            for line in lines[index:index + batch_size]
        ]
    elif not add_start and add_end:
        lines = [
            line + ['</s>']
            for line in lines[index:index + batch_size]
        ]
    elif not add_start and not add_end:
        lines = [
            line
            for line in lines[index:index + batch_size]
        ]
    lines = [line[:max_len] for line in lines]

    lens = [len(line) for line in lines]
    max_len = max(lens)

    input_lines = [
        [word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[:-1]] +
        [word2ind['<pad>']] * (max_len - len(line))
        for line in lines
    ]

    output_lines = [
        [word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[1:]] +
        [word2ind['<pad>']] * (max_len - len(line))
        for line in lines
    ]

    mask = [
        ([1] * (l)) + ([0] * (max_len - l))
        for l in lens
    ]

    input_lines = Variable(torch.LongTensor(input_lines)).cuda()
    output_lines = Variable(torch.LongTensor(output_lines)).cuda()
    mask = Variable(torch.FloatTensor(mask)).cuda()

    return input_lines, output_lines, lens, mask


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def time2dict(time_list):
    vocab = {}
    for line in time_list:
        for word in line:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
    for key in vocab:
        vocab[key] = int(key)
    ''''<s>': 0,
        '<pad>': 1,
        '</s>': 2,
        '<unk>': 3,'''
    vocab['<s>'] = 0
    vocab['<pad>'] = 1
    vocab['</s>'] = 2
    vocab['<unk>'] = 3
    return vocab
