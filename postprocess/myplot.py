import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig_path = 'D:/DeepMapMatching/figs/'
lw = 5
ms = 14
tick_size = 40
label_size = 44


def plot_noise():
    seq2seq_path = 'D:/DeepMapMatching/data/tencent/seq2seq/combine/attention/noise.csv'
    hmm_path = 'D:/DeepMapMatching/data/tencent/HMM_result/test_data/test/'
    s2s = pd.read_csv(seq2seq_path)
    s2s['dataset'] = s2s['dataset'].apply(lambda x: int(float(x.split('_')[5])))
    s2s_mean = s2s.groupby('dataset').mean()
    s2s_std = s2s.groupby('dataset').std()

    hmm = pd.read_csv(hmm_path + 'noise.csv')

    # s2s_dict = dict(zip(s2s['dataset'].apply(lambda x: int(x.split('_')[5])), s2s['test_accu']))
    # hmm_dict = dict(zip(hmm['noise'], hmm['accu']))

    # data = [[], [], []]
    # for noise in [10, 20, 40, 60, 80, 100, 120]:
    #     data[0].append(noise)
    #     data[1].append(s2s_dict[noise])
    #     data[2].append(hmm_dict[noise])

    idx = s2s_mean.index.tolist()
    plt.figure(figsize=[11, 9])
    plt.plot(idx, s2s_mean['test_accu'], '-o', lw=5, ms=14)
    plt.plot(idx, hmm['accu'], '--s', lw=5, ms=14)
    plt.plot(idx, [0.64, 0.61, 0.56, 0.56, 0.54, 0.54, 0.50], '-.<', lw=5, ms=14)
    plt.fill_between(idx, s2s_mean['test_accu'] - s2s_std['test_accu'], s2s_mean['test_accu'] + s2s_std['test_accu'], facecolor='cornflowerblue', alpha=0.5)

    # plt.plot(data[0], data[1], '-o', lw=5, ms=14)
    # plt.plot(data[0], data[2], '--s', lw=5, ms=14)
    # plt.plot(data[0], [0.64, 0.61, 0.56, 0.56, 0.54, 0.54, 0.50], '-.<', lw=5, ms=14)
    plt.xticks([10, 20, 40, 60, 80, 100, 120], fontsize=40)
    plt.yticks(fontsize=40)
    plt.xlabel('Noise (meter)', fontsize=44)
    plt.ylabel('Accuracy', fontsize=44)
    plt.legend(['DeepMM', 'HMM', 'CTS'], fontsize=36)
    plt.tight_layout()
    plt.savefig(fig_path + 'noise_std.png')
    plt.show()


def plot_time_gap(seq2seq_path):
    # seq2seq_path = 'D:/DeepMapMatching/data/tencent/seq2seq/combine/attention/time_gap_emb256.csv'
    hmm_path = 'D:/DeepMapMatching/data/tencent/HMM_result/test_data/test/'

    s2s = pd.read_csv(seq2seq_path)
    s2s['dataset'] = s2s['dataset'].apply(lambda x: int(float(x.split('_')[3])))
    s2s_mean = s2s.groupby('dataset').mean()
    s2s_std = s2s.groupby('dataset').std()

    hmm = pd.read_csv(hmm_path + 'time_gap.csv')

    # s2s_dict = dict(zip(s2s['dataset'].apply(lambda x: int(x.split('_')[1])), s2s['test_accu']))
    # hmm_dict = dict(zip(hmm['time_gap'], hmm['accu']))

    # data = [[], [], []]
    # temp = []
    # para_list = sorted(s2s_dict.keys())
    # for time_gap in para_list:
    #     data[0].append(time_gap)
    #     data[1].append(s2s_dict[time_gap])
    #     # try:
    #     #     data[1].append(s2s_dict[time_gap])
    #     #     temp.append(time_gap)
    #     # except:
    #     #     pass
    #     data[2].append(hmm_dict[time_gap])

    idx = s2s_mean.index.tolist()
    plt.figure(figsize=[11, 9])
    plt.plot(idx, s2s_mean['test_accu'], '-o', lw=5, ms=14)
    plt.plot(idx, hmm['accu'], '--s', lw=5, ms=14)
    plt.plot(idx, [0.54, 0.54, 0.54, 0.49, 0.45, 0.44], '-.<', lw=5, ms=14)
    plt.fill_between(idx, s2s_mean['test_accu'] - s2s_std['test_accu'], s2s_mean['test_accu'] + s2s_std['test_accu'], facecolor='cornflowerblue', alpha=0.5)
    plt.xticks(idx, fontsize=38)
    plt.yticks(fontsize=38)
    plt.ylim([0.4, 0.75])
    plt.xlabel('Sampling Interval (second)', fontsize=44)
    plt.ylabel('Accuracy', fontsize=44)
    plt.legend(['DeepMM', 'HMM', 'CTS'], fontsize=36)
    plt.tight_layout()
    plt.savefig(fig_path + 'time_gap_std.png')
    plt.show()


def plot_dup_num(seq2seq_path):
    s2s = pd.read_csv(seq2seq_path + 'dup_num.csv')

    s2s_dict = dict(zip(s2s['dataset'].apply(lambda x: int(x.split('_')[0].split('-')[1])), s2s['test_accu']))

    data = [[], [], []]
    para_list = sorted(list(s2s_dict.keys()))  # [1, 2, 5, 10, 20]
    for time_gap in para_list:
        data[0].append(time_gap)
        data[1].append(s2s_dict[time_gap])

    plt.figure(figsize=[11, 9])
    plt.plot(data[0], data[1], '-o', lw=lw, ms=ms)
    plt.plot(data[0], [0.55]*len(data[1]), '--', lw=lw)

    plt.xticks(para_list, fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlabel('Duplication Times', fontsize=label_size)
    plt.ylabel('Accuracy', fontsize=label_size)
    plt.legend(['DeepMM', 'HMM'], fontsize=36)
    plt.tight_layout()
    plt.savefig(fig_path + 'dup_num.png')
    plt.show()


def plot_gen_num(seq2seq_path, shortest_path):
    s2s = pd.read_csv(seq2seq_path + 'gen_num.csv')
    s2s['dataset'] = s2s['dataset'].apply(lambda x:  int(float(x.split('_')[1])/85480))
    # s2s = s2s.groupby('dataset').mean()
    s2s_mean = s2s.groupby('dataset').mean()
    s2s_std = s2s.groupby('dataset').std()

    s2s_short = pd.read_csv(shortest_path + 'gen_num.csv')
    s2s_short['dataset'] = s2s_short['dataset'].apply(lambda x: int(float(x.split('_')[1]) / 85480))

    s2s_short_mean = s2s_short.groupby('dataset').mean().iloc[[0, 2, 3, 4]]
    # s2s_short = s2s_short.iloc[[0, 2, 3, 4]]
    s2s_short_std = s2s_short.groupby('dataset').std().iloc[[0, 2, 3, 4]]
    # s2s_short = s2s_short.iloc[[0, 2, 3, 4]]

    # s2s = s2s.iloc[[0, 2, 3, 4]]

    idx = s2s_mean.index.tolist()
    print(s2s_mean)
    print(s2s_short_mean)

    # s2s_dict = dict(zip(s2s['dataset'].apply(lambda x: int(float(x.split('_')[1])/85480)), s2s['test_accu']))
    # data = [[], [], []]
    # para_list = sorted(list(s2s_dict.keys()))
    # for time_gap in para_list:
    #     data[0].append(time_gap)
    #     data[1].append(s2s_dict[time_gap])

    plt.figure(figsize=[11, 9])
    plt.plot(idx, s2s_mean['test_accu'], '-o', lw=lw, ms=ms)
    # plt.fill_between(idx, s2s_mean['test_accu'] - s2s_std['test_accu'], s2s_mean['test_accu'] + s2s_std['test_accu'], facecolor='cornflowerblue', alpha=0.5)
    plt.plot(idx, [0.55] * len(idx), '--', lw=lw)
    plt.plot(idx, s2s_short_mean['test_accu'], '-.s', lw=lw, ms=ms)
    # plt.fill_between(idx, s2s_short_mean['test_accu'] - s2s_short_std['test_accu'], s2s_short_mean['test_accu'] + s2s_short_std['test_accu'], facecolor='cornflowerblue', alpha=0.5)

    plt.xticks(idx, fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlabel('# Generated / # Real', fontsize=label_size)
    plt.ylabel('Accuracy', fontsize=label_size)
    plt.legend(['DeepMM', 'HMM', 'DeepMM-shortest'], fontsize=36, loc='best', bbox_to_anchor=(0.25, 0.05, 0.3, 0.5))
    plt.tight_layout()
    plt.savefig(fig_path + 'gen_num.png')

    plt.show()


def plot_trace2trace_dist_accu(feature):
    data_path = 'D:/DeepMapMatching/data/tencent/gpx/result/%s_trace2trace_dist.csv' % feature
    data = pd.read_csv(data_path, header=0).sort_values(by=[feature])

    # s2s_mean = data[['seq2seq']].mean(axis=1)
    s2s_mean = data[['seq2seq_0', 'seq2seq_1', 'seq2seq_2']].mean(axis=1)
    s2s_std = data[['seq2seq_0', 'seq2seq_1', 'seq2seq_2']].std(axis=1)

    plt.figure(figsize=[11, 9])
    plt.plot(data[feature], s2s_mean, '-o', lw=lw, ms=ms)
    plt.fill_between(data[feature], s2s_mean - s2s_std, s2s_mean + s2s_std, facecolor='cornflowerblue', alpha=0.5)
    plt.plot(data[feature], data['hmm'], '--s', lw=lw, ms=ms)
    plt.plot(data[feature], data['cts'], '-.<', lw=lw, ms=ms)
    plt.xticks(data[feature], fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    if feature == 'time_gap':
        plt.xlabel('Sampling Interval (second)', fontsize=label_size)
        plt.ylim([35, 180])
    else:
        plt.xlabel('Noise (meter)', fontsize=label_size)
    plt.ylabel('Spatial Skewing (meter)', fontsize=label_size)
    plt.legend(['DeepMM', 'HMM', 'CTS'], fontsize=36, loc='upper left')
    plt.tight_layout()
    plt.savefig(fig_path + 't2t_dist_%s_std.png' % feature)

    plt.show()


def plot_tune_hidden():
    data_path = 'D:/DeepMapMatching/data/tencent/seq2seq/combine/timegap-60_noise-gaussian_sigma-100/dup-10_sl-100/trace_700000/attention_hidden.csv'
    data = pd.read_csv(data_path, header=0).sort_values(by=['hidden_dim'])

    std = data.groupby('hidden_dim').std()['test_accu']
    data = data.groupby('hidden_dim').mean()

    plt.figure(figsize=[11, 9])
    # plt.plot(range(data.shape[0]), data['test_accu'], '-o', lw=4, ms=10)
    # # plt.plot(range(data.shape[0]), [0.55]*data.shape[0], '--')
    # plt.xticks(range(data.shape[0]), data['hidden_dim'], fontsize=32)
    plt.plot(data.index.tolist(), data['test_accu'], '-o', lw=4, ms=10)

    plt.yticks(fontsize=32)
    # plt.ylim([0, max(data['test_accu']) + 0.1])
    plt.xlabel('Hidden Dimension', fontsize=40)
    plt.ylabel('Accuracy(%)', fontsize=40)
    # plt.fill_between(data.index.tolist(), data['test_accu'] - std, data['test_accu'] + std, facecolor='cornflowerblue', alpha=0.5)
    plt.xscale('log', basex=2)
    plt.xticks(data.index.tolist(), [128, 256, 512, 1024], fontsize=32)
    plt.tight_layout()
    plt.savefig(fig_path + 'attention_tune_hidden.png')

    plt.show()


def plot_tune_emb():
    data_path = 'D:/DeepMapMatching/data/tencent/seq2seq/combine/timegap-60_noise-gaussian_sigma-100/dup-10_sl-100/trace_700000/attention_emb.csv'
    data = pd.read_csv(data_path, header=0).sort_values(by=['src_loc_emb'])

    std = data.groupby('src_loc_emb').std().loc[[64, 128, 256, 512, 1024]]['test_accu']
    # std[1024] = 0.007
    data = data.groupby('src_loc_emb').mean()
    data = data.loc[[64, 128, 256, 512, 1024]]

    plt.figure(figsize=[11, 9])
    # plt.plot(range(data.shape[0]), data['test_accu'], '-o', lw=4, ms=10)
    # plt.xticks(range(data.shape[0]), data['trg_seg_emb'], fontsize=32)
    plt.plot(data['trg_seg_emb'], data['test_accu'], '-o', lw=4, ms=10)

    plt.yticks(fontsize=32)
    plt.ylim([0.64, 0.665])
    plt.xlabel('Embedding Dimension', fontsize=40)
    plt.ylabel('Accuracy(%)', fontsize=40)
    # plt.fill_between(data['trg_seg_emb'], data['test_accu'] - std, data['test_accu'] + std, facecolor='cornflowerblue', alpha=0.5)
    plt.xscale('log', basex=2)
    plt.xticks(data['trg_seg_emb'], [64, 128, 256, 512, 1024], fontsize=32)
    plt.tight_layout()
    plt.savefig(fig_path + 'attention_tune_emb.png')

    plt.show()


def plot_main():
    objects = ('HMM', 'CTS', 'DeepMM\nno-attention', 'DeepMM')
    y_pos = np.arange(len(objects))
    s2s_vanilla = [0.631748, 0.636024, 0.630955, 0.629761]
    s2s = [0.663056, 0.662684, 0.662296, 0.664456, 0.661958, 0.649988]
    performance_mean = [0.55, 0.54, np.mean(s2s_vanilla), np.mean(s2s)]
    print('mean', performance_mean)
    performance_std = [0, 0, np.std(s2s_vanilla), np.std(s2s)]

    plt.figure(figsize=[16, 9])
    plt.bar(y_pos, performance_mean, yerr=performance_std, align='center', alpha=0.8, color=['darkgreen', 'steelblue', 'sienna', 'firebrick'], width=0.5, error_kw=dict(lw=4))
    plt.xticks(y_pos, objects, fontsize=32)
    plt.yticks(fontsize=32)
    plt.ylim([0.5, 0.7])
    # plt.xlabel('algorithms')
    plt.ylabel('Accuracy', fontsize=36)
    plt.savefig(fig_path + 'main_result.png')

    plt.show()


def plot_embedding():
    emb_dist = [0.7618386060683957, 1.3041853845951406, 1.6541447974606798, 1.9617273915885953]
    plt.figure(figsize=[11, 9])
    plt.plot([1, 2, 3, 4], emb_dist, 'r-o', lw=lw, ms=ms)
    plt.plot([1, 2, 3, 4], [0.7618386060683957, 0.7618386060683957*2, 0.7618386060683957*3, 0.7618386060683957*4], 'k--', lw=lw, ms=ms)
    plt.xticks([1, 2, 3, 4], fontsize=tick_size)
    plt.yticks([0.7618386060683957, 0.7618386060683957*2, 0.7618386060683957*3, 0.7618386060683957*4], [1, 2, 3, 4], fontsize=tick_size)
    plt.xlabel('Spatial Distance', fontsize=label_size)
    plt.ylabel('Embedding Distance', fontsize=label_size)
    plt.text(2.2, 2.8, 'Reference Line', fontsize=tick_size)
    # plt.legend(['', 'Reference Line'], fontsize=tick_size)
    plt.tight_layout()
    plt.savefig(fig_path + 'embedding_visual.png')
    plt.show()


def main():
    plot_main()

    # plot_noise()
    # plot_time_gap('D:/DeepMapMatching/data/tencent/seq2seq/combine_fixlen/time_gap.csv')

    # plot_dup_num('D:/DeepMapMatching/data/tencent/seq2seq/combine/timegap-60_noise-gaussian_sigma-100/attention_old/')
    # plot_gen_num('D:/DeepMapMatching/data/tencent/seq2seq/combine/timegap-60_noise-gaussian_sigma-100/attention/')
    # plot_gen_num('D:/DeepMapMatching/data/tencent/seq2seq/combine/timegap-60_noise-gaussian_sigma-100/attention/', 'D:/DeepMapMatching/data/tencent/seq2seq/combine_shortest/')

    # plot_tune_hidden()
    # plot_tune_emb()

    # plot_trace2trace_dist_accu('noise')
    # plot_trace2trace_dist_accu('time_gap')

    # plot_embedding()


if __name__ == '__main__':
    main()
