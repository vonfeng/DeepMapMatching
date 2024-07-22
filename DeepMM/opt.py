import json
from itertools import product
import random

templatef = 'configs/config_best.json'
savef = 'configs/attention_noise/'

lr = [0.001]
layer = [[2, 1]]
hidden_dim = [512]
emb_dim = [256]
time_encoding = [['NoEncoding', 16, 32]]
noise = [10, 20, 40, 60, 80, 120]


config_product = product(lr, layer, hidden_dim, emb_dim, time_encoding, noise)
config_list = []
for i in config_product:
    config_list.append(i)

random.shuffle(config_list)

if __name__ == '__main__':
    with open(templatef, 'r') as template_file:
        load_temp_json = json.load(template_file)
        for idx, item in enumerate(config_list):
            #create_json = 'lr-%f_layer-%d-%d_hd-%d_ed-%d_te-%s-%d.json' % (item[0], item[1][0], item[1][1], item[2], item[3], item[4][0], item[4][1])
            create_json = 'config_noise_%d.json' % item[5]
            with open(savef + create_json, 'w+') as create_file:
                create_temp_json = load_temp_json
                create_temp_json['model']['max_src_length'] = 40
                create_temp_json['model']['max_trg_length'] = 54
                create_temp_json['training']['lrate'] = item[0]
                create_temp_json['model']['n_layers_src'] = item[1][0]
                create_temp_json['model']['n_layers_trg'] = item[1][1]
                create_temp_json['training']['batch_size'] = 128
                create_temp_json['model']['src_hidden_dim'] = item[2]
                create_temp_json['model']['trg_hidden_dim'] = item[2]
                create_temp_json['model']['dim_loc_src'] = item[3]
                create_temp_json['model']['dim_seg_trg'] = item[3]
                create_temp_json['data']['folder'] = '../timegap-60_noise-gaussian_sigma-%d/dup-10_sl-100/trace_700000/with_real_train/' % item[5]
                # fixed
                create_temp_json['management']['sample_num'] = 0
                create_temp_json['model']['time_encoding'] = item[4][0]
                if item[4][0] == 'OneEncoding':
                    create_temp_json['model']['dim_tim1_src'] = item[4][1]
                elif item[4][0] == 'TwoEncoding':
                    create_temp_json['model']['dim_tim2_1_src'] = item[4][1]
                    create_temp_json['model']['dim_tim2_2_src'] = item[4][2]
                json.dump(create_temp_json, create_file, indent=4)

            id = idx % 2
            with open('run_attention_noise_%d.sh' % id, 'a+') as sh_file:
                sh_file.write('python nmt_lr_decay.py --gpu=%d --config=%s'% (id+3, savef) + create_json + '\n')

