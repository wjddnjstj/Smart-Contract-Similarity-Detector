import os
import json


def my_bin2asm():
    f_config = open('../src/config.json', 'r')
    config_dic = json.load(f_config)
    f_config.close()
    config_dic['TRAINING_SET'] = "./training/"
    print(config_dic.get('TRAINING_SET'))

my_bin2asm()
