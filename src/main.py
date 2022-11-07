import os
from database import Database
import json
import utils
import doc2vec_imp
import asm2vec_imp


def main():
    f_config = open('./config.json', 'r')
    config_dic = json.load(f_config)
    f_config.close()

    if config_dic['ASM_2_VEC']:
        utils.bin2asm(config_dic)

        asm_dir = config_dic['ASM_REPO_DIR']
        for proj in os.listdir(asm_dir):
            asm2vec_imp.train(config_dic, os.path.join(asm_dir, proj))

        target_dir = config_dic['ASM_CONFIG']['TARGET_PROJ_DIR']
        for tp in os.listdir(target_dir):
            asm2vec_imp.test(config_dic, os.path.join(asm_dir, tp))

        asm2vec_imp.compare_sim(config_dic)
    else:
        db = Database(config_dic)
        db.scan_files()
        doc2vec_imp.train_model(config_dic)
        doc2vec_imp.compare_sim(config_dic)


if __name__ == '__main__':
    main()
