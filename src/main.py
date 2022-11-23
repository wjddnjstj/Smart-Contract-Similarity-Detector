import os
from database import Database
import json
import utils
import doc2vec_imp
import asm2vec_imp


def prepare_directory(config_dic):
    out_dir = config_dic['OUT']
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_training_dir = os.path.join(out_dir, config_dic["DATA"]["TRAINING_DIR"])
    if not os.path.isdir(out_training_dir):
        os.mkdir(out_training_dir)

    out_training_dir_opt = os.path.join(out_dir, config_dic["DATA"]["TRAINING_DIR_OPT"])
    if not os.path.isdir(out_training_dir_opt):
        os.mkdir(out_training_dir_opt)

    out_testing_dir = os.path.join(out_dir, config_dic["DATA"]["TESTING_DIR"])
    if not os.path.isdir(out_testing_dir):
        os.mkdir(out_testing_dir)

    out_testing_dir_opt = os.path.join(out_dir, config_dic["DATA"]["TESTING_DIR_OPT"])
    if not os.path.isdir(out_testing_dir_opt):
        os.mkdir(out_testing_dir_opt)

    opcode_dir = config_dic['OPCODE']
    if not os.path.isdir(opcode_dir):
        os.mkdir(opcode_dir)

    opc_training_dir = os.path.join(opcode_dir, config_dic["DATA"]["TRAINING_DIR"])
    if not os.path.isdir(opc_training_dir):
        os.mkdir(opc_training_dir)

    opc_training_dir_opt = os.path.join(opcode_dir, config_dic["DATA"]["TRAINING_DIR_OPT"])
    if not os.path.isdir(opc_training_dir_opt):
        os.mkdir(opc_training_dir_opt)

    opc_testing_dir = os.path.join(opcode_dir, config_dic["DATA"]["TESTING_DIR"])
    if not os.path.isdir(opc_testing_dir):
        os.mkdir(opc_testing_dir)

    opc_testing_dir_opt = os.path.join(opcode_dir, config_dic["DATA"]["TESTING_DIR_OPT"])
    if not os.path.isdir(opc_testing_dir_opt):
        os.mkdir(opc_testing_dir_opt)

    bin_dir = config_dic['BIN']
    if not os.path.isdir(bin_dir):
        os.mkdir(bin_dir)

    bin_training_dir = os.path.join(bin_dir, config_dic["DATA"]["TRAINING_DIR"])
    if not os.path.isdir(bin_training_dir):
        os.mkdir(bin_training_dir)

    bin_training_dir_opt = os.path.join(bin_dir, config_dic["DATA"]["TRAINING_DIR_OPT"])
    if not os.path.isdir(bin_training_dir_opt):
        os.mkdir(bin_training_dir_opt)

    bin_testing_dir = os.path.join(bin_dir, config_dic["DATA"]["TESTING_DIR"])
    if not os.path.isdir(bin_testing_dir):
        os.mkdir(bin_testing_dir)

    bin_testing_dir_opt = os.path.join(bin_dir, config_dic["DATA"]["TESTING_DIR_OPT"])
    if not os.path.isdir(bin_testing_dir_opt):
        os.mkdir(bin_testing_dir_opt)

    asm_dir = config_dic['ASM']
    if not os.path.isdir(asm_dir):
        os.mkdir(asm_dir)

    asm_training_dir = os.path.join(asm_dir, config_dic["DATA"]["TRAINING_DIR"])
    if not os.path.isdir(asm_training_dir):
        os.mkdir(asm_training_dir)

    asm_training_dir_opt = os.path.join(asm_dir, config_dic["DATA"]["TRAINING_DIR_OPT"])
    if not os.path.isdir(asm_training_dir_opt):
        os.mkdir(asm_training_dir_opt)

    asm_testing_dir = os.path.join(asm_dir, config_dic["DATA"]["TESTING_DIR"])
    if not os.path.isdir(asm_testing_dir):
        os.mkdir(asm_testing_dir)

    asm_testing_dir_opt = os.path.join(asm_dir, config_dic["DATA"]["TESTING_DIR_OPT"])
    if not os.path.isdir(asm_testing_dir_opt):
        os.mkdir(asm_testing_dir_opt)

    if not os.path.isdir(config_dic['LOG_DIR']):
        os.mkdir(config_dic['LOG_DIR'])


def main():
    f_config = open('./config.json', 'r')
    config_dic = json.load(f_config)
    f_config.close()

    utils.clean_dataset(config_dic)
    prepare_directory(config_dic)

    training_set = config_dic['TRAINING_SET']
    testing_set = config_dic['TESTING_SET']

    db = Database(config_dic)
    db.scan_files(training_set)
    db.scan_files(testing_set)

    if config_dic['ASM_2_VEC']:
        utils.bin2asm(config_dic)

        asm_dir = config_dic['ASM']
        asm_training_dir = os.path.join(asm_dir, config_dic["DATA"]["TRAINING_DIR"])
        asm_training_dir_opt = os.path.join(asm_dir, config_dic["DATA"]["TRAINING_DIR_OPT"])
        for proj in os.listdir(asm_training_dir):
            asm2vec_imp.train(config_dic, os.path.join(asm_training_dir, proj))

        for proj in os.listdir(asm_training_dir_opt):
            asm2vec_imp.train(config_dic, os.path.join(asm_training_dir_opt, proj))

        asm_testing_dir = os.path.join(asm_dir, config_dic["DATA"]["TESTING_DIR"])
        for tp in os.listdir(asm_testing_dir):
            asm2vec_imp.test(config_dic, os.path.join(asm_testing_dir, tp))

        asm2vec_imp.compare_sim(config_dic)
        asm2vec_imp.compare_contract_sim(config_dic)
    else:
        doc2vec_imp.train_model(config_dic)
        doc2vec_imp.compute_project_level_sim(config_dic)
        doc2vec_imp.compute_contract_level_sim(config_dic)


if __name__ == '__main__':
    main()
