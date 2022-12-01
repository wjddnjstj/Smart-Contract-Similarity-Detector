import csv
import os.path
import r2pipe
import json
import shutil
import math
import numpy as np
from matplotlib import pyplot as plt
from sympy.physics.control.control_plots import matplotlib


def bin2asm(config):
    bin_dir = config['BIN']
    bin_training_dir = os.path.join(bin_dir, config["DATA"]["TRAINING_DIR"])
    bin_training_dir_opt = os.path.join(bin_dir, config["DATA"]["TRAINING_DIR_OPT"])
    bin_testing_dir = os.path.join(bin_dir, config["DATA"]["TESTING_DIR"])
    bin_testing_dir_opt = os.path.join(bin_dir, config["DATA"]["TESTING_DIR_OPT"])

    asm_dir = config['ASM']
    asm_training_dir = os.path.join(asm_dir, config["DATA"]["TRAINING_DIR"])
    asm_training_dir_opt = os.path.join(asm_dir, config["DATA"]["TRAINING_DIR_OPT"])
    asm_testing_dir = os.path.join(asm_dir, config["DATA"]["TESTING_DIR"])
    asm_testing_dir_opt = os.path.join(asm_dir, config["DATA"]["TESTING_DIR_OPT"])

    generate_inst(bin_training_dir, asm_training_dir)
    generate_inst(bin_training_dir_opt, asm_training_dir_opt)
    generate_inst(bin_testing_dir, asm_testing_dir)
    generate_inst(bin_testing_dir_opt, asm_testing_dir_opt)


def generate_inst(source, target):
    for proj in os.listdir(source):
        proj_dir = os.path.join(source, proj)
        for file in os.listdir(proj_dir):
            file_dir = os.path.join(proj_dir, file)
            r2 = r2pipe.open(file_dir)
            r2.cmd('aaa')  # analysis all
            asm_sub_dir = os.path.join(target, proj)
            if not os.path.isdir(asm_sub_dir):
                os.mkdir(asm_sub_dir)
            try:
                funcs = json.loads(r2.cmd('aflj'))  # list all functions
                for func in funcs:
                    func_name = func['name']
                    r2.cmd('s ' + func_name)  # move to the function start address
                    cfg = r2.cmdj("agj")  # get the control flow graph
                    cnt = 0
                    for graph in cfg:
                        blocks = graph['blocks']
                        for block in blocks:
                            instruction_path = asm_sub_dir + '/%s_%s_%d.txt' % (file, func_name, cnt)
                            cnt += 1
                            f = open(instruction_path, 'w')
                            inst = [i['opcode'] + '\n' for i in block['ops'] if 'opcode' in i]
                            for i in inst:
                                f.write(' ' + i)
                            f.close()
            except Exception as err:
                print(err)


def clean_dataset(config):
    gc = 0
    bc = 0
    uc = 0
    contracts = config['SRC_CONFIG']["CONTRACTS"]
    blacklist = config['SRC_CONFIG']["BLACK_LIST"]
    whitelist = config['SRC_CONFIG']["WHITE_LIST"]
    with open(whitelist) as f:
        white_list = (f.read().splitlines())
    f.close()
    print("Number of good contracts in white list: ", len(white_list))
    with open(blacklist) as f:
        black_list = (f.read().splitlines())
    f.close()
    print("Number of bad contracts in black list: ", len(black_list))
    for d1 in os.listdir(contracts):
        for d2 in os.listdir(os.path.join(contracts, d1)):
            p1 = os.path.join(contracts, d1)
            for d3 in os.listdir(os.path.join(p1, d2)):
                p2 = os.path.join(p1, d2)
                p3 = os.path.join(p2, d3)
                if d3 in black_list:
                    bc += 1
                    shutil.rmtree(p3)
                elif d3 in white_list:
                    gc += 1
                else:
                    uc += 1

    print(f'Found {gc} good, {bc} bad, and {uc} unknown contracts')

    training_set = config['TRAINING_SET']
    testing_set = config['TESTING_SET']
    if not os.path.isdir(training_set):
        os.mkdir(training_set)
    if not os.path.isdir(testing_set):
        os.mkdir(testing_set)
    training_set_count = math.ceil(gc * 0.8)
    count = 0
    for d1 in os.listdir(contracts):
        for d2 in os.listdir(os.path.join(contracts, d1)):
            p1 = os.path.join(contracts, d1)
            for d3 in os.listdir(os.path.join(p1, d2)):
                p2 = os.path.join(p1, d2)
                p3 = os.path.join(p2, d3)
                count = count + 1
                if count <= training_set_count:
                    if not os.path.isdir(os.path.join(training_set, d3)):
                        shutil.copytree(p3, os.path.join(training_set, d3))
                else:
                    if not os.path.isdir(os.path.join(testing_set, d3)):
                        shutil.copytree(p3, os.path.join(testing_set, d3))


def createPDF(simvalues):
    import warnings
    warnings.filterwarnings('ignore')
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    count, bins_count = np.histogram(simvalues, bins=100)
    pdf = count / sum(count)
    plt.plot(bins_count[1:], pdf, label="PDF")
    plt.legend()
    plt.show()


def generate_csv_report(filename, fields, rows):
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)
