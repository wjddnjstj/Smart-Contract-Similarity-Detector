import os
import random
import torch
import gensim
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models.doc2vec import TaggedDocument
import numpy as np


def preprocess(inst_path, data_dir):
    f = open(os.path.join(data_dir, inst_path), 'r')
    inst = f.readlines()
    f.close()
    return inst


def load_func(inst_path, data_dir):
    instructions = preprocess(inst_path, data_dir)
    sentences = TaggedDocument(instructions, [0])
    return sentences


def load_data(data_dir):
    instructions = [preprocess(d, data_dir) for d in os.listdir(data_dir)]
    sentences = [TaggedDocument(l, [i]) for i, l in enumerate(instructions)]
    random.shuffle(sentences)
    return sentences


def train_model(config: dict):
    sentences = []
    op_training_dir = os.path.join(config['OPCODE'], config['DATA']['TRAINING_DIR'])
    op_training_dir_opt = os.path.join(config['OPCODE'], config['DATA']['TRAINING_DIR_OPT'])
    for proj in os.listdir(op_training_dir):
        sentences += load_data(os.path.join(op_training_dir, proj))
    for proj in os.listdir(op_training_dir_opt):
        sentences += load_data(os.path.join(op_training_dir_opt, proj))
    # dm = 1 means ‘distributed memory’ (PV-DM)
    # dm = 0 means ‘distributed bag of words’ (PV-DBOW)
    model = gensim.models.doc2vec.Doc2Vec(dm=0)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=50)

    if not os.path.isdir(config['MODEL_DIR_DOC2VEC']):
        os.mkdir(config['MODEL_DIR_DOC2VEC'])
    torch.save(model, config['MODEL_DIR_DOC2VEC'] + 'doc2vec.pt')


def compare_contract_sim(config: dict):
    model = torch.load(config['MODEL_DIR_DOC2VEC'] + 'doc2vec.pt')
    op_testing_dir = os.path.join(config['OPCODE'], config['DATA']['TESTING_DIR'])
    op_testing_dir_opt = os.path.join(config['OPCODE'], config['DATA']['TESTING_DIR_OPT'])
    # op_testing_dir_opt = os.path.join(config['OPCODE'], config['DATA']['TESTING_DIR_SELF'])
    # op_testing_dir_opt = os.path.join(config['OPCODE'], config['DATA']['TESTING_DIR_MOD'])
    correctness = 0
    total = 0
    for proj in os.listdir(op_testing_dir):
        for cont in os.listdir(os.path.join(op_testing_dir, proj)):
            target_cont = load_func(cont, os.path.join(op_testing_dir, proj))
            tc_vec = model.infer_vector(target_cont[0])
            max_cont_sim = 0
            max_cont_name = ""
            total = total + 1
            for proj_opt in os.listdir(op_testing_dir_opt):
                for cont_opt in os.listdir(os.path.join(op_testing_dir_opt, proj_opt)):
                    repo_cont = load_func(cont_opt, os.path.join(op_testing_dir_opt, proj_opt))
                    rc_vec = model.infer_vector(repo_cont[0])
                    sim = cosine_similarity(tc_vec, rc_vec)
                    sim = cosine_similarity(tc_vec, rc_vec)
                    if sim > max_cont_sim:
                        max_cont_sim = sim
                        max_cont_name = proj_opt + '/' + cont_opt
            print('*****************************************')
            print('Most similar contract of ' + proj + '/' + cont + ' is ' + max_cont_name)
            print('The similarity is {}'.format(max_cont_sim))
            print('*****************************************')

            if proj in max_cont_name and cont in max_cont_name:
                correctness = correctness + 1
    print('Accuracy of smart contract similarity detection is {}'.format(correctness / total))


def compare_sim(config: dict):
    model = torch.load(config['MODEL_DIR_DOC2VEC'] + 'doc2vec.pt')
    op_testing_dir = os.path.join(config['OPCODE'], config['DATA']['TESTING_DIR'])
    op_testing_dir_opt = os.path.join(config['OPCODE'], config['DATA']['TESTING_DIR_OPT'])
    # op_testing_dir_opt = os.path.join(config['OPCODE'], config['DATA']['TESTING_DIR_SELF'])
    # op_testing_dir_opt = os.path.join(config['OPCODE'], config['DATA']['TESTING_DIR_MOD'])
    correctness = 0
    for opcode in os.listdir(op_testing_dir):
        max_proj_sim = 0
        max_proj_name = ""
        for opcode_opt in os.listdir(op_testing_dir_opt):
            target = os.path.join(op_testing_dir, opcode)
            source = os.path.join(op_testing_dir_opt, opcode_opt)
            if len(os.listdir(target)) > 0:
                proj_sim = compute_proj_sim(target, source, model)
                if proj_sim > max_proj_sim:
                    max_proj_sim = proj_sim
                    max_proj_name = source.rsplit('/', 1)[-1]
        if len(os.listdir(target)) > 0:
            print('*****************************************')
            print('Most similar project of ' + target.rsplit('/', 1)[-1] + ' is ' + max_proj_name)
            print('The similarity is {}'.format(max_proj_sim))
            print('*****************************************')
        if target.rsplit('/', 1)[-1] in max_proj_name:
            correctness = correctness + 1
    print('Accuracy of smart contract similarity detection is {}'.format(correctness/len(os.listdir(op_testing_dir))))


def compute_proj_sim(target, source, model):
    # print('=========================================')
    total_func_sim = 0
    # tp = target.rsplit('/', 1)[-1]
    # rp = source.rsplit('/', 1)[-1]
    # print('target project: ', tp)
    # print('repo project: ', rp)
    for tf in os.listdir(target):
        target_func = load_func(tf, target)
        tf_vec = model.infer_vector(target_func[0])
        max_func_sim = 0
        for rf in os.listdir(source):
            repo_func = load_func(rf, source)
            rp_vec = model.infer_vector(repo_func[0])
            sim = cosine_similarity(tf_vec, rp_vec)
            max_func_sim = max(max_func_sim, sim)
            # print('sim("{}", "{}") = {}'.format(tf, rf, sim))
        total_func_sim += max_func_sim
    if len(os.listdir(target)) > 0:
        proj_sim = total_func_sim / len(os.listdir(target))
    else:
        proj_sim = 0
    # print('sim("{}", "{}") = {}'.format(tp, rp, proj_sim))
    # print('=========================================')
    return proj_sim


def cosine_similarity(v1, v2):
    return round(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 4)
