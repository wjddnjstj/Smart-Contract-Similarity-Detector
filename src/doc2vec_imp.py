import os
import random
from datetime import datetime
import torch
import gensim
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import utils
from tqdm import tqdm


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
    model.train(sentences, total_examples=model.corpus_count, epochs=30)

    if not os.path.isdir(config['MODEL_DIR_DOC2VEC']):
        os.mkdir(config['MODEL_DIR_DOC2VEC'])
    torch.save(model, config['MODEL_DIR_DOC2VEC'] + 'doc2vec.pt')


# This function is to compute the actual similarity value between any two projects
def project_similarity(target, source, model):
    total_func_sim = 0

    for tf in os.listdir(target):
        target_func = load_func(tf, target)
        tf_vec = model.infer_vector(target_func[0])
        max_func_sim = 0
        for rf in os.listdir(source):
            repo_func = load_func(rf, source)
            rp_vec = model.infer_vector(repo_func[0])
            sim = cosine_similarity(tf_vec, rp_vec)
            max_func_sim = max(max_func_sim, sim)
        total_func_sim += max_func_sim
    if len(os.listdir(target)) > 0:
        proj_sim = total_func_sim / len(os.listdir(target))
    else:
        proj_sim = 0
    return proj_sim


# This function is to compute the cosine similarity between any two vectors
def cosine_similarity(v1, v2):
    return round(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 4)


# This function is to loop through data in both testing and testing_optimized directory to compare the project-level
# similarity between each other
def compute_project_level_sim(config: dict, test: bool):
    pl_similarity_value = []
    uuid = datetime.now().strftime("%y%m%dT%H%M%S")
    model = torch.load(config['MODEL_DIR_DOC2VEC'] + 'doc2vec.pt')
    if not test:
        op_dir = os.path.join(config['OPCODE'], config['DATA']['TRAINING_DIR'])
        op_dir_opt = os.path.join(config['OPCODE'], config['DATA']['TRAINING_DIR_OPT'])

        res_dir_path = os.path.join(config['RESULT'], config['REPORT']['PROJ_SIM'])

        filename = os.path.join(res_dir_path, 'doc2vec_proj_similarity_report_' + uuid + '.csv')
        fields = ['project A', 'project B', 'similarity']
        rows = []

    else:
        op_dir = os.path.join(config['OPCODE'], config['DATA']['TESTING_DIR'])
        op_dir_opt = os.path.join(config['OPCODE'], config['DATA']['TESTING_DIR_OPT'])

        res_dir_path = os.path.join(config['RESULT'], config['REPORT']['PROJ_CO_CLONE'])

        filename = os.path.join(res_dir_path, 'doc2vec_proj_co-clone_report_' + uuid + '.csv')
        fields = ['project A', 'project B', 'similarity', 'same project', 'co-clone']
        rows = []

    for opcode in tqdm(os.listdir(op_dir)):
        for opcode_opt in tqdm(os.listdir(op_dir_opt)):
            target = os.path.join(op_dir, opcode)
            source = os.path.join(op_dir_opt, opcode_opt)
            tp = target.rsplit('/', 1)[-1]
            rp = source.rsplit('/', 1)[-1]
            if len(os.listdir(target)) > 0:
                if not test:
                    if tp in rp:
                        proj_sim = project_similarity(target, source, model)
                        pl_similarity_value.append(proj_sim)
                        rows.append([tp, rp, proj_sim])
                else:
                    proj_sim = project_similarity(target, source, model)
                    is_same_proj = False
                    if tp in rp:
                        is_same_proj = True
                    is_co_clone = proj_sim >= config['THRESHOLD']['DOC_2_VEC_PROJ']
                    rows.append([tp, rp, proj_sim, is_same_proj, is_co_clone])

    utils.generate_csv_report(filename, fields, rows)

    if not test:
        utils.createPDF(pl_similarity_value)


# This function is to loop through data in both testing and testing_optimized directory to compare the contract-level
# similarity between each other
def compute_contract_level_sim(config: dict, test: bool):
    cl_similarity_value = []
    uuid = datetime.now().strftime("%y%m%dT%H%M%S")
    model = torch.load(config['MODEL_DIR_DOC2VEC'] + 'doc2vec.pt')

    if not test:
        op_dir = os.path.join(config['OPCODE'], config['DATA']['TRAINING_DIR'])
        op_dir_opt = os.path.join(config['OPCODE'], config['DATA']['TRAINING_DIR_OPT'])

        res_dir_path = os.path.join(config['RESULT'], config['REPORT']['CONT_SIM'])

        filename = os.path.join(res_dir_path, 'doc2vec_cont_similarity_report_' + uuid + '.csv')
        fields = ['contract A', 'contract B', 'similarity']
        rows = []

    else:
        op_dir = os.path.join(config['OPCODE'], config['DATA']['TESTING_DIR'])
        op_dir_opt = os.path.join(config['OPCODE'], config['DATA']['TESTING_DIR_OPT'])

        res_dir_path = os.path.join(config['RESULT'], config['REPORT']['CONT_CO_CLONE'])

        filename = os.path.join(res_dir_path, 'doc2vec_cont_co-clone_report_' + uuid + '.csv')
        fields = ['contract A', 'contract B', 'similarity', 'same contract', 'co-clone']
        rows = []

    for proj in tqdm(os.listdir(op_dir)):
        for cont in tqdm(os.listdir(os.path.join(op_dir, proj))):
            target_cont = load_func(cont, os.path.join(op_dir, proj))
            tc_vec = model.infer_vector(target_cont[0])
            for proj_opt in tqdm(os.listdir(op_dir_opt)):
                for cont_opt in tqdm(os.listdir(os.path.join(op_dir_opt, proj_opt))):
                    repo_cont = load_func(cont_opt, os.path.join(op_dir_opt, proj_opt))
                    rc_vec = model.infer_vector(repo_cont[0])
                    tc = proj + '/' + cont
                    rc = proj_opt + '/' + cont_opt
                    if not test:
                        if cont in cont_opt:
                            sim = cosine_similarity(tc_vec, rc_vec)
                            cl_similarity_value.append(sim)
                            rows.append([tc, rc, sim])
                    else:
                        sim = cosine_similarity(tc_vec, rc_vec)
                        is_same_cont = False
                        if cont in cont_opt:
                            is_same_cont = True
                        is_co_clone = sim >= config['THRESHOLD']['DOC_2_VEC_CONT']
                        rows.append([tc, rc, sim, is_same_cont, is_co_clone])

    utils.generate_csv_report(filename, fields, rows)

    if not test:
        utils.createPDF(cl_similarity_value)
