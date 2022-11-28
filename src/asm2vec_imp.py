import os
import torch
import asm2vec
import utils
from datetime import datetime
from tqdm import tqdm


def train(config, proj):
    if config['ASM_CONFIG']['DEVICE'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mpath = config['MODEL_DIR_ASM2VEC'] + 'asm2vec.pt'
    if os.path.exists(mpath):
        model, tokens = asm2vec.utils.load_model(mpath, device=device)
        functions, tokens_new = asm2vec.utils.load_data(proj, limit=config['ASM_CONFIG']['LIMIT'])
        tokens.update(tokens_new)
        model.update(len(functions), tokens.size())
    else:
        model = None
        functions, tokens = asm2vec.utils.load_data(proj, limit=config['ASM_CONFIG']['LIMIT'])

    opath = config['MODEL_DIR_ASM2VEC']
    if not os.path.isdir(opath):
        os.mkdir(opath)

    def callback(context):
        progress = f'{context["epoch"]} | time = {context["time"]:.2f}, loss = {context["loss"]:.4f}'
        if context["accuracy"]:
            progress += f', accuracy = {context["accuracy"]:.4f}'
        asm2vec.utils.save_model(opath + 'asm2vec.pt', context["model_doc2vec"], context["tokens"])

    model = asm2vec.utils.train(
        functions,
        tokens,
        model=model,
        embedding_size=config['ASM_CONFIG']['EMBEDDING_SIZE'],
        batch_size=config['ASM_CONFIG']['BATCH_SIZE'],
        epochs=config['ASM_CONFIG']['EPOCHS'],
        neg_sample_num=config['ASM_CONFIG']['NEG_SAMPLE_NUM'],
        calc_acc=config['ASM_CONFIG']['CALC_ACC'],
        device=device,
        callback=callback,
        learning_rate=config['ASM_CONFIG']['LEARNING_RATE']
    )


def test(config, target_proj):
    print('====================================================')
    tp = target_proj.rsplit('/', 1)[-1]
    print('target project: ', tp)
    if config['ASM_CONFIG']['DEVICE'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load asm2vec model, tokens
    for tc in os.listdir(target_proj):
        model, tokens = asm2vec.utils.load_model(config['MODEL_DIR_ASM2VEC'] + 'asm2vec.pt', device=device)
        functions, tokens_new = asm2vec.utils.load_data(os.path.join(target_proj, tc))
        tokens.update(tokens_new)
        model.update(1, tokens.size())
        model = model.to(device)

        # train function embedding
        model = asm2vec.utils.train(
            functions,
            tokens,
            model=model,
            epochs=config['ASM_CONFIG']['EPOCHS'],
            neg_sample_num=config['ASM_CONFIG']['NEG_SAMPLE_NUM'],
            device=device,
            mode='test',
            learning_rate=config['ASM_CONFIG']['LEARNING_RATE']
        )

        # show predicted probability results
        x, y = asm2vec.utils.preprocess(functions, tokens)
        probs = model.predict(x.to(device), y.to(device))
        print('target contract: ', tc)
        asm2vec.utils.show_probs(x, y, probs, tokens, limit=config['ASM_CONFIG']['LIMIT'], pretty=True)
    print('====================================================')


# This function is to compute the actual similarity value between any two projects
def project_similarity(target, source, config):
    if config['ASM_CONFIG']['DEVICE'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokens = asm2vec.utils.load_model(config['MODEL_DIR_ASM2VEC'] + 'asm2vec.pt', device=device)
    total_contract_sim = 0
    for tc in os.listdir(target):
        max_contract_sim = 0
        for rc in os.listdir(source):
            functions, tokens_new = asm2vec.utils.load_data([os.path.join(target, tc), os.path.join(source, rc)])
            tokens.update(tokens_new)
            model.update(2, tokens.size())
            model = model.to(device)

            # train function embedding
            model = asm2vec.utils.train(
                functions,
                tokens,
                model=model,
                epochs=config['ASM_CONFIG']['EPOCHS'],
                device=device,
                mode='test',
                learning_rate=config['ASM_CONFIG']['LEARNING_RATE']
            )

            # compare 2 function vectors
            v1, v2 = model.to('cpu').embeddings_f(torch.tensor([0, 1]))
            sim = cosine_similarity(v1, v2)
            max_contract_sim = max(max_contract_sim, sim)
        total_contract_sim += max_contract_sim
    return total_contract_sim / len(os.listdir(target))


def cosine_similarity(v1, v2):
    return v1 @ v2 / (v1.norm() * v2.norm()).item()


# This function is to loop through data in both testing and testing_optimized directory to compare the project-level
# similarity between each other
def compute_project_level_sim(config: dict, test: bool):
    pl_similarity_value = []
    uuid = datetime.now().strftime("%y%m%dT%H%M%S")

    if not test:
        asm_dir = os.path.join(config['ASM'], config['DATA']['TRAINING_DIR'])
        asm_dir_opt = os.path.join(config['ASM'], config['DATA']['TRAINING_DIR_OPT'])

        res_dir_path = os.path.join(config['RESULT'], config['REPORT']['PROJ_SIM'])

        filename = os.path.join(res_dir_path, 'asm2vec_proj_similarity_report_' + uuid + '.csv')
        fields = ['project A', 'project B', 'similarity']
        rows = []

    else:
        asm_dir = os.path.join(config['ASM'], config['DATA']['TESTING_DIR'])
        asm_dir_opt = os.path.join(config['ASM'], config['DATA']['TESTING_DIR_OPT'])

        res_dir_path = os.path.join(config['RESULT'], config['REPORT']['PROJ_CO_CLONE'])

        filename = os.path.join(res_dir_path, 'asm2vec_proj_co-clone_report_' + uuid + '.csv')
        fields = ['project A', 'project B', 'similarity', 'same project', 'co-clone']
        rows = []

    for asm in tqdm(os.listdir(asm_dir)):
        for asm_opt in tqdm(os.listdir(asm_dir_opt)):
            target = os.path.join(asm_dir, asm)
            source = os.path.join(asm_dir_opt, asm_opt)
            tp = target.rsplit('/', 1)[-1]
            rp = source.rsplit('/', 1)[-1]
            if len(os.listdir(target)) > 0:
                if not test:
                    if tp in rp:
                        proj_sim = project_similarity(target, source, config)
                        pl_similarity_value.append(proj_sim)
                        rows.append([tp, rp, proj_sim])
                else:
                    proj_sim = project_similarity(target, source, config)
                    is_same_proj = False
                    if tp in rp:
                        is_same_proj = True
                    is_co_clone = proj_sim >= config['THRESHOLD']['ASM_2_VEC_PROJ']
                    rows.append([tp, rp, proj_sim, is_same_proj, is_co_clone])

    utils.generate_csv_report(filename, fields, rows)

    if not test:
        with torch.no_grad():
            utils.createPDF(pl_similarity_value)


# This function is to loop through data in both testing and testing_optimized directory to compare the contract-level
# similarity between each other
def compute_contract_level_sim(config: dict, test: bool):
    cl_similarity_value = []
    uuid = datetime.now().strftime("%y%m%dT%H%M%S")

    if not test:
        asm_dir = os.path.join(config['ASM'], config['DATA']['TRAINING_DIR'])
        asm_dir_opt = os.path.join(config['ASM'], config['DATA']['TRAINING_DIR_OPT'])

        res_dir_path = os.path.join(config['RESULT'], config['REPORT']['CONT_SIM'])

        filename = os.path.join(res_dir_path, 'asm2vec_cont_similarity_report_' + uuid + '.csv')
        fields = ['contract A', 'contract B', 'similarity']
        rows = []

    else:
        asm_dir = os.path.join(config['ASM'], config['DATA']['TESTING_DIR'])
        asm_dir_opt = os.path.join(config['ASM'], config['DATA']['TESTING_DIR_OPT'])

        res_dir_path = os.path.join(config['RESULT'], config['REPORT']['CONT_CO_CLONE'])

        filename = os.path.join(res_dir_path, 'asm2vec_cont_co-clone_report_' + uuid + '.csv')
        fields = ['contract A', 'contract B', 'similarity', 'same contract', 'co-clone']
        rows = []

    if config['ASM_CONFIG']['DEVICE'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokens = asm2vec.utils.load_model(config['MODEL_DIR_ASM2VEC'] + 'asm2vec.pt', device=device)

    for proj in tqdm(os.listdir(asm_dir)):
        for cont in tqdm(os.listdir(os.path.join(asm_dir, proj))):
            for proj_opt in tqdm(os.listdir(asm_dir_opt)):
                for cont_opt in tqdm(os.listdir(os.path.join(asm_dir_opt, proj_opt))):
                    functions, tokens_new = asm2vec.utils.load_data(
                        [os.path.join(os.path.join(asm_dir, proj), cont),
                         os.path.join(os.path.join(asm_dir_opt, proj_opt), cont_opt)])
                    tokens.update(tokens_new)
                    model.update(2, tokens.size())
                    model = model.to(device)

                    # train function embedding
                    model = asm2vec.utils.train(
                        functions,
                        tokens,
                        model=model,
                        epochs=config['ASM_CONFIG']['EPOCHS'],
                        device=device,
                        mode='test',
                        learning_rate=config['ASM_CONFIG']['LEARNING_RATE']
                    )

                    # compare 2 function vectors
                    v1, v2 = model.to('cpu').embeddings_f(torch.tensor([0, 1]))
                    tc = proj + '/' + cont
                    rc = proj_opt + '/' + cont_opt

                    if not test:
                        if cont in cont_opt:
                            cl_similarity_value.append(sim)
                            sim = cosine_similarity(v1, v2)
                            rows.append([tc, rc, sim])
                    else:
                        sim = cosine_similarity(v1, v2)
                        is_same_cont = False
                        if cont in cont_opt:
                            is_same_cont = True
                        is_co_clone = sim >= config['THRESHOLD']['ASM_2_VEC_CONT']
                        rows.append([tc, rc, sim, is_same_cont, is_co_clone])

    utils.generate_csv_report(filename, fields, rows)

    if not test:
        with torch.no_grad():
            utils.createPDF(cl_similarity_value)


def match_max_sim_proj(config):
    uuid = datetime.now().strftime("%y%m%dT%H%M%S")
    asm_testing_dir = os.path.join(config['ASM'], config['DATA']['TESTING_DIR'])
    asm_testing_dir_opt = os.path.join(config['ASM'], config['DATA']['TESTING_DIR_OPT'])

    res_dir_path = os.path.join(config['RESULT'], config['REPORT']['PROJ_MAX_SIM'])
    filename = os.path.join(res_dir_path, 'asm2vec_proj_max_sim_report_' + uuid + '.csv')
    fields = ['project', 'max_sim project', 'similarity', 'same project']
    rows = []

    for asm in tqdm(os.listdir(asm_testing_dir)):
        max_proj_sim = 0
        max_proj_name = ""
        is_same_cont = False
        for asm_opt in tqdm(os.listdir(asm_testing_dir_opt)):
            target = os.path.join(asm_testing_dir, asm)
            source = os.path.join(asm_testing_dir_opt, asm_opt)
            if len(os.listdir(target)) > 0:
                proj_sim = project_similarity(target, source, config)
                if proj_sim > max_proj_sim:
                    max_proj_sim = proj_sim
                    max_proj_name = source.rsplit('/', 1)[-1]
        if target.rsplit('/', 1)[-1] in max_proj_name:
            is_same_cont = True
        if len(os.listdir(target)) > 0:
            rows.append([target.rsplit('/', 1)[-1], max_proj_name, max_proj_sim, is_same_cont])

    utils.generate_csv_report(filename, fields, rows)


def match_max_sim_contract(config: dict):
    uuid = datetime.now().strftime("%y%m%dT%H%M%S")
    asm_testing_dir = os.path.join(config['ASM'], config['DATA']['TESTING_DIR'])
    asm_testing_dir_opt = os.path.join(config['ASM'], config['DATA']['TESTING_DIR_OPT'])

    res_dir_path = os.path.join(config['RESULT'], config['REPORT']['CONT_MAX_SIM'])
    filename = os.path.join(res_dir_path, 'asm2vec_cont_max_sim_report_' + uuid + '.csv')
    fields = ['contract A', 'max_sim contract', 'similarity', 'same contract']
    rows = []

    if config['ASM_CONFIG']['DEVICE'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokens = asm2vec.utils.load_model(config['MODEL_DIR_ASM2VEC'] + 'asm2vec.pt', device=device)

    for proj in tqdm(os.listdir(asm_testing_dir)):
        for cont in tqdm(os.listdir(os.path.join(asm_testing_dir, proj))):
            max_cont_sim = 0
            max_cont_name = ""
            is_same_cont = False
            for proj_opt in tqdm(os.listdir(asm_testing_dir_opt)):
                for cont_opt in tqdm(os.listdir(os.path.join(asm_testing_dir_opt, proj_opt))):
                    functions, tokens_new = asm2vec.utils.load_data(
                        [os.path.join(os.path.join(asm_testing_dir, proj), cont), os.path.join(os.path.join(asm_testing_dir_opt, proj_opt), cont_opt)])
                    tokens.update(tokens_new)
                    model.update(2, tokens.size())
                    model = model.to(device)

                    # train function embedding
                    model = asm2vec.utils.train(
                        functions,
                        tokens,
                        model=model,
                        epochs=config['ASM_CONFIG']['EPOCHS'],
                        device=device,
                        mode='test',
                        learning_rate=config['ASM_CONFIG']['LEARNING_RATE']
                    )

                    # compare 2 function vectors
                    v1, v2 = model.to('cpu').embeddings_f(torch.tensor([0, 1]))
                    sim = cosine_similarity(v1, v2)

                    if sim > max_cont_sim:
                        max_cont_sim = sim
                        max_cont_name = proj_opt + '/' + cont_opt
            if cont in max_cont_name:
                is_same_cont = True
            rows.append([proj + '/' + cont, max_cont_name, max_cont_sim, is_same_cont])

    utils.generate_csv_report(filename, fields, rows)
