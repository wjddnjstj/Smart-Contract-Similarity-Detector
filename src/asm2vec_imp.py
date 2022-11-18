import os
import torch
import asm2vec


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


def compare_contract_sim(config: dict):
    asm_testing_dir = os.path.join(config['ASM'], config['DATA']['TESTING_DIR'])
    asm_testing_dir_opt = os.path.join(config['ASM'], config['DATA']['TESTING_DIR_OPT'])
    # asm_testing_dir_opt = os.path.join(config['ASM'], config['DATA']['TESTING_DIR_SELF'])
    # asm_testing_dir_opt = os.path.join(config['ASM'], config['DATA']['TESTING_DIR_MOD'])
    correctness = 0
    total = 0

    if config['ASM_CONFIG']['DEVICE'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokens = asm2vec.utils.load_model(config['MODEL_DIR_ASM2VEC'] + 'asm2vec.pt', device=device)

    for proj in os.listdir(asm_testing_dir):
        for cont in os.listdir(os.path.join(asm_testing_dir, proj)):
            max_cont_sim = 0
            max_cont_name = ""
            total = total + 1
            for proj_opt in os.listdir(asm_testing_dir_opt):
                for cont_opt in os.listdir(os.path.join(asm_testing_dir_opt, proj_opt)):
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

            print('*****************************************')
            print('Most similar contract of ' + proj + '/' + cont + ' is ' + max_cont_name)
            print('The similarity is {}'.format(max_cont_sim))
            print('*****************************************')

            if proj in max_cont_name and cont in max_cont_name:
                correctness = correctness + 1
    print('Accuracy of smart contract similarity detection is {}'.format(correctness / total))


def compare_sim(config):
    asm_testing_dir = os.path.join(config['ASM'], config['DATA']['TESTING_DIR'])
    asm_testing_dir_opt = os.path.join(config['ASM'], config['DATA']['TESTING_DIR_OPT'])
    # asm_testing_dir_opt = os.path.join(config['ASM'], config['DATA']['TESTING_DIR_SELF'])
    # asm_testing_dir_opt = os.path.join(config['ASM'], config['DATA']['TESTING_DIR_MOD'])
    correctness = 0
    for asm in os.listdir(asm_testing_dir):
        max_proj_sim = 0
        max_proj_name = ""
        for asm_opt in os.listdir(asm_testing_dir_opt):
            target = os.path.join(asm_testing_dir, asm)
            source = os.path.join(asm_testing_dir_opt, asm_opt)
            if len(os.listdir(target)) > 0:
                proj_sim = compute_proj_sim(target, source, config)
                if proj_sim > max_proj_sim:
                    max_proj_sim = proj_sim
                    max_proj_name = source.rsplit('/', 1)[-1]
        if len(os.listdir(target)) > 0:
            print('*****************************************')
            print('Most similar project of ' + target.rsplit('/', 1)[-1] + ' is ' + max_proj_name)
            print('The similarity is {}'.format(proj_sim))
            print('*****************************************')

        if target.rsplit('/', 1)[-1] in max_proj_name:
            correctness = correctness + 1
    print('Accuracy of smart contract similarity detection is {}'.format(correctness / len(os.listdir(asm_testing_dir))))


def compute_proj_sim(target, source, config):
    if config['ASM_CONFIG']['DEVICE'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokens = asm2vec.utils.load_model(config['MODEL_DIR_ASM2VEC'] + 'asm2vec.pt', device=device)

    # print('====================================================')
    total_contract_sim = 0
    # tp = target.rsplit('/', 1)[-1]
    # rp = source.rsplit('/', 1)[-1]
    # print('target project: ', tp)
    # print('repo project: ', rp)
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
            # print('sim("{}", "{}") = {}'.format(tc, rc, sim))
        total_contract_sim += max_contract_sim
    # print('sim("{}", "{}") = {}'.format(tp, rp, total_contract_sim / len(os.listdir(target))))
    # print('====================================================')
    return total_contract_sim / len(os.listdir(target))


def cosine_similarity(v1, v2):
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()
