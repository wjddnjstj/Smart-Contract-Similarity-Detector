import os
import torch
import asm2vec


def train(config):
    if config['ASM_CONFIG']['DEVICE'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = None
    functions, tokens = asm2vec.utils.load_data(config['ASM_PROJ_DIR'], limit=config['ASM_CONFIG']['LIMIT'])

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


def test(config):
    if config['ASM_CONFIG']['DEVICE'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load asm2vec model, tokens
    target_dir = config['ASM_CONFIG']['TARGET_PROJ_DIR']
    for tp in os.listdir(target_dir):
        print('====================================================')
        model, tokens = asm2vec.utils.load_model(config['MODEL_DIR_ASM2VEC'] + 'asm2vec.pt', device=device)
        functions, tokens_new = asm2vec.utils.load_data(os.path.join(target_dir, tp))
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
        print('target project: ', tp)
        asm2vec.utils.show_probs(x, y, probs, tokens, limit=config['ASM_CONFIG']['LIMIT'], pretty=True)
        print('====================================================')


def compare_sim(config):
    if config['ASM_CONFIG']['DEVICE'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    repo_dir = config['ASM_PROJ_DIR']
    target_dir = config['ASM_CONFIG']['TARGET_PROJ_DIR']
    for tp in os.listdir(target_dir):
        for rp in os.listdir(repo_dir):
            print('====================================================')
            model, tokens = asm2vec.utils.load_model(config['MODEL_DIR_ASM2VEC'] + 'asm2vec.pt', device=device)
            functions, tokens_new = asm2vec.utils.load_data([os.path.join(target_dir, tp), os.path.join(repo_dir, rp)])
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

            print('target project: ', tp)
            print('repo project: ', rp)
            print(f'cosine similarity : {cosine_similarity(v1, v2):.6f}')
            print('====================================================')


def cosine_similarity(v1, v2):
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()


