import json
import pytest
import torch
import src.main as main
import src.utils as utils
import src.doc2vec_imp as doc2vec_imp


@pytest.fixture
def config_dic():
    f_config = open('proj_tests/configT.json', 'r')
    config_dic = json.load(f_config)
    f_config.close()
    utils.clean_dataset(config_dic)
    main.prepare_directory(config_dic)
    return config_dic


def test_preprocess():
    inst_path = "Address.opcode"
    data_dir = "proj_tests/__fixtures__/0x01c5f9163845ed9fe55e404831800b50edfcaa9e"
    assert(len(doc2vec_imp.preprocess(inst_path, data_dir)) > 0)


def test_load_func():
    inst_path = "Address.opcode"
    data_dir = "proj_tests/__fixtures__/0x01c5f9163845ed9fe55e404831800b50edfcaa9e"
    assert (len(doc2vec_imp.load_func(inst_path, data_dir)) > 0)


def test_load_data():
    data_dir = "proj_tests/__fixtures__/0x01c5f9163845ed9fe55e404831800b50edfcaa9e"
    assert (len(doc2vec_imp.load_data(data_dir)) > 0)


def test_project_similarity():
    source = "proj_tests/__fixtures__/0x01c5f9163845ed9fe55e404831800b50edfcaa9e"
    target = "proj_tests/__fixtures__/0x01c5f9163845ed9fe55e404831800b50edfcaa9e"
    model = torch.load("proj_tests/__fixtures__/doc2vec.pt")
    print(doc2vec_imp.project_similarity(target, source, model))


def test_cosine_similarity_same():
    v1 = 0.9990400075912476
    v2 = 0.9990400075912476
    assert (doc2vec_imp.cosine_similarity(v1, v2) == 1.0)


def test_cosine_similarity_not_same():
    v1 = 0.9990400075912476
    v2 = -0.1110400075912476
    assert (doc2vec_imp.cosine_similarity(v1, v2) != 0)


@pytest.mark.skip(reason="need to refactor compute_contract_level_sim() to make it testable")
def test_compute_project_level_sim():
    assert True


@pytest.mark.skip(reason="need to refactor compute_project_level_sim() to make it testable")
def test_compute_contract_level_sim():
    assert True
