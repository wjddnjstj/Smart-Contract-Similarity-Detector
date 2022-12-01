import json
import pytest
import main
import utils

def test_main_for_doc2vec()
def test_main_for_asm2vec()

@pytest.fixture()
def config_dic():
    f_config = open('proj_tests/configT.json', 'r')
    config_dic = json.load(f_config)
    f_config.close()
    utils.clean_dataset(config_dic)
    return config_dic


def test_prepare_directory(config_dic):
    main.prepare_directory(config_dic)
    assert True

@pytest.mark.skip(reason="other tests cover all methods used here")
def test_main():
    assert True
