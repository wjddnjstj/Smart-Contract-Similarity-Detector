import json
import src.main as main
import src.utils as utils
import pytest


@pytest.fixture()
def setUp(self):
    print("setup")
    yield "resource"
    print("teardown")


@pytest.fixture
def config_dic():
    f_config = open('proj_tests/configT.json', 'r')
    config_dic = json.load(f_config)
    f_config.close()
    utils.clean_dataset(config_dic)
    main.prepare_directory(config_dic)
    return config_dic


@pytest.mark.skip(reason="need to refactor compute_contract_level_sim() to make it testable")
def test_scan_files():
    assert True
