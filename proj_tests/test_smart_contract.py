import json
import pytest
import src.main as main
import src.utils as utils
import src.smart_contract as smart_contract
from smart_contract import SmartContract


@pytest.fixture
def config_dic():
    f_config = open('proj_tests/configT.json', 'r')
    config_dic = json.load(f_config)
    f_config.close()
    utils.clean_dataset(config_dic)
    main.prepare_directory(config_dic)
    return config_dic

@pytest.mark.skip(reason="other tests cover all methods used here")
def test_post_compilation():
    assert True

@pytest.mark.skip(reason="need to refactor compile_contract() to make it testable")
def test_compile_contract():
    assert True

def test_remove_empty_files():
    data_dir = "proj_tests/__fixtures__/0x01c5f9163845ed9fe55e404831800b50edfcaa9e"
    smart_contract.SmartContract.remove_empty_files(data_dir)

@pytest.mark.skip(reason="need to refactor format_opcode to make it testable")
def test_format_opcode():
    assert True

@pytest.mark.skip(reason="need to refactor log_message() to make it testable")
def test_log_message():
    assert True

def test_save_opcode():
    source = "proj_tests/__fixtures__"
    target = "proj_tests/__fixtures__"
    SmartContract.save_opcode(source, target)

def test_save_bin_code():
    source = "proj_tests/__fixtures__"
    target = "proj_tests/__fixtures__"
    SmartContract.save_bin_code(source, target)