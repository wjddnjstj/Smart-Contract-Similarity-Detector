import os
import json
import pytest
import src.main as main
import src.utils as utils


@pytest.fixture
def config_dic():
    f_config = open('proj_tests/configT.json', 'r')
    config_dic = json.load(f_config)
    f_config.close()
    utils.clean_dataset(config_dic)
    main.prepare_directory(config_dic)
    return config_dic


# remaining bin2asm functionality is tested under test_generate_inst()
def test_bin2asm(config_dic):
    assert len(config_dic) == 16


def test_generate_inst(config_dic):
    bin_dir = config_dic['BIN']
    bin_training_dir = os.path.join(bin_dir, config_dic["DATA"]["TRAINING_DIR"])
    bin_testing_dir = os.path.join(bin_dir, config_dic["DATA"]["TESTING_DIR"])
    asm_dir = config_dic['ASM']
    asm_training_dir = os.path.join(asm_dir, config_dic["DATA"]["TRAINING_DIR"])
    asm_testing_dir = os.path.join(asm_dir, config_dic["DATA"]["TESTING_DIR"])
    utils.generate_inst(bin_training_dir, asm_training_dir)
    utils.generate_inst(bin_testing_dir, asm_testing_dir)
    assert len(os.listdir(asm_testing_dir)) == 0


# clean_dataset() already ran during fixture setup. Just testing outcome.
def test_clean_dataset():
    assert len(os.listdir('proj_tests/testing_set')) > 0


def test_create_pdf():
    result = utils.createPDF([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result is None


def test_generate_csv_report():
    tmppath = "proj_tests/temp_dir"
    csvfile = os.path.join(tmppath, "temp_csv_file")
    cmd = "rm " + csvfile
    if os.path.exists(tmppath):
        if os.path.exists(csvfile):
            os.system(cmd)
    else:
        os.makedirs(tmppath)
    utils.generate_csv_report(csvfile, ["project1", "project2", "similarity"], [])
    assert os.path.exists(csvfile)
    os.system(cmd)
