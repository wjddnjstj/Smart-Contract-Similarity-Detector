import os
import os.path
import json
import torch
import pytest
import main as main
import utils as utils
import smart_contract as smart_contract
from smart_contract import SmartContract
from database import *
import asm2vec_imp
import doc2vec_imp


@pytest.fixture
def config_dic():
    f_config = open('proj_tests/configT.json', 'r')
    config_dic = json.load(f_config)
    f_config.close()
    utils.clean_dataset(config_dic)
    main.prepare_directory(config_dic)
    return config_dic




@pytest.mark.order1
def test_bin2asm(config_dic):
    assert len(config_dic) == 16
    # remaining bin2asm functionality is tested under test_generate_inst()

@pytest.mark.order2
def test_prepare_directory(config_dic):
    asm_dir = config_dic['ASM']
    asm_testing_dir = os.path.join(asm_dir, config_dic["DATA"]["TESTING_DIR"])
    if os.path.exists(asm_testing_dir):
        os.rmdir(asm_testing_dir)
    main.prepare_directory(config_dic)
    assert len(os.listdir(asm_testing_dir)) == 0


@pytest.mark.order3
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


@pytest.mark.order4
def test_clean_dataset():
    #clean_dataset() already ran during fixture setup. Just testing outcome.
    assert len(os.listdir('proj_tests/testing_set')) > 0


@pytest.mark.order5
def test_create_pdf():
    result = utils.createPDF([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result is None


@pytest.mark.order6
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


@pytest.fixture()
def run_main(config_dic):
    training_set = config_dic['TRAINING_SET']
    testing_set = config_dic['TESTING_SET']
    db = Database(config_dic)
    db.scan_files(training_set)
    db.scan_files(testing_set)

    if config_dic['ASM_2_VEC']:
        utils.bin2asm(config_dic)

        asm_dir = config_dic['ASM']
        asm_training_dir = os.path.join(asm_dir, config_dic["DATA"]["TRAINING_DIR"])
        asm_training_dir_opt = os.path.join(asm_dir, config_dic["DATA"]["TRAINING_DIR_OPT"])
        for proj in os.listdir(asm_training_dir):
            asm2vec_imp.train(config_dic, os.path.join(asm_training_dir, proj))

        for proj in os.listdir(asm_training_dir_opt):
            asm2vec_imp.train(config_dic, os.path.join(asm_training_dir_opt, proj))

        asm_testing_dir = os.path.join(asm_dir, config_dic["DATA"]["TESTING_DIR"])
        for tp in os.listdir(asm_testing_dir):
            asm2vec_imp.test(config_dic, os.path.join(asm_testing_dir, tp))

        asm2vec_imp.compute_project_level_sim(config_dic, test=True)
        asm2vec_imp.compute_contract_level_sim(config_dic, test=True)
    else:
        doc2vec_imp.train_model(config_dic)
        doc2vec_imp.compute_project_level_sim(config_dic, test=True)
        doc2vec_imp.compute_contract_level_sim(config_dic, test=True)
    return run_main


@pytest.mark.order7
def test_main(run_main):
    f_config = open('proj_tests/configT.json', 'r')
    config = json.load(f_config)
    f_config.close()
    asm_dir = config['ASM']
    bin_dir = config['BIN']
    log_dir = config['LOG_DIR']
    opcode_dir = config['OPCODE']
    out_dir = config['OUT']
    result_dir = config['RESULT']
    training_dir = config['TRAINING_SET']
    testing_dir = config['TESTING_SET']
    model_dir = config['MODEL_DIR_DOC2VEC']
    assert os.path.exists(model_dir)
    for path in [asm_dir, bin_dir, log_dir, opcode_dir, out_dir, result_dir, training_dir, testing_dir]:
        cmd = "rm -fR " + path
        try:
            os.system(cmd)
        except:
            pass


@pytest.mark.skip(reason="other tests cover all methods used here")
def test_post_compilation():
    assert True


@pytest.mark.skip(reason="need to refactor compile_contract() to make it testable")
def test_compile_contract():
    assert True


@pytest.mark.order8
def test_remove_empty_files():
    data_dir = "proj_tests/__fixtures__/0x01c5f9163845ed9fe55e404831800b50edfcaa9e"
    smart_contract.SmartContract.remove_empty_files(data_dir)
    assert len(os.listdir(data_dir)) > 1


@pytest.mark.skip(reason="need to refactor format_opcode to make it testable")
def test_format_opcode():
    assert True


@pytest.mark.skip(reason="need to refactor log_message() to make it testable")
def test_log_message():
    assert True


@pytest.mark.order9
def test_save_opcode():
    source = "proj_tests/__fixtures__"
    target = "proj_tests/__fixtures__"
    SmartContract.save_opcode(source, target)
    assert len(os.listdir(target)) > 1


@pytest.mark.order10
def test_save_bin_code():
    source = "proj_tests/__fixtures__"
    target = "proj_tests/__fixtures__"
    SmartContract.save_bin_code(source, target)
    assert len(os.listdir(target)) > 1


@pytest.mark.order11
def test_preprocess():
    inst_path = "Address.opcode"
    data_dir = "proj_tests/__fixtures__/0x01c5f9163845ed9fe55e404831800b50edfcaa9e"
    assert(len(doc2vec_imp.preprocess(inst_path, data_dir)) > 0)


@pytest.mark.order12
def test_load_func():
    inst_path = "Address.opcode"
    data_dir = "proj_tests/__fixtures__/0x01c5f9163845ed9fe55e404831800b50edfcaa9e"
    assert (len(doc2vec_imp.load_func(inst_path, data_dir)) > 0)


@pytest.mark.order13
def test_load_data():
    data_dir = "proj_tests/__fixtures__/0x01c5f9163845ed9fe55e404831800b50edfcaa9e"
    assert (len(doc2vec_imp.load_data(data_dir)) > 0)


@pytest.mark.order14
def test_project_similarity():
    source = "proj_tests/__fixtures__/0x01c5f9163845ed9fe55e404831800b50edfcaa9e"
    target = "proj_tests/__fixtures__/0x01c5f9163845ed9fe55e404831800b50edfcaa9e"
    model = torch.load("proj_tests/__fixtures__/doc2vec.pt")
    print(doc2vec_imp.project_similarity(target, source, model))
    assert True


@pytest.mark.order15
def test_cosine_similarity_same():
    v1 = 0.9990400075912476
    v2 = 0.9990400075912476
    assert (doc2vec_imp.cosine_similarity(v1, v2) == 1.0)


@pytest.mark.order16
def test_cosine_similarity_not_same():
    v1 = 0.9990400075912476
    v2 = -0.1110400075912476
    assert (doc2vec_imp.cosine_similarity(v1, v2) != 0)


@pytest.mark.skip(reason="need to refactor compute_contract_level_sim() to make it testable")
def test_compute_project_level_sim():
    assert True


@pytest.mark.skip(reason="need to refactor compute_contract_level_sim() to make it testable")
def test_compute_contract_level_sim():
    assert True
