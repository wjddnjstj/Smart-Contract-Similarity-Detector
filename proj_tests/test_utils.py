import os
import json
import pytest
import src.utils as utils
import coverage


def test_bin2asm():
    f_config = open('src/config.json', 'r')
    config_dic = json.load(f_config)
    f_config.close()

@pytest.mark.skip(reason="non-essential helper function")
def test_create_pdf():
    pass


def test_generate_csv_report():
    tmppath = "proj_tests/temp_dir"
    csvfile = os.path.join(tmppath,"temp_csv_file")
    cmd = "rm " + csvfile
    if os.path.exists(tmppath):
        if os.path.exists(csvfile):
            os.system(cmd)
    else:
        os.makedirs(tmppath)
    utils.generate_csv_report(csvfile, ["project1", "project2", "similarity"], [])
    assert os.path.exists(csvfile)
    os.system(cmd)
