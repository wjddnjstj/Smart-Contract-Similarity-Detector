import os
from smart_contract import SmartContract


class Database:

    def __init__(self, config: dict):
        self.config = config

    def scan_files(self):
        for project_name in os.listdir(self.config['ORIG_DIR']):
            sub_dir = os.path.join(self.config['ORIG_DIR'], project_name)
            if os.path.isfile(sub_dir):
                continue
            file_list = os.listdir(sub_dir)
            assert len(file_list) == 2
            source_file = os.path.join(sub_dir, project_name + '.sol')
            SmartContract(source_file, project_name, self.config)
