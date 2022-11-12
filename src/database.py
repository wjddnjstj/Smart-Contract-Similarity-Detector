import os
from smart_contract import SmartContract


class Database:

    def __init__(self, config: dict):
        self.config = config

    def scan_files(self, proj_dir):
        for project_name in os.listdir(proj_dir):
            sub_dir = os.path.join(proj_dir, project_name)
            if os.path.isfile(sub_dir):
                continue
            for file in os.listdir(sub_dir):
                source_file = os.path.join(sub_dir, file)
                if os.path.isfile(source_file) and file.endswith('.sol'):
                    source_file = os.path.join(sub_dir, file)
                    SmartContract(source_file, project_name, proj_dir, self.config)
