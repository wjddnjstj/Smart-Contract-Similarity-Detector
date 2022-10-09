from typing import Dict
import os

class  SmartContract:
    '''
    This class represents a smart contract
    '''
    def __init__(self, file_name, config: Dict):
        decide the type of file_name

        config['save_path']
    
    def log_message(self):
        

    def convert_binary_opcode(self):
        XXX

    def convert_source_opcode(self):
        XXX

    def convert_bin(self):
	bin_file = Dict["bin_file"]
        if os.path.isfile(bin_file) and os.access(bin_file, os.R_OK):
            os.system(f'rax2 -s < {bin_file} > {bin_file}.bin')
        else:
            print(f'{bin_file} not found')

    def save_opcode(self):


class DataBase:
    def __init__(self, dir_name, config):   
        scan file in dir_name
        SmartContract(file_name, config)
