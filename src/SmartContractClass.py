from config import *
#from typing import Dict
#import os

class  SmartContract:
    '''
    This class represents a smart contract
    '''
    def __init__(self, file_name, config: Dict):
        #decide the type of file_name

        #config['save_path']
        pass
    
    def log_message(self):
        pass
        
    def convert_binary_opcode(self):
        pass

    def convert_source_opcode(self):
        pass

    def convert_bin(self):
        proj_path = Dict["proj_path"]
        save_path = Dict["save_path"]
        bin_file = f'{proj_path}/{save_path}/{Dict["bin_file"]}'
        if os.path.isfile(bin_file) and os.access(bin_file, os.R_OK):
            os.system(f'rax2 -s < {bin_file} > {bin_file}.bin')
        else:
            print(f'{bin_file} not found')

    def save_opcode(self):
        pass


class DataBase:
    def __init__(self, dir_name, config):   
        #scan file in dir_name
        #SmartContract(file_name, config)
        pass

smc = SmartContract("./SolcExample/contract_out/Project.bin", Dict["bin_file"])
smc.convert_bin()
