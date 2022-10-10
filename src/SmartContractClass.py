from config import *
#from typing import Dict
import os

OPCODE_SET = {"STOP", "ADD", "MUL", "SUB", "DIV", "SDIV", "MOD", "SMOD", "ADDMOD", "MULMOD", "EXP", "SIGNEXTEND", "LT", "GT", "SLT", "SGT", "EQ", "ISZERO", "AND", "OR", "EVMOR", "XOR", "NOT", "BYTE", "SHL", "SHR", "SAR", "SHA3", "ADDRESS", "BALANCE", "ORIGIN", "CALLER", "CALLVALUE", "CALLDATALOAD", "CALLDATASIZE", "CALLDATACOPY", "CODESIZE", "CODECOPY", "GASPRICE", "EXTCODESIZE", "EXTCODECOPY", "RETURNDATASIZE", "RETURNDATACOPY", "EXTCODEHASH", "BLOCKHASH", "COINBASE", "TIMESTAMP", "NUMBER", "DIFFICULTY", "GASLIMIT", "POP", "MLOAD", "MSTORE", "MSTORE8", "SLOAD", "SSTORE", "JUMP", "JUMPI", "PC", "MSIZE", "GAS", "JUMPDEST", "PUSH1", "PUSH2", "PUSH3", "PUSH4", "PUSH5", "PUSH6", "PUSH7", "PUSH8", "PUSH9", "PUSH10", "PUSH11", "PUSH12", "PUSH13", "PUSH14", "PUSH15", "PUSH16", "PUSH17", "PUSH18", "PUSH19", "PUSH20", "PUSH21", "PUSH22", "PUSH23", "PUSH24", "PUSH25", "PUSH26", "PUSH27", "PUSH28", "PUSH29", "PUSH30", "PUSH31", "PUSH32", "DUP1", "DUP2", "DUP3", "DUP4", "DUP5", "DUP6", "DUP7", "DUP8", "DUP9", "DUP10", "DUP11", "DUP12", "DUP13", "DUP14", "DUP15", "DUP16", "SWAP1", "SWAP2", "SWAP3", "SWAP4", "SWAP5", "SWAP6", "SWAP7", "SWAP8", "SWAP9", "SWAP10", "SWAP11", "SWAP12", "SWAP13", "SWAP14", "SWAP15", "SWAP16", "LOG0", "LOG1", "LOG2", "LOG3", "LOG4", "CREATE", "CALL", "CALLCODE", "RETURN", "DELEGATECALL", "CREATE2", "STATICCALL", "REVERT", "INVALID", "SELFDESTRUCT", "SUICIDE", "SELFBALANCE"}

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
    
    def format_opcode(self, dir_name):
        for opcode_filename in os.listdir(dir_name):
        if opcode_filename.endswith("opcode"):
            fullPath = os.path.join(dir_name, opcode_filename)
            f = open(fullPath, 'r')
            for line in f:
                formatted_opcode = open(fullPath, 'w')
                hasfirstline = False
                for opcode in line.split():
                    if bool(hasfirstline): # line 1 has been written
                        if opcode in OPCODE_SET:
                            formatted_opcode.write('\n' + opcode + ' ')
                        else:
                            formatted_opcode.write(opcode + ' ')
                    else: # No line has been written yet
                        formatted_opcode.write(opcode + ' ')
                        hasfirstline = True

    def save_opcode(self):
        pass


class DataBase:
    def __init__(self, dir_name, config):   
        #scan file in dir_name
        #SmartContract(file_name, config)
        pass

smc = SmartContract("./SolcExample/contract_out/Project.bin", Dict["bin_file"])
smc.convert_bin()
