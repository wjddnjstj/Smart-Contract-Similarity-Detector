import os, re
from pathlib import Path
from typing import Optional
import solcx
from solcx.install import get_executable
from solcx.install import install_solc_pragma
import subprocess
import csv
import json
from random import sample
import gensim
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models.doc2vec import TaggedDocument
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm


class SmartContract:
    opcode_set = {"STOP", "ADD", "MUL", "SUB", "DIV", "SDIV", "MOD", "SMOD", "ADDMOD", "MULMOD", "EXP", "SIGNEXTEND",
                  "LT",
                  "GT", "SLT", "SGT", "EQ", "ISZERO", "AND", "OR", "EVMOR", "XOR", "NOT", "BYTE", "SHL", "SHR", "SAR",
                  "SHA3", "ADDRESS", "BALANCE", "ORIGIN", "CALLER", "CALLVALUE", "CALLDATALOAD", "CALLDATASIZE",
                  "CALLDATACOPY", "CODESIZE", "CODECOPY", "GASPRICE", "EXTCODESIZE", "EXTCODECOPY", "RETURNDATASIZE",
                  "RETURNDATACOPY", "EXTCODEHASH", "BLOCKHASH", "COINBASE", "TIMESTAMP", "NUMBER", "DIFFICULTY",
                  "GASLIMIT",
                  "POP", "MLOAD", "MSTORE", "MSTORE8", "SLOAD", "SSTORE", "JUMP", "JUMPI", "PC", "MSIZE", "GAS",
                  "JUMPDEST",
                  "PUSH1", "PUSH2", "PUSH3", "PUSH4", "PUSH5", "PUSH6", "PUSH7", "PUSH8", "PUSH9", "PUSH10", "PUSH11",
                  "PUSH12", "PUSH13", "PUSH14", "PUSH15", "PUSH16", "PUSH17", "PUSH18", "PUSH19", "PUSH20", "PUSH21",
                  "PUSH22", "PUSH23", "PUSH24", "PUSH25", "PUSH26", "PUSH27", "PUSH28", "PUSH29", "PUSH30", "PUSH31",
                  "PUSH32", "DUP1", "DUP2", "DUP3", "DUP4", "DUP5", "DUP6", "DUP7", "DUP8", "DUP9", "DUP10", "DUP11",
                  "DUP12", "DUP13", "DUP14", "DUP15", "DUP16", "SWAP1", "SWAP2", "SWAP3", "SWAP4", "SWAP5", "SWAP6",
                  "SWAP7", "SWAP8", "SWAP9", "SWAP10", "SWAP11", "SWAP12", "SWAP13", "SWAP14", "SWAP15", "SWAP16",
                  "LOG0",
                  "LOG1", "LOG2", "LOG3", "LOG4", "CREATE", "CALL", "CALLCODE", "RETURN", "DELEGATECALL", "CREATE2",
                  "STATICCALL", "REVERT", "INVALID", "SELFDESTRUCT", "SUICIDE", "SELFBALANCE"}

    def __init__(self, file_name: str, proj_name: str, config: dict):
        self.file_name = file_name
        self.proj_name = proj_name
        self.config = config

        self.convert_source_opcode(True)
        # decide the type of file_name

    def remove_void(self, line):
        while m := re.compile('//|/\*|"|\'').search(line):
            if m[0] == '//':
                return (line[:m.start()], False)
            if m[0] == '/*':
                end = line.find('*/', m.end())
                if end == -1:
                    return (line[:m.start()], True)
                else:
                    line = line[:m.start()] + line[end + 2:]
                    continue
            if m[0] == '"':
                m2 = re.compile('(?<!\\\\)"').search(line[m.end():])
            else:  # m[0] == "'":
                m2 = re.compile("(?<!\\\\)'").search(line[m.end():])
            if m2:
                line = line[:m.start()] + line[m.end() + m2.end():]
                continue
            # we should not arrive here for a correct Solidity program
            return line[:m.start()], False
        return line, False

    def get_pragma(self, file: str) -> Optional[str]:
        in_comment = False
        for line in file.splitlines():
            if in_comment:
                end = line.find('*/')
                if end == -1:
                    continue
                else:
                    line = line[end + 2:]
            line, in_comment = self.remove_void(line)
            if m := re.compile('pragma solidity.*?;').search(line):
                return m[0]
        return None

    def get_solc(self, filename: str) -> Optional[Path]:
        with open(filename) as f:
            file = f.read()
        try:
            pragma = self.get_pragma(file)
            pragma = re.sub(r">=0\.", r"^0.", pragma)
            version = solcx.install_solc_pragma(pragma)
            return solcx.get_executable(version)
        except:
            return None

    def get_solc(self, filename: str) -> Optional[Path]:
        with open(filename) as f:
            file = f.read()
        try:
            pragma = self.get_pragma(file)
            pragma = re.sub(r">=0\.", r"^0.", pragma)
            version = install_solc_pragma(pragma)
            return get_executable(version)
        except:
            return None

    def convert_source_opcode(self, is_optimized: bool):
        if not os.path.isdir(self.config['COMPILED_DIR']):
            os.mkdir(self.config['COMPILED_DIR'])

        if not os.path.isdir(self.config['COMPILED_DIR_OPT']):
            os.mkdir(self.config['COMPILED_DIR_OPT'])

        if not os.path.isdir(self.config['LOG_DIR']):
            os.mkdir(self.config['LOG_DIR'])

        solc_compiler = self.get_solc(self.file_name)
        if is_optimized:
            save_dir = os.path.join(self.config['COMPILED_DIR_OPT'], self.proj_name)
            prefix = '--overwrite --opcodes --bin --bin-runtime --abi --asm-json --optimize'
        else:
            save_dir = os.path.join(self.config['COMPILED_DIR'], self.proj_name)
            prefix = '--overwrite --opcodes --bin --bin-runtime --abi --asm-json'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        cmd = '%s %s -o %s %s' % (solc_compiler, prefix, save_dir, self.file_name)

        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.wait()

        self.format_opcode(save_dir)
        self.log_message(process, save_dir)

    def format_opcode(self, dir_name):
        for opcode_filename in os.listdir(dir_name):
            if opcode_filename.endswith("opcode"):
                full_path = os.path.join(dir_name, opcode_filename)
                f = open(full_path, 'r')
                for line in f:
                    formatted_opcode = open(full_path, 'w')
                    has_first_line = False
                    for opcode in line.split():
                        if bool(has_first_line):  # line 1 has been written
                            if opcode in self.opcode_set:
                                formatted_opcode.write('\n' + opcode + ' ')
                            else:
                                formatted_opcode.write(opcode + ' ')
                        else:  # No line has been written yet
                            formatted_opcode.write(opcode + ' ')
                            has_first_line = True

    def log_message(self, process, save_dir):
        log_dir_success = os.path.join(self.config['LOG_DIR'], self.proj_name + "_out.csv")
        log_dir_error = os.path.join(self.config['LOG_DIR'], self.proj_name + "_err.csv")
        fields = ['input', 'output', 'error']
        if process.returncode == 0:
            row = [self.file_name, save_dir, 'NONE']
            filename = log_dir_success

        else:
            row = [self.file_name, save_dir, process.stderr.read()]
            filename = log_dir_error

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerow(row)

    def convert_binary_opcode(self):
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
