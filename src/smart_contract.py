import os, re
from pathlib import Path
from typing import Optional
import solcx
import shutil
from solcx.install import get_executable
from solcx.install import install_solc_pragma
import subprocess
import csv


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

    def __init__(self, file_name: str, proj_name: str, proj_dir, config: dict):
        self.file_name = file_name
        self.proj_name = proj_name
        self.proj_dir = proj_dir
        self.config = config

        self.compile_contract()

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

    # Ask Simin why we need two (what the difference is between the two get_solc functions)
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

    def post_compilation(self, save, process):
        self.remove_empty_files(save)
        self.format_opcode(save)
        self.log_message(process, save)

    def exe_command(self, save, cmd):
        solc_compiler = self.get_solc(self.file_name)
        if not os.path.isdir(save):
            os.mkdir(save)
        cmd = '%s %s -o %s %s' % (solc_compiler, cmd, save, self.file_name)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.wait()
        print("Compiling project" + save + "...")
        self.post_compilation(save, process)

    def compile_contract(self):
        if self.proj_dir == self.config['TRAINING_SET']:
            out_dir = os.path.join(self.config['OUT'], self.config['DATA']['TRAINING_DIR'])
            out_dir_opt = os.path.join(self.config['OUT'], self.config['DATA']['TRAINING_DIR_OPT'])
            opcode_dir = os.path.join(self.config['OPCODE'], self.config["DATA"]["TRAINING_DIR"])
            opcode_dir_opt = os.path.join(self.config['OPCODE'], self.config["DATA"]["TRAINING_DIR_OPT"])
            bin_dir = os.path.join(self.config['BIN'], self.config["DATA"]["TRAINING_DIR"])
            bin_dir_opt = os.path.join(self.config['BIN'], self.config["DATA"]["TRAINING_DIR_OPT"])
        else:
            out_dir = os.path.join(self.config['OUT'], self.config['DATA']['TESTING_DIR'])
            out_dir_opt = os.path.join(self.config['OUT'], self.config['DATA']['TESTING_DIR_OPT'])
            opcode_dir = os.path.join(self.config['OPCODE'], self.config["DATA"]["TESTING_DIR"])
            opcode_dir_opt = os.path.join(self.config['OPCODE'], self.config["DATA"]["TESTING_DIR_OPT"])
            bin_dir = os.path.join(self.config['BIN'], self.config["DATA"]["TESTING_DIR"])
            bin_dir_opt = os.path.join(self.config['BIN'], self.config["DATA"]["TESTING_DIR_OPT"])

        save_dir = os.path.join(out_dir, self.proj_name)
        prefix = '--overwrite --opcodes --bin'
        self.exe_command(save_dir, prefix)

        opcode_dir = os.path.join(opcode_dir, self.proj_name)
        if not os.path.isdir(opcode_dir) and len(os.listdir(save_dir)) > 0:
            os.mkdir(opcode_dir)
        self.save_opcode(save_dir, opcode_dir)

        bin_dir = os.path.join(bin_dir, self.proj_name)
        if not os.path.isdir(bin_dir) and len(os.listdir(save_dir)) > 0:
            os.mkdir(bin_dir)
        self.save_bin_code(save_dir, bin_dir)

        save_dir_opt = os.path.join(out_dir_opt, self.proj_name + '_opt')
        prefix_opt = '--overwrite --opcodes --bin --optimize'
        self.exe_command(save_dir_opt, prefix_opt)

        opcode_dir_opt = os.path.join(opcode_dir_opt, self.proj_name + '_opt')
        if not os.path.isdir(opcode_dir_opt) and len(os.listdir(save_dir_opt)) > 0:
            os.mkdir(opcode_dir_opt)
        self.save_opcode(save_dir_opt, opcode_dir_opt)

        bin_dir_opt = os.path.join(bin_dir_opt, self.proj_name + '_opt')
        if not os.path.isdir(bin_dir_opt) and len(os.listdir(save_dir_opt)) > 0:
            os.mkdir(bin_dir_opt)
        self.save_bin_code(save_dir_opt, bin_dir_opt)

    @staticmethod
    def remove_empty_files(dir_name):
        for file in os.listdir(dir_name):
            full_path = os.path.join(dir_name, file)
            if os.stat(full_path).st_size == 0:
                os.remove(full_path)

    def format_opcode(self, dir_name):
        for opcode_filename in os.listdir(dir_name):
            if opcode_filename.endswith("opcode"):
                full_path = os.path.join(dir_name, opcode_filename)
                f = open(full_path, 'r')
                ori_opcodes = f.readlines()
                assert len(ori_opcodes) == 1
                ori_opcodes = ori_opcodes[0]
                f.close()
                formatted_opcode = ''
                has_first_line = False
                for opcode in ori_opcodes.split():
                    if bool(has_first_line):  # line 1 has been written
                        if opcode in self.opcode_set:
                            formatted_opcode += ('\n' + opcode + ' ')
                        else:
                            formatted_opcode += (opcode + ' ')
                    else:  # No line has been written yet
                        formatted_opcode += (opcode + ' ')
                        has_first_line = True
                f = open(full_path, 'w')
                f.write(formatted_opcode)
                f.close()

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

    @staticmethod
    def save_opcode(src_dirname, tar_dirname):
        for file in os.listdir(src_dirname):
            if file.endswith("opcode"):
                source_full_path = os.path.join(src_dirname, file)
                target_full_path = os.path.join(tar_dirname, file)
                shutil.move(source_full_path, target_full_path)
            else:
                continue

    @staticmethod
    def save_bin_code(src_dirname, tar_dirname):
        for file in os.listdir(src_dirname):
            if file.endswith("bin"):
                source_full_path = os.path.join(src_dirname, file)
                target_full_path = os.path.join(tar_dirname, file)
                shutil.move(source_full_path, target_full_path)
            else:
                continue
