import r2pipe
import json
import os

if not os.path.isdir('/disk/CM/Project/SmartContract/Similarity/example/opcode/'):
    os.mkdir('/disk/CM/Project/SmartContract/Similarity/example/opcode/')
if not os.path.isdir('/disk/CM/Project/SmartContract/Similarity/example/opcode/instruction'):
    os.mkdir('/disk/CM/Project/SmartContract/Similarity/example/opcode/instruction')

# TODO: write a function to reformat bin and bin-runtime
# TODO: https://blog.positive.com/reversing-evm-bytecode-with-radare2-ab77247e5e53
def convert_bin(proj, file):
    bin_file = f'./{proj}/contract_out/{file}'
    if os.path.isfile(bin_file) and os.access(bin_file, os.R_OK):
        os.system(f'rax2 -s < {bin_file} > {bin_file}.bin')
    else:
	print(f'{bin_file} not found')
		

cnt = 0
for binary_path in os.listdir('/disk/CM/Project/SmartContract/Similarity/example/bin'):
    binary_path = os.path.join('/disk/CM/Project/SmartContract/Similarity/example/bin', binary_path)
    r2 = r2pipe.open(binary_path, flags=['-a', 'evm'])
    r2.cmd('aaa')   # anlysis all
    try:
        funcs = json.loads(r2.cmd('aflj'))   # list all functions
        func_list = []
        ass_list = []
        for func in funcs:
            func_name = func['name']
            func_list.append(func_name)
            r2.cmd('s ' + func_name)    # move to the function start address
            cfg = r2.cmdj("agj")     # get the controal flow graph
            instruction = r2.cmdj("pdj")  # get list of instructions.
            if len(cfg):
                instance = (instruction, cfg, func_name) # project name, contract name, function name
                ass_list.append(instance)
            instruction_path = '/disk/CM/Project/SmartContract/Similarity/example/opcode/instruction/%d.txt' % cnt
            cnt += 1
            f = open(instruction_path, 'w')
            inst = [i['opcode'] + '\n' for i in instruction if 'opcode' in i]
            f.writelines(inst)
            f.close()
    except:
        print('error')

