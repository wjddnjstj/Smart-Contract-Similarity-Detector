import os.path
import r2pipe
import json


def bin2asm(config):
    bin_dir = config['BIN_REPO_DIR']
    asm_dir = config['ASM_REPO_DIR']
    for proj in os.listdir(bin_dir):
        proj_dir = os.path.join(bin_dir, proj)
        for file in os.listdir(proj_dir):
            file_dir = os.path.join(proj_dir, file)
            r2 = r2pipe.open(file_dir)
            r2.cmd('aaa')  # analysis all
            if not os.path.isdir(asm_dir):
                os.mkdir(asm_dir)
            asm_sub_dir = os.path.join(asm_dir, proj)
            if not os.path.isdir(asm_sub_dir):
                os.mkdir(asm_sub_dir)
            try:
                funcs = json.loads(r2.cmd('aflj'))  # list all functions
                for func in funcs:
                    func_name = func['name']
                    r2.cmd('s ' + func_name)  # move to the function start address
                    cfg = r2.cmdj("agj")  # get the control flow graph
                    cnt = 0
                    for graph in cfg:
                        blocks = graph['blocks']
                        for block in blocks:
                            instruction_path = asm_sub_dir + '/%s_%s_%d.txt' % (file, func_name, cnt)
                            cnt += 1
                            f = open(instruction_path, 'w')
                            inst = [i['opcode'] + '\n' for i in block['ops'] if 'opcode' in i]
                            for i in inst:
                                f.write(' ' + i)
                            f.close()
            except Exception as err:
                print(err)
