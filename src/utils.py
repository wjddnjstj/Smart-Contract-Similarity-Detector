import os.path
import r2pipe
import json


def createCFG():
    cnt = 0
    for file_path in os.listdir("./testing/"):
        binary_path = os.path.join("./testing/", file_path)
        r2 = r2pipe.open(binary_path)
        r2.cmd('aaa')  # analysis all
        try:
            funcs = json.loads(r2.cmd('aflj'))  # list all functions
            func_list = []
            ass_list = []
            for func in funcs:
                func_name = func['name']
                func_list.append(func_name)
                r2.cmd('s ' + func_name)  # move to the function start address
                cfg = r2.cmdj("agj")  # get the control flow graph
                instruction = r2.cmdj("pdj")  # get list of instructions.
                if len(cfg):
                    instance = (instruction, cfg, func_name)  # project name, contract name, function name
                    ass_list.append(instance)
                for graph in cfg:
                    blocks = graph['blocks']
                    for block in blocks:
                        instruction_path = './instructions/%d.txt' % cnt
                        cnt += 1
                        f = open(instruction_path, 'w')
                        inst = [i['opcode'] + '\n' for i in block['ops'] if 'opcode' in i]
                        f.writelines(inst)
                        f.close()
        except:
            print('error')
