import re
import os
import r2pipe
import hashlib


def sha3(data):
    return hashlib.sha3_256(data.encode()).hexdigest()


def normalize(opcode):
    opcode = opcode.replace(' - ', ' + ')
    opcode = re.sub(r'0x[0-9a-f]+', 'CONST', opcode)
    opcode = re.sub(r'\*[0-9]', '*CONST', opcode)
    opcode = re.sub(r' [0-9]', ' CONST', opcode)
    return opcode


def fn2asm(pdf, min_len):
    # check
    if pdf is None:
        return
    if len(pdf[
               'ops']) < min_len:
        return
    if 'invalid' in [op['type'] for op in pdf['ops']]:
        return

    ops = pdf['ops']

    # set label
    labels, scope = {}, [op['offset'] for op in ops]
    assert (None not in scope)
    for i, op in enumerate(ops):
        if op.get('jump') in scope:
            labels.setdefault(op.get('jump'), i)

    # dump output
    output = ''
    for op in ops:
        # add label
        if labels.get(op.get('offset')) is not None:
            output += f'LABEL{labels[op["offset"]]}:\n'
        # add instruction
        if labels.get(op.get('jump')) is not None:
            output += f' {op["type"]} LABEL{labels[op["jump"]]}\n'
        else:
            output += f' {normalize(op["opcode"])}\n'

    return output


def convert(filename, opath):
    r = r2pipe.open(filename)
    r.cmd('aaaa')

    count = 0

    for fn in r.cmdj('aflj'):
        r.cmd(f's {fn["offset"]}')
        asm = fn2asm(r.cmdj('pdfj'), 10)
        if asm:
            uid = sha3(asm)
            asm = f''' .name {fn["name"]}.offset {fn["offset"]:016x}.file {filename} ''' + asm
            with open(opath + uid, 'w') as f:
                f.write(asm)
                count += 1

    return count


def bin2asm(config):
    f_count, b_count = 0, 0

    bin_dir = config['BIN_PROJ_DIR']
    asm_dir = config['ASM_PROJ_DIR']
    if not os.path.isdir(asm_dir):
        os.mkdir(asm_dir)

    if os.path.isdir(bin_dir):
        for f in os.listdir(bin_dir):
            f_count += convert(os.path.join(bin_dir, f), asm_dir)
            b_count += 1
    else:
        print(f'[Error] No such file or directory')

    print(f'[+] Total scan binary: {b_count} => Total generated assembly functions: {f_count}')
