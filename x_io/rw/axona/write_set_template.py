import os


set_path = r"K:\ke\sta\data\cumc\20140815-behavior-90\20140815-behavior2-90.set"
target_path = r"K:\ke\dev\neuroscikit\x_io\axona\write_set_blank_dict.py"

def write_set_template(set_path, target_path, dict_type='blank'):
    with open(set_path, 'r') as r:
        with open (target_path, 'w') as w:
            rlines = r.readlines()
            wlines = ['{\n']
            for rline in rlines:
                key, value = rline.split(' ', 1)
                if dict_type=='template':
                    value = value.strip().split(' ')
                    type_hints = []
                    for i in range(len(value)):
                        value[i] = value[i].strip()
                        if value[i].isdigit():
                            type_hints.append('int')
                        elif value[i].replace('.', '', 1).isdigit():
                            type_hints.append('float')
                        else:
                            type_hints.append('str')
                    wlines.append('    "{}": {},\n'.format(key, type_hints))
                elif dict_type=='example':
                    wlines.append("    '{}': '{}',\n".format(key, value.strip()))
                elif dict_type=='blank':
                    wlines.append("    '{}': {},\n".format(key, None))
                else:
                    raise ValueError('dict_type must be one of "template", "example", "blank".')
            wlines.append('}\n')
            w.writelines(wlines)

if __name__ == '__main__':
    write_set_template(set_path, target_path, dict_type='blank')
    print('done')