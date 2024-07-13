import yaml
import glob
import ruamel.yaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq
import os
datasets_list = ['artaxor', 'clipart1k', 'dior', 'fish', 'neu-det', 'uodd']
models = ['b', 's']
for dataset in datasets_list:
    yaml_files = glob.glob(f'./{dataset}/*.yaml')
    for file in yaml_files:
        if "vitl" not in file:
            continue
        for model in models:
            with open(file, 'r') as f:
                yaml = ruamel.yaml.YAML()
                data = yaml.load(f)
            data['DE']['CLASS_PROTOTYPES'] = data['DE']['CLASS_PROTOTYPES'].replace('vitl', f'vit{model}')
            data['DE']['BG_PROTOTYPES'] = data['DE']['BG_PROTOTYPES'].replace('vitl', f'vit{model}')
            if model == 'b':
                data['MODEL']['BACKBONE']['TYPE'] = 'base'
            elif model == 's':
                data['MODEL']['BACKBONE']['TYPE'] = 'small'
            with open(file.replace('vitl', f'vit{model}'), 'w') as f:
                yaml.dump(data, f)
                if isinstance(data, (CommentedMap, CommentedSeq)):
                    f.write("\n")
            print(f"save {file.replace('vitl', f'vit{model}')}")
